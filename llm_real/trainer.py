"""GraderTrackingGRPOTrainer: a thin GRPOTrainer subclass that logs each grader's share of
gradient contributions per step (mean |advantage| x count), by hooking
_generate_and_score_completions -- same mechanism as llm/trainer.py's FamilyTrackingGRPOTrainer,
generalized to the 'grader' dataset column instead of 'family'.

HeldOutEvalCallback: every eval_every steps, generates SAMPLED (training-temperature)
completions for a FIXED set of real GSM8K held-out problems and reports accuracy, mean
completion length (tokens), mean n_steps, each grader's unscaled mean reward, and each
grader's reward STD across the held-out set (a diagnostic for TRL's reward-normalization
epsilon negligibility, same spirit as llm/'s within-group-std check, just computed over the
whole held-out set rather than per training group)."""
import statistics
import time

import torch
from transformers import TrainerCallback
from trl import GRPOTrainer

from config import MAX_COMPLETION_LENGTH, TEMPERATURE
from reward import eval_metrics_for_completion


class GraderTrackingGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grader_share_log = []  # list of dicts: {step, grader: {mean_abs_adv, count, share}}
        self.first_step_advantages = None  # captured once, for a determinism check
        self.first_step_rewards = None

    def _generate_and_score_completions(self, inputs):
        output = super()._generate_and_score_completions(inputs)
        if not self.model.training:
            return output
        advantages = output["advantages"]
        A = advantages.shape[0]
        if self.first_step_advantages is None:
            self.first_step_advantages = advantages.detach().clone()
            reward_lists = list(self._logs["rewards"].values())
            if reward_lists:
                self.first_step_rewards = torch.tensor(list(reward_lists[0])[-A:], dtype=torch.float64)
        if len(inputs) != A:
            return output  # defensive: skip logging rather than crash on a shape surprise
        graders = [x["grader"] for x in inputs]

        abs_adv = advantages.detach().abs().cpu()
        per_grader = {}
        for grd in sorted(set(graders)):
            idx = [i for i, g in enumerate(graders) if g == grd]
            vals = abs_adv[idx]
            per_grader[grd] = dict(mean_abs_adv=float(vals.mean()), count=len(idx))
        total_contrib = sum(d["mean_abs_adv"] * d["count"] for d in per_grader.values()) + 1e-12
        for grd, d in per_grader.items():
            d["contribution"] = d["mean_abs_adv"] * d["count"]
            d["share"] = d["contribution"] / total_contrib

        self.grader_share_log.append(dict(step=int(self.state.global_step), **per_grader))
        return output


def _completion_token_length(row_ids, eos_token_id):
    """Number of generated tokens strictly before the first EOS (or the full padded length if
    no EOS was generated, i.e. the completion was truncated at max_completion_length)."""
    eos_positions = (row_ids == eos_token_id).nonzero(as_tuple=True)[0]
    if len(eos_positions) > 0:
        return int(eos_positions[0].item())
    return int(row_ids.shape[0])


class HeldOutEvalCallback(TrainerCallback):
    """Every `eval_every` steps: SAMPLED (training-temperature) decoding on a FIXED list of
    real held-out GSM8K problems (`held_out`, each {prompt, gold}), scored under BOTH graders."""

    def __init__(self, tokenizer, held_out, eval_every=20, batch_size=25,
                 max_new_tokens=MAX_COMPLETION_LENGTH, temperature=TEMPERATURE):
        self.tokenizer = tokenizer
        self.held_out = held_out
        self.eval_every = eval_every
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.history = []

    def run_eval(self, model):
        """Runs the held-out eval directly (no step-count gating) -- used both by the
        callback's on_step_end hook and by baseline_eval.py's untrained-model check."""
        device = next(model.parameters()).device
        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id or eos_token_id
        was_training = model.training
        model.eval()
        per_example = []
        for i in range(0, len(self.held_out), self.batch_size):
            chunk = self.held_out[i : i + self.batch_size]
            rendered = [self.tokenizer.apply_chat_template([{"role": "user", "content": ex["prompt"]}],
                                                             tokenize=False, add_generation_prompt=True)
                        for ex in chunk]
            enc = self.tokenizer(rendered, return_tensors="pt", padding=True, padding_side="left").to(device)
            with torch.no_grad():
                out_ids = model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=True,
                                          temperature=self.temperature, pad_token_id=pad_token_id)
            gen_ids = out_ids[:, enc["input_ids"].shape[1] :]
            gen_texts = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            for row_ids, text, ex in zip(gen_ids, gen_texts, chunk):
                n_tokens = _completion_token_length(row_ids, eos_token_id)
                per_example.append(eval_metrics_for_completion(text, ex["gold"], n_tokens))
        if was_training:
            model.train()
        return per_example

    @staticmethod
    def summarize(per_example):
        if not per_example:
            return dict(n=0, accuracy=float("nan"), mean_n_tokens=float("nan"), mean_n_steps=float("nan"),
                        reward_a=dict(mean=float("nan"), std=float("nan")),
                        reward_b=dict(mean=float("nan"), std=float("nan")))
        accuracy = statistics.mean(float(m["correct"]) for m in per_example)
        mean_n_tokens = statistics.mean(m["n_tokens"] for m in per_example)
        mean_n_steps = statistics.mean(m["n_steps"] for m in per_example)
        rewards_a = [m["reward_a"] for m in per_example]
        rewards_b = [m["reward_b"] for m in per_example]

        def mean_std(vals):
            mean = statistics.mean(vals)
            std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
            return dict(mean=mean, std=std)

        return dict(n=len(per_example), accuracy=accuracy, mean_n_tokens=mean_n_tokens, mean_n_steps=mean_n_steps,
                    reward_a=mean_std(rewards_a), reward_b=mean_std(rewards_b))

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step == 0 or state.global_step % self.eval_every != 0:
            return control
        model = model if model is not None else kwargs.get("model")
        t0 = time.time()
        per_example = self.run_eval(model)
        summary = self.summarize(per_example)
        entry = dict(step=int(state.global_step), eval_wall_clock_s=time.time() - t0, **summary)
        self.history.append(entry)
        print(f"[eval @ step {state.global_step}] accuracy={summary['accuracy']:.2f}  "
              f"mean_n_tokens={summary['mean_n_tokens']:.1f}  mean_n_steps={summary['mean_n_steps']:.1f}  "
              f"reward_a={summary['reward_a']['mean']:.2f}(std={summary['reward_a']['std']:.2f})  "
              f"reward_b={summary['reward_b']['mean']:.2f}(std={summary['reward_b']['std']:.2f})")
        return control
