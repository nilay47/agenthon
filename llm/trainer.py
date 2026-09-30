"""FamilyTrackingGRPOTrainer: a thin GRPOTrainer subclass that additionally logs each
family's share of gradient contributions per step (mean |advantage| x count, matching the
spec exactly), by hooking _generate_and_score_completions -- the one place in TRL's
GRPOTrainer where the raw per-example dataset rows (which carry our 'family' column) and
the freshly-computed advantages tensor are both in scope together. Also defines
FamilyEvalCallback, which runs a greedy, UNSCALED-reward eval on each family's held-out set
every EVAL_EVERY steps (bypassing the trainer's own reward path entirely, since that path
always uses the training-time k-scaled reward function)."""
import time

import torch
from transformers import TrainerCallback
from trl import GRPOTrainer

from reward import unscaled_reward_batch


class FamilyTrackingGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.family_share_log = []  # list of dicts: {step, family: {mean_abs_adv, count, share}}

    def _generate_and_score_completions(self, inputs):
        output = super()._generate_and_score_completions(inputs)
        if not self.model.training:
            return output  # only track gradient-contribution share during training
        advantages = output["advantages"]
        A = advantages.shape[0]
        # `inputs` (len == A here) is ALREADY expanded to one row per completion -- each
        # underlying unique prompt appears as `num_generations` CONSECUTIVE rows with
        # identical family/true_answer/scale (verified empirically: len(inputs) == B*G,
        # not B). So a direct 1:1 index mapping is correct for the per-row |advantage|
        # aggregation below; grouping for the within-group reward std (further down) needs
        # the actual group size G = self.num_generations, not inferred from A/len(inputs).
        if len(inputs) != A:
            return output  # defensive: skip logging rather than crash the run on a shape surprise
        families = [x["family"] for x in inputs]

        abs_adv = advantages.detach().abs().cpu()
        per_family = {}
        for fam in sorted(set(families)):
            idx = [i for i, f in enumerate(families) if f == fam]
            vals = abs_adv[idx]
            per_family[fam] = dict(mean_abs_adv=float(vals.mean()), count=len(idx))
        for fam, d in per_family.items():
            d["contribution"] = d["mean_abs_adv"] * d["count"]
        total_contrib = sum(d["contribution"] for d in per_family.values()) + 1e-12
        for fam, d in per_family.items():
            d["share"] = d["contribution"] / total_contrib

        # Also recover the raw (pre-group-normalization) within-group reward std per family,
        # for the sigma-proportional sampler -- pulled from self._logs["rewards"][<reward_fn_name>]
        # (a maxlen=generation_batch_size deque the base class just extended with exactly A new
        # raw-reward values, in the SAME row order as `inputs`/`advantages`). Rows are grouped in
        # consecutive blocks of G = num_generations (same convention TRL itself uses internally,
        # e.g. `rewards.view(-1, num_generations)`).
        G = self.num_generations
        reward_lists = list(self._logs["rewards"].values())
        if reward_lists and G > 1 and A % G == 0:
            deque_obj = reward_lists[0]
            raw = torch.tensor(list(deque_obj)[-A:], dtype=torch.float64).view(-1, G)  # (A//G, G)
            group_std = raw.std(dim=1, unbiased=False)
            group_family = families[::G]  # one family label per group (consecutive-block convention)
            for fam in per_family:
                idx_g = [i for i, f in enumerate(group_family) if f == fam]
                if idx_g:
                    per_family[fam]["group_std"] = float(group_std[idx_g].mean())

        self.family_share_log.append(dict(step=int(self.state.global_step), **per_family))
        return output


class FamilyEvalCallback(TrainerCallback):
    """Greedy decoding on each family's held-out set every `eval_every` steps, unscaled
    graded reward. Independent of the trainer's own (k-scaled, sampled) reward path."""

    def __init__(self, tokenizer, eval_sets, eval_every=25, max_new_tokens=64, batch_size=50):
        self.tokenizer = tokenizer
        self.eval_sets = eval_sets  # dict: family -> list of {prompt, true_answer}
        self.eval_every = eval_every
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.history = []  # list of {step, wall_clock, family: mean_reward}

    def _greedy_eval_family(self, model, rows):
        device = next(model.parameters()).device
        rewards = []
        model.eval()
        for i in range(0, len(rows), self.batch_size):
            chunk = rows[i : i + self.batch_size]
            texts = [self.tokenizer.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True) for r in chunk]
            enc = self.tokenizer(texts, return_tensors="pt", padding=True, padding_side="left").to(device)
            with torch.no_grad():
                out_ids = model.generate(
                    **enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                )
            gen_ids = out_ids[:, enc["input_ids"].shape[1] :]
            gen_texts = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            true_answers = [r["true_answer"] for r in chunk]
            rewards.extend(unscaled_reward_batch(gen_texts, true_answers))
        model.train()
        return sum(rewards) / len(rewards)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step == 0 or state.global_step % self.eval_every != 0:
            return control
        model = model if model is not None else kwargs.get("model")
        t0 = time.time()
        entry = dict(step=int(state.global_step))
        for fam, rows in self.eval_sets.items():
            entry[fam] = self._greedy_eval_family(model, rows)
        entry["eval_wall_clock_s"] = time.time() - t0
        self.history.append(entry)
        print(f"[eval @ step {state.global_step}] " + "  ".join(f"{k}={v:.4f}" for k, v in entry.items() if k not in ("step",)))
        return control
