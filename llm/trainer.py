"""FamilyTrackingGRPOTrainer: a thin GRPOTrainer subclass that logs each family's share of
gradient contributions per step (mean |advantage| x count), by hooking
_generate_and_score_completions -- the one place in TRL's GRPOTrainer where the raw
per-example dataset rows (which carry our 'family' column) and the freshly-computed
advantages tensor are both in scope together. Generic in what 'family' means, so it is
unchanged by the task redesign below.

PreferenceEvalCallback: every eval_every steps, draws N_EVAL_FRESH FRESH (random-paraphrase)
prompts per family and evaluates BOTH greedy and training-temperature-sampled decoding,
reporting mean/median u, parse rate, unscaled mean reward, and a collapse indicator (fraction
of samples equal to the modal parsed answer) for each -- independent of the trainer's own
(k-scaled) reward path entirely. The SAMPLED metrics are the primary ones used downstream
(pilot.py / summarize.py); greedy is logged alongside for comparison."""
import random
import statistics
import time
from collections import Counter

import torch
from transformers import TrainerCallback
from trl import GRPOTrainer

from config import FAMILY_A, FAMILY_B, MAX_NEW_TOKENS_EVAL, N_EVAL_FRESH, PARAPHRASES, TEMPERATURE
from reward import clip_u, parse_u, unscaled_reward


class FamilyTrackingGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.family_share_log = []  # list of dicts: {step, family: {mean_abs_adv, count, share}}
        self.first_step_advantages = None  # captured once, for check_determinism.py
        self.first_step_rewards = None

    def _generate_and_score_completions(self, inputs):
        output = super()._generate_and_score_completions(inputs)
        if not self.model.training:
            return output  # only track gradient-contribution share during training
        advantages = output["advantages"]
        A = advantages.shape[0]
        if self.first_step_advantages is None:
            self.first_step_advantages = advantages.detach().clone()
            reward_lists = list(self._logs["rewards"].values())
            if reward_lists:
                self.first_step_rewards = torch.tensor(list(reward_lists[0])[-A:], dtype=torch.float64)
        # `inputs` (len == A here) is ALREADY expanded to one row per completion -- each
        # underlying unique prompt appears as `num_generations` CONSECUTIVE rows with
        # identical family/scale (verified empirically: len(inputs) == B*G, not B). So a
        # direct 1:1 index mapping is correct for the per-row |advantage| aggregation below;
        # grouping for the within-group reward std (further down) needs the actual group
        # size G = self.num_generations, not inferred from A/len(inputs).
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


class PreferenceEvalCallback(TrainerCallback):
    """Every `eval_every` steps: for each family in (A, B), draws `n` FRESH (random
    paraphrase) prompts and evaluates both greedy and training-temperature-sampled decoding,
    reporting mean_u, median_u, parse_rate, and mean_reward (unscaled) for each mode."""

    def __init__(self, tokenizer, paraphrases=PARAPHRASES, families=(FAMILY_A, FAMILY_B),
                 n=N_EVAL_FRESH, eval_every=10, batch_size=50, max_new_tokens=MAX_NEW_TOKENS_EVAL, seed=0):
        self.tokenizer = tokenizer
        self.paraphrases = paraphrases
        self.families = families
        self.n = n
        self.eval_every = eval_every
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.rng = random.Random(seed)
        self.history = []

    def _draw_prompts(self):
        return [self.rng.choice(self.paraphrases) for _ in range(self.n)]

    def _generate_us(self, model, prompt_texts, do_sample):
        device = next(model.parameters()).device
        us = []
        model.eval()
        for i in range(0, len(prompt_texts), self.batch_size):
            chunk = prompt_texts[i : i + self.batch_size]
            rendered = [self.tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                                             add_generation_prompt=True) for p in chunk]
            enc = self.tokenizer(rendered, return_tensors="pt", padding=True, padding_side="left").to(device)
            gen_kwargs = dict(max_new_tokens=self.max_new_tokens,
                               pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
            if do_sample:
                gen_kwargs.update(do_sample=True, temperature=TEMPERATURE)
            else:
                gen_kwargs.update(do_sample=False)
            with torch.no_grad():
                out_ids = model.generate(**enc, **gen_kwargs)
            gen_ids = out_ids[:, enc["input_ids"].shape[1] :]
            gen_texts = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            us.extend(parse_u(t) for t in gen_texts)
        model.train()
        return us

    def _metrics_for(self, us_raw, family):
        parsable = [u for u in us_raw if u is not None]
        parse_rate = len(parsable) / len(us_raw) if us_raw else 0.0
        clipped = [clip_u(u) for u in parsable]
        mean_u = statistics.mean(clipped) if clipped else float("nan")
        median_u = statistics.median(clipped) if clipped else float("nan")
        mean_reward = statistics.mean(unscaled_reward(u, family) for u in us_raw) if us_raw else float("nan")
        # Collapse indicator: fraction of ALL samples (parsable or not) equal to the modal
        # (most common) parsed integer -- a large fraction means the policy has collapsed to
        # repeating a single answer rather than genuinely converging a distribution around it.
        if clipped:
            modal_value, modal_count = Counter(clipped).most_common(1)[0]
        else:
            modal_value, modal_count = None, 0
        modal_fraction = modal_count / len(us_raw) if us_raw else 0.0
        return dict(mean_u=mean_u, median_u=median_u, parse_rate=parse_rate, mean_reward=mean_reward,
                    modal_value=modal_value, modal_fraction=modal_fraction)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step == 0 or state.global_step % self.eval_every != 0:
            return control
        model = model if model is not None else kwargs.get("model")
        t0 = time.time()
        entry = dict(step=int(state.global_step))
        for fam in self.families:
            us_greedy = self._generate_us(model, self._draw_prompts(), do_sample=False)
            us_sampled = self._generate_us(model, self._draw_prompts(), do_sample=True)
            entry[fam] = dict(greedy=self._metrics_for(us_greedy, fam), sampled=self._metrics_for(us_sampled, fam))
        entry["eval_wall_clock_s"] = time.time() - t0
        self.history.append(entry)
        print(f"[eval @ step {state.global_step}] " + "  ".join(
            f"{fam}:greedy_mean_u={entry[fam]['greedy']['mean_u']:.1f}" for fam in self.families))
        return control
