"""Conflicting-preference task data: ONE fixed prompt (a few paraphrases, see
config.PARAPHRASES) for every training example. Each example carries a HIDDEN family label
A/B (never shown to the model) that only the reward function and the sampler see."""
import random

from config import FAMILY_A, FAMILY_B, N_TRAIN_PER_FAMILY, PARAPHRASES


def _make_conversational(prompt_text):
    return [{"role": "user", "content": prompt_text}]


def _gen_family_rows(n, rng, family):
    return [dict(prompt=_make_conversational(rng.choice(PARAPHRASES)), family=family) for _ in range(n)]


def build_family_datasets(seed=0):
    """Returns (train_a, train_b): N_TRAIN_PER_FAMILY rows each, prompt text drawn uniformly
    from PARAPHRASES. No fixed held-out eval set is built here -- PreferenceEvalCallback and
    baseline_eval.py draw FRESH paraphrase samples directly at eval time instead, since this
    task has no varying input to generalize to (the prompt is semantically constant); what's
    being measured is the policy's output DISTRIBUTION, not accuracy on held-out inputs."""
    rng = random.Random(seed)
    train_a = _gen_family_rows(N_TRAIN_PER_FAMILY, rng, FAMILY_A)
    train_b = _gen_family_rows(N_TRAIN_PER_FAMILY, rng, FAMILY_B)
    return train_a, train_b


def build_train_dataset(train_a, train_b, k, seed=0):
    """Attach the per-example 'scale' column (family B's reward x k during training only),
    interleave, and shuffle. Returns a list of dicts ready for datasets.Dataset.from_list."""
    rows = []
    for r in train_a:
        rows.append(dict(r, scale=1.0))
    for r in train_b:
        rows.append(dict(r, scale=float(k)))
    random.Random(seed).shuffle(rows)
    return rows
