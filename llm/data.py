"""Synthetic task generation: Family A (multiplication), Family B (sum of N 3-digit
numbers). Train/eval sets are drawn from disjoint RNG streams and explicitly deduped
against each other so eval is genuinely held out. Task difficulty is controlled by
config.TASK_VARIANT_A / TASK_VARIANT_B (see config.py's one-line switch)."""
import random

from config import (FAMILY_A, FAMILY_A_VARIANTS, FAMILY_B, FAMILY_B_VARIANTS, N_EVAL_PER_FAMILY,
                     N_TRAIN_PER_FAMILY, TASK_VARIANT_A, TASK_VARIANT_B)

INSTRUCTION_SUFFIX = "End your response with a line of the exact form 'Answer: <int>' where <int> is your final integer answer."


def _digit_range(d):
    lo = 10 ** (d - 1) if d > 1 else 1
    hi = 10 ** d - 1
    return lo, hi


def _prompt_a(x, y):
    return f"What is {x} x {y}? {INSTRUCTION_SUFFIX}"


def _prompt_b(nums):
    return f"What is {' + '.join(str(n) for n in nums)}? {INSTRUCTION_SUFFIX}"


def _make_conversational(prompt_text):
    return [{"role": "user", "content": prompt_text}]


def _gen_family_a(n, rng, variant=TASK_VARIANT_A, exclude=frozenset()):
    params = FAMILY_A_VARIANTS[variant]
    lo_x, hi_x = _digit_range(params["digits_x"])
    lo_y, hi_y = _digit_range(params["digits_y"])
    out = []
    seen = set(exclude)
    while len(out) < n:
        x, y = rng.randint(lo_x, hi_x), rng.randint(lo_y, hi_y)
        key = (x, y)
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(prompt=_make_conversational(_prompt_a(x, y)), true_answer=x * y, family=FAMILY_A))
    return out, seen


def _gen_family_b(n, rng, variant=TASK_VARIANT_B, exclude=frozenset()):
    params = FAMILY_B_VARIANTS[variant]
    lo, hi = _digit_range(params["digits"])
    n_terms = params["n_terms"]
    out = []
    seen = set(exclude)
    while len(out) < n:
        nums = tuple(rng.randint(lo, hi) for _ in range(n_terms))
        if nums in seen:
            continue
        seen.add(nums)
        out.append(dict(prompt=_make_conversational(_prompt_b(nums)), true_answer=sum(nums), family=FAMILY_B))
    return out, seen


def build_family_datasets(seed=0, variant_a=TASK_VARIANT_A, variant_b=TASK_VARIANT_B):
    """Returns (train_a, train_b, eval_a, eval_b), each a list of dicts with keys
    prompt (conversational), true_answer (int), family (str). Eval is generated from a
    disjoint RNG stream and explicitly excludes any tuple already used in train, so held-out
    eval has zero overlap with training examples by construction."""
    rng_train = random.Random(seed)
    rng_eval = random.Random(seed + 1_000_000)

    train_a, seen_a = _gen_family_a(N_TRAIN_PER_FAMILY, rng_train, variant=variant_a)
    train_b, seen_b = _gen_family_b(N_TRAIN_PER_FAMILY, rng_train, variant=variant_b)
    eval_a, _ = _gen_family_a(N_EVAL_PER_FAMILY, rng_eval, variant=variant_a, exclude=seen_a)
    eval_b, _ = _gen_family_b(N_EVAL_PER_FAMILY, rng_eval, variant=variant_b, exclude=seen_b)
    return train_a, train_b, eval_a, eval_b


def build_train_dataset(train_a, train_b, k, seed=0):
    """Attach the per-example 'scale' column (Family B reward x k during training only),
    interleave, and shuffle. Returns a list of dicts ready for datasets.Dataset.from_list."""
    rows = []
    for r in train_a:
        rows.append(dict(r, scale=1.0))
    for r in train_b:
        rows.append(dict(r, scale=float(k)))
    random.Random(seed).shuffle(rows)
    return rows
