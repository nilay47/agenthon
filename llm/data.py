"""Synthetic task generation: Family A (2-digit x 2-digit multiplication), Family B (sum of
four 3-digit numbers). Train/eval sets are drawn from disjoint RNG streams and explicitly
deduped against each other so eval is genuinely held out."""
import random

from config import FAMILY_A, FAMILY_B, N_EVAL_PER_FAMILY, N_TRAIN_PER_FAMILY

INSTRUCTION_SUFFIX = "End your response with a line of the exact form 'Answer: <int>' where <int> is your final integer answer."


def _prompt_a(x, y):
    return f"What is {x} x {y}? {INSTRUCTION_SUFFIX}"


def _prompt_b(a, b, c, d):
    return f"What is {a} + {b} + {c} + {d}? {INSTRUCTION_SUFFIX}"


def _make_conversational(prompt_text):
    return [{"role": "user", "content": prompt_text}]


def _gen_family_a(n, rng, exclude=frozenset()):
    out = []
    seen = set(exclude)
    while len(out) < n:
        x, y = rng.randint(10, 99), rng.randint(10, 99)
        key = (x, y)
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(prompt=_make_conversational(_prompt_a(x, y)), true_answer=x * y, family=FAMILY_A))
    return out, seen


def _gen_family_b(n, rng, exclude=frozenset()):
    out = []
    seen = set(exclude)
    while len(out) < n:
        nums = tuple(rng.randint(100, 999) for _ in range(4))
        if nums in seen:
            continue
        seen.add(nums)
        out.append(dict(prompt=_make_conversational(_prompt_b(*nums)), true_answer=sum(nums), family=FAMILY_B))
    return out, seen


def build_family_datasets(seed=0):
    """Returns (train_a, train_b, eval_a, eval_b), each a list of dicts with keys
    prompt (conversational), true_answer (int), family (str). Eval is generated from a
    disjoint RNG stream and explicitly excludes any (x,y)/(a,b,c,d) tuple already used in
    train, so held-out eval has zero overlap with training examples by construction."""
    rng_train = random.Random(seed)
    rng_eval = random.Random(seed + 1_000_000)

    train_a, seen_a = _gen_family_a(N_TRAIN_PER_FAMILY, rng_train)
    train_b, seen_b = _gen_family_b(N_TRAIN_PER_FAMILY, rng_train)
    eval_a, _ = _gen_family_a(N_EVAL_PER_FAMILY, rng_eval, exclude=seen_a)
    eval_b, _ = _gen_family_b(N_EVAL_PER_FAMILY, rng_eval, exclude=seen_b)
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
