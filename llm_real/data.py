"""GSM8K loading, grader assignment, and held-out eval-set construction.

Grader assignment is per-example, random, 50/50, and NEVER shown to the model (same
"hidden family label" design as llm/data.py's family A/B split) -- it only determines which
reward function scores that example during training."""
import random
import re

import config as C

_GOLD_RE = re.compile(r"####\s*(-?[\d,]+)")


def parse_gold_answer(answer_field):
    """GSM8K's own answer field always ends with '#### <number>' -- this is the ground-truth
    parse, not the model-completion parse (see reward.parse_final_answer for that)."""
    m = _GOLD_RE.search(answer_field)
    if m is None:
        raise ValueError(f"GSM8K answer field missing '#### <number>': {answer_field!r}")
    return int(m.group(1).replace(",", ""))


def format_prompt(question):
    return C.PROMPT_TEMPLATE.format(question=question)


def load_gsm8k():
    from datasets import load_dataset

    ds = load_dataset(C.DATASET_NAME, C.DATASET_CONFIG)
    return ds["train"], ds["test"]


def build_train_rows(train_split, seed):
    """One row per GSM8K train-split example: a chat-formatted prompt, the parsed gold
    integer answer, and a randomly (seeded) assigned grader label. Uses Python's stdlib
    `random` (not numpy) for a plain per-row coin flip -- deterministic given `seed`."""
    rng = random.Random(seed)
    rows = []
    for ex in train_split:
        grader = C.GRADER_A if rng.random() < 0.5 else C.GRADER_B
        rows.append(dict(prompt=[{"role": "user", "content": format_prompt(ex["question"])}],
                          gold=parse_gold_answer(ex["answer"]), grader=grader))
    return rows


def build_held_out_set(test_split, n, seed=C.EVAL_SPLIT_SEED):
    """A reproducible, fixed-size slice of GSM8K's TEST split (never trained on). Returns
    plain dicts with a chat-formatted prompt and the gold integer answer (no grader label --
    eval scores every held-out completion under BOTH graders, see trainer.eval_metrics)."""
    rng = random.Random(seed)
    idx = list(range(len(test_split)))
    rng.shuffle(idx)
    idx = idx[:n]
    return [dict(prompt=format_prompt(test_split[i]["question"]), gold=parse_gold_answer(test_split[i]["answer"]))
            for i in idx]


def periodic_eval_subset(held_out_set, n=C.N_EVAL_PERIODIC):
    """The SAME fixed n-problem subset of the full reserved held-out set every time (the first
    n, by construction of build_held_out_set's already-shuffled order) -- used for the cheaper
    every-EVAL_EVERY-steps in-training eval, so the training curve isn't also noisy from a
    changing eval set. The full reserved set (all N_TEST_RESERVED) is used once for the Step 1
    baseline check, which wants more statistical power and only runs once."""
    return held_out_set[:n]
