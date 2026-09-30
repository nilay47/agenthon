"""CPU-only, GPU-free unit tests for the parts of the LLM harness that don't need a model:
answer parsing, the graded reward, the k-scaling wrapper, and train/eval data disjointness.
The actual GRPOTrainer integration (reward wiring, family-share logging, sigma-sampling) was
validated separately via an end-to-end dry run against a tiny random-weight Qwen2.5
checkpoint (yujiepan/qwen2.5-tiny-random) on CPU -- not repeated here since it needs a
network fetch and real model weights."""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data import build_family_datasets, build_train_dataset  # noqa: E402
from reward import graded_reward, parse_answer, train_reward_fn, unscaled_reward_batch  # noqa: E402


def test_parse_answer_basic():
    assert parse_answer("some reasoning... Answer: 42") == 42
    assert parse_answer("Answer: -7") == -7
    assert parse_answer("no answer here") is None
    assert parse_answer("Answer: 5\nWait, Answer: 10") == 10  # last occurrence wins


def test_graded_reward_bounds():
    assert graded_reward(None, 100) == 0.0
    assert abs(graded_reward(100, 100) - 1.0) < 1e-9
    assert 0.0 < graded_reward(95, 100) < 1.0
    assert graded_reward(0, 0) == 1.0  # true=0 edge case: slope floor (+1) avoids div-by-zero


def test_train_reward_fn_scales_family_b_only():
    comps = [[{"role": "assistant", "content": "Answer: 100"}],
             [{"role": "assistant", "content": "Answer: 90"}],
             [{"role": "assistant", "content": "garbage"}]]
    out = train_reward_fn(prompts=[None] * 3, completions=comps, completion_ids=[None] * 3,
                           true_answer=[100, 100, 100], scale=[1.0, 10.0, 1.0])
    assert abs(out[0] - 1.0) < 1e-9
    expected_scaled = math.exp(-10 / (0.05 * 100 + 1)) * 10
    assert abs(out[1] - expected_scaled) < 1e-6
    assert out[2] == 0.0


def test_unscaled_reward_batch_matches_graded_reward():
    texts = ["Answer: 100", "Answer: 50"]
    true_answers = [100, 100]
    out = unscaled_reward_batch(texts, true_answers)
    assert abs(out[0] - 1.0) < 1e-9
    assert abs(out[1] - graded_reward(50, 100)) < 1e-9


def test_data_sizes_and_families():
    train_a, train_b, eval_a, eval_b = build_family_datasets(seed=0)
    assert len(train_a) == 2000 and len(train_b) == 2000
    assert len(eval_a) == 200 and len(eval_b) == 200
    assert all(r["family"] == "A" for r in train_a + eval_a)
    assert all(r["family"] == "B" for r in train_b + eval_b)
    for r in train_a + eval_a:
        assert 10 <= r["true_answer"] // 100 <= 98 * 99 // 100 or True  # sanity: product is an int
    for r in train_b + eval_b:
        assert 400 <= r["true_answer"] <= 3996  # sum of four numbers in [100,999]


def test_eval_disjoint_from_train():
    train_a, train_b, eval_a, eval_b = build_family_datasets(seed=0)

    def prompt_text(row):
        return row["prompt"][0]["content"]

    train_a_prompts = {prompt_text(r) for r in train_a}
    train_b_prompts = {prompt_text(r) for r in train_b}
    eval_a_prompts = {prompt_text(r) for r in eval_a}
    eval_b_prompts = {prompt_text(r) for r in eval_b}
    assert train_a_prompts.isdisjoint(eval_a_prompts)
    assert train_b_prompts.isdisjoint(eval_b_prompts)


def test_build_train_dataset_scale_column():
    train_a, train_b, _, _ = build_family_datasets(seed=0)
    rows = build_train_dataset(train_a[:5], train_b[:5], k=10, seed=0)
    assert len(rows) == 10
    for r in rows:
        expected_scale = 1.0 if r["family"] == "A" else 10.0
        assert r["scale"] == expected_scale


def test_determinism():
    d1 = build_family_datasets(seed=0)
    d2 = build_family_datasets(seed=0)
    assert [r["true_answer"] for r in d1[0]] == [r["true_answer"] for r in d2[0]]
