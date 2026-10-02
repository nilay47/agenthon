"""CPU-only, GPU-free unit tests for llm_real's reward parsing and data logic. The GRPOTrainer
integration itself is validated separately via an end-to-end dry run against a tiny
random-weight Qwen2.5 checkpoint (yujiepan/qwen2.5-tiny-random), not repeated here."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config as C  # noqa: E402
from data import build_held_out_set, build_train_rows, parse_gold_answer, periodic_eval_subset  # noqa: E402
from reward import (  # noqa: E402
    count_reasoning_steps,
    eval_metrics_for_completion,
    grader_a_reward,
    grader_b_reward,
    parse_final_answer,
    train_reward_fn,
)


def test_parse_gold_answer():
    assert parse_gold_answer("Some reasoning.\n#### 42") == 42
    assert parse_gold_answer("Negative case.\n#### -5") == -5
    assert parse_gold_answer("Thousands separator.\n#### 1,200") == 1200


def test_parse_final_answer():
    assert parse_final_answer("reasoning...\n#### 18") == 18
    assert parse_final_answer("no marker here") is None
    assert parse_final_answer("#### 1,024") == 1024
    assert parse_final_answer("#### not-a-number") is None


def test_count_reasoning_steps():
    text = "Step one.\nStep two.\n\nStep three.\n#### 7"
    assert count_reasoning_steps(text) == 3  # blank line doesn't count
    assert count_reasoning_steps("#### 7") == 0  # nothing before the marker
    assert count_reasoning_steps("just rambling, no marker at all") == 1


def test_grader_a_reward_reasoning_rubric():
    # correct, 6+ steps -> full score
    assert grader_a_reward(parsed=10, gold=10, n_steps=6) == 10.0
    assert grader_a_reward(parsed=10, gold=10, n_steps=12) == 10.0  # saturates, doesn't exceed
    # correct, 3 steps -> half credit
    assert abs(grader_a_reward(parsed=10, gold=10, n_steps=3) - 5.0) < 1e-9
    # correct, 0 steps -> zero credit despite being correct
    assert grader_a_reward(parsed=10, gold=10, n_steps=0) == 0.0
    # incorrect -> zero regardless of steps
    assert grader_a_reward(parsed=9, gold=10, n_steps=6) == 0.0
    # unparsable -> zero
    assert grader_a_reward(parsed=None, gold=10, n_steps=6) == 0.0


def test_grader_b_reward_terse_verifier():
    assert grader_b_reward(parsed=10, gold=10, n_tokens=40) == 1.0  # exactly at the cap
    assert grader_b_reward(parsed=10, gold=10, n_tokens=41) == 0.0  # one token over -> hard cliff
    assert grader_b_reward(parsed=10, gold=10, n_tokens=5) == 1.0
    assert grader_b_reward(parsed=9, gold=10, n_tokens=5) == 0.0  # incorrect -> zero even if terse
    assert grader_b_reward(parsed=None, gold=10, n_tokens=5) == 0.0


def test_train_reward_fn_routes_by_grader():
    comps = [[{"role": "assistant", "content": "a\nb\nc\nd\ne\nf\n#### 7"}],  # grader A, 6 steps, correct
             [{"role": "assistant", "content": "#### 7"}]]  # grader B, 0 tokens of "reasoning", correct, terse
    out = train_reward_fn(prompts=[None] * 2, completions=comps, completion_ids=[[0] * 3, [0] * 3],
                           gold=[7, 7], grader=[C.GRADER_A, C.GRADER_B])
    assert out[0] == 10.0
    assert out[1] == 1.0


def test_train_reward_fn_unparsable_is_zero_for_both_graders():
    comps = [[{"role": "assistant", "content": "I don't know."}]] * 2
    out = train_reward_fn(prompts=[None] * 2, completions=comps, completion_ids=[[0] * 2, [0] * 2],
                           gold=[7, 7], grader=[C.GRADER_A, C.GRADER_B])
    assert out == [0.0, 0.0]


def test_eval_metrics_for_completion_scores_both_graders():
    text = "a\nb\nc\nd\ne\nf\n#### 7"
    m = eval_metrics_for_completion(text, gold=7, n_tokens=30)
    assert m["correct"] is True
    assert m["n_steps"] == 6
    assert m["reward_a"] == 10.0
    assert m["reward_b"] == 1.0  # 30 <= 40


def test_build_train_rows_grader_assignment_is_seeded_and_balanced():
    fake_train = [dict(question=f"q{i}", answer=f"reasoning\n#### {i}") for i in range(200)]
    rows1 = build_train_rows(fake_train, seed=0)
    rows2 = build_train_rows(fake_train, seed=0)
    assert [r["grader"] for r in rows1] == [r["grader"] for r in rows2]  # deterministic
    graders = [r["grader"] for r in rows1]
    frac_a = graders.count(C.GRADER_A) / len(graders)
    assert 0.35 < frac_a < 0.65  # roughly 50/50 over 200 examples
    for r in rows1:
        assert r["prompt"][0]["content"].startswith("q")
        assert "grader" not in r["prompt"][0]["content"].lower()  # never leaked into the prompt text


def test_build_held_out_set_and_periodic_subset():
    fake_test = [dict(question=f"q{i}", answer=f"reasoning\n#### {i}") for i in range(300)]
    held_out = build_held_out_set(fake_test, n=200, seed=C.EVAL_SPLIT_SEED)
    assert len(held_out) == 200
    assert len(set(d["gold"] for d in held_out)) == 200  # no duplicate problems
    subset = periodic_eval_subset(held_out, n=100)
    assert subset == held_out[:100]
    # reproducible across calls
    held_out2 = build_held_out_set(fake_test, n=200, seed=C.EVAL_SPLIT_SEED)
    assert held_out == held_out2


def test_micro_batch_and_accum():
    micro, accum = C.micro_batch_and_accum(C.PROMPTS_PER_STEP, C.G)
    assert micro * accum == C.PROMPTS_PER_STEP * C.G
    assert micro % C.G == 0
    assert micro <= C.MICRO_BATCH_COMPLETIONS
