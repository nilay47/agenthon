"""CPU-only, GPU-free unit tests for llm_real's reward parsing and data logic. The GRPOTrainer
integration itself is validated separately via an end-to-end dry run against a tiny
random-weight Qwen2.5 checkpoint (yujiepan/qwen2.5-tiny-random), not repeated here."""
import json
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


def test_parse_final_answer_hash_tier():
    assert parse_final_answer("reasoning...\n#### 18") == 18
    assert parse_final_answer("#### 1,024") == 1024


def test_parse_final_answer_boxed_tier():
    # no '####' marker -> falls back to \boxed{...}
    assert parse_final_answer("The answer is \\boxed{42}.") == 42
    assert parse_final_answer("reasoning\n\\boxed{1,200}\nmore text") == 1200


def test_parse_final_answer_last_number_tier():
    # no '####' and no \boxed -> falls back to the LAST number anywhere in the text
    assert parse_final_answer("I think it's 7, no wait, 12, final answer 18") == 18
    assert parse_final_answer("Natalia sold 48 clips, half as many is 24, total 72") == 72


def test_parse_final_answer_tier_precedence():
    # '####' wins even if \boxed{} or a later number is also present
    assert parse_final_answer("\\boxed{99}\n#### 42\nextra 7") == 42
    # \boxed{} wins over a later bare number when no '####' is present
    assert parse_final_answer("\\boxed{42}\nactually 7") == 42


def test_parse_final_answer_truly_unparsable():
    assert parse_final_answer("no number anywhere in this text") is None


def test_count_reasoning_steps():
    text = "Step one.\nStep two.\n\nStep three.\n#### 7"
    assert count_reasoning_steps(text) == 3  # blank line doesn't count
    assert count_reasoning_steps("#### 7") == 0  # nothing before the marker
    assert count_reasoning_steps("just rambling, no marker at all") == 1


def test_grader_a_reward_reasoning_rubric_token_based():
    # attempt 5: token-based, saturates at MAX_COMPLETION_LENGTH (not a trivially-reached
    # line-count cap, which gave no gradient once the model naturally exceeded it).
    cap = C.MAX_COMPLETION_LENGTH
    assert grader_a_reward(parsed=10, gold=10, n_tokens=cap) == 10.0
    assert grader_a_reward(parsed=10, gold=10, n_tokens=cap * 2) == 10.0  # saturates, doesn't exceed
    assert abs(grader_a_reward(parsed=10, gold=10, n_tokens=cap // 2) - 5.0) < 1e-9  # halfway -> half credit
    assert grader_a_reward(parsed=10, gold=10, n_tokens=0) == 0.0  # 0 tokens -> zero credit
    assert grader_a_reward(parsed=9, gold=10, n_tokens=cap) == 0.0  # incorrect -> zero regardless of length
    assert grader_a_reward(parsed=None, gold=10, n_tokens=cap) == 0.0  # unparsable -> zero


def test_grader_b_reward_terse_verifier_smooth():
    # smooth linear taper: correct * max(0, 1 - n_tokens/TERSE_SMOOTH_DENOM)
    assert grader_b_reward(parsed=10, gold=10, n_tokens=0) == 1.0  # shortest possible -> full credit
    halfway = C.TERSE_SMOOTH_DENOM // 2
    assert abs(grader_b_reward(parsed=10, gold=10, n_tokens=halfway) - 0.5) < 1e-9  # halfway -> half credit
    assert grader_b_reward(parsed=10, gold=10, n_tokens=C.TERSE_SMOOTH_DENOM) == 0.0  # at the denom -> 0
    assert grader_b_reward(parsed=10, gold=10, n_tokens=C.TERSE_SMOOTH_DENOM * 2) == 0.0  # beyond -> still 0, not negative
    assert grader_b_reward(parsed=9, gold=10, n_tokens=0) == 0.0  # incorrect -> zero even if terse
    assert grader_b_reward(parsed=None, gold=10, n_tokens=0) == 0.0


def test_train_reward_fn_routes_by_grader():
    comps = [[{"role": "assistant", "content": "a\nb\nc\nd\ne\nf\n#### 7"}],  # grader A, correct
             [{"role": "assistant", "content": "#### 7"}]]  # grader B, short + correct
    out = train_reward_fn(prompts=[None] * 2, completions=comps, completion_ids=[[0] * 3, [0] * 3],
                           gold=[7, 7], grader=[C.GRADER_A, C.GRADER_B])
    assert abs(out[0] - 10.0 * (3 / C.MAX_COMPLETION_LENGTH)) < 1e-9  # grader A: token-based now
    assert abs(out[1] - (1.0 - 3 / C.TERSE_SMOOTH_DENOM)) < 1e-9


def test_train_reward_fn_unparsable_is_zero_for_both_graders():
    comps = [[{"role": "assistant", "content": "I don't know."}]] * 2
    out = train_reward_fn(prompts=[None] * 2, completions=comps, completion_ids=[[0] * 2, [0] * 2],
                           gold=[7, 7], grader=[C.GRADER_A, C.GRADER_B])
    assert out == [0.0, 0.0]


def test_eval_metrics_for_completion_scores_both_graders():
    text = "a\nb\nc\nd\ne\nf\n#### 7"
    m = eval_metrics_for_completion(text, gold=7, n_tokens=30)
    assert m["correct"] is True
    assert m["n_steps"] == 6  # still computed as a diagnostic, just not fed into grader A anymore
    assert abs(m["reward_a"] - 10.0 * (30 / C.MAX_COMPLETION_LENGTH)) < 1e-9
    assert abs(m["reward_b"] - (1.0 - 30 / C.TERSE_SMOOTH_DENOM)) < 1e-9


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


def test_v4_config_values():
    # v4/attempt 5: raised LR, gate reverted to a Dr.GRPO>=1.15x-GRPO framing, no more
    # REASONING_STEPS_CAP (grader A is token-based now).
    assert C.MAX_STEPS == 80
    assert C.MAX_COMPLETION_LENGTH == 320
    assert C.TERSE_SMOOTH_DENOM == 400
    assert C.LR == 5e-5
    assert C.BASELINE_MIN_ACCURACY == 0.20
    assert C.N_GROUP_STD_PROMPTS > 0
    assert C.MODEL_NAME == "Qwen/Qwen2.5-0.5B-Instruct"
    assert C.MODEL_NAME_FALLBACK == "Qwen/Qwen2.5-1.5B-Instruct"
    assert C.PILOT_GATE_MIN_DRGRPO_TO_GRPO_LENGTH_RATIO == 1.15
    assert not hasattr(C, "REASONING_STEPS_CAP")
    assert C.PROMPT_TEMPLATE.format(question="Q") == "Q\n\nSolve step by step, then give the final answer as '#### <number>'."


def test_attempts_log_records_and_preserves_prior_entries(tmp_path):
    from attempts_log import config_snapshot, record_attempt

    out_dir = str(tmp_path)
    e1 = record_attempt(out_dir, stage="baseline", config=config_snapshot(C), outcome=dict(gate_passed=False))
    assert e1["attempt_number"] == 1
    assert e1["stage"] == "baseline"
    assert e1["outcome"]["gate_passed"] is False
    assert "prediction" in e1 and len(e1["prediction"]) > 0

    e2 = record_attempt(out_dir, stage="pilot", config=config_snapshot(C), outcome=dict(passed=True))
    assert e2["attempt_number"] == 2  # appended, not overwritten

    with open(os.path.join(out_dir, "attempts.json")) as f:
        log = json.load(f)
    assert len(log) == 2
    assert log[0]["outcome"]["gate_passed"] is False  # first entry untouched by the second write


def test_attempts_log_config_snapshot_model_name_override():
    from attempts_log import config_snapshot

    snap = config_snapshot(C, model_name=C.MODEL_NAME_FALLBACK)
    assert snap["model_name"] == C.MODEL_NAME_FALLBACK
    assert snap["model_name_fallback"] == C.MODEL_NAME_FALLBACK  # the field name itself is unaffected


def test_attempts_log_config_snapshot_tracks_lr_and_new_gate():
    from attempts_log import config_snapshot

    snap = config_snapshot(C)
    assert snap["lr"] == C.LR
    assert snap["pilot_gate_min_drgrpo_to_grpo_length_ratio"] == C.PILOT_GATE_MIN_DRGRPO_TO_GRPO_LENGTH_RATIO
    assert "reasoning_steps_cap" not in snap
