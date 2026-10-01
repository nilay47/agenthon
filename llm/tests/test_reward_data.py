"""CPU-only, GPU-free unit tests for the parts of the LLM harness that don't need a model:
integer parsing, the conflicting-preference reward, the k-scaling wrapper, and data
generation. The actual GRPOTrainer integration (reward wiring, family-share logging,
sigma-sampling, the eval callback, the full sweep + pass-check + summary pipeline) was
validated separately via end-to-end dry runs against a tiny random-weight Qwen2.5 checkpoint
(yujiepan/qwen2.5-tiny-random) on CPU -- not repeated here since it needs a network fetch and
real model weights."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config as C  # noqa: E402
from data import build_family_datasets, build_train_dataset  # noqa: E402
from reward import clip_u, parse_u, train_reward_fn, unscaled_reward, unscaled_reward_batch  # noqa: E402


def test_parse_u_basic():
    assert parse_u("42") == 42
    assert parse_u("I'll pick 73.") == 73
    assert parse_u("no number here") is None
    assert parse_u("-5") == -5


def test_clip_u():
    assert clip_u(150) == 100
    assert clip_u(-5) == 0
    assert clip_u(50) == 50


def test_unscaled_reward():
    assert unscaled_reward(None, C.FAMILY_A) == C.UNPARSABLE_PENALTY
    assert unscaled_reward(None, C.FAMILY_B) == C.UNPARSABLE_PENALTY
    assert abs(unscaled_reward(80, C.FAMILY_A) - 0.0) < 1e-9  # A's target
    assert abs(unscaled_reward(20, C.FAMILY_B) - 0.0) < 1e-9  # B's target
    assert unscaled_reward(0, C.FAMILY_A) < 0
    # 100 is closer to A's target (80) than 50 is -> less negative (better) reward
    assert unscaled_reward(100, C.FAMILY_A) > unscaled_reward(50, C.FAMILY_A)
    # 20 is further from A's target (80) than 50 is -> more negative (worse) reward
    assert unscaled_reward(20, C.FAMILY_A) < unscaled_reward(50, C.FAMILY_A)


def test_train_reward_fn_scales_family_b_entirely_including_unparsable():
    comps = [[{"role": "assistant", "content": "80"}],   # family A, at target, scale 1
             [{"role": "assistant", "content": "20"}],   # family B, at target, scale 10
             [{"role": "assistant", "content": "garbage"}]]  # unparsable, family B, scale 10
    out = train_reward_fn(prompts=[None] * 3, completions=comps, completion_ids=[None] * 3,
                           family=[C.FAMILY_A, C.FAMILY_B, C.FAMILY_B], scale=[1.0, 10.0, 10.0])
    assert abs(out[0]) < 1e-9
    assert abs(out[1]) < 1e-9
    # Uniform scaling is required for GRPO's (r-mean)/std normalization to be an exact
    # invariant of k -- a fixed, unscaled penalty here would reintroduce a k-dependent
    # distortion whenever a group mixes parsable and unparsable completions.
    assert out[2] == C.UNPARSABLE_PENALTY * 10.0


def test_train_reward_fn_is_a_pure_rescaling_for_family_b():
    """The property the whole reward design depends on: for ANY completions, family B's
    reward vector at k=10 must equal EXACTLY 10x its k=1 reward vector (a pure rescaling,
    required for GRPO's group normalization to be k-invariant)."""
    comps = [[{"role": "assistant", "content": "20"}], [{"role": "assistant", "content": "55"}],
             [{"role": "assistant", "content": "garbage"}], [{"role": "assistant", "content": "0"}]]
    family = [C.FAMILY_B] * 4
    out_k1 = train_reward_fn(prompts=[None] * 4, completions=comps, completion_ids=[None] * 4,
                              family=family, scale=[1.0] * 4)
    out_k10 = train_reward_fn(prompts=[None] * 4, completions=comps, completion_ids=[None] * 4,
                               family=family, scale=[10.0] * 4)
    for r1, r10 in zip(out_k1, out_k10):
        assert abs(r10 - 10.0 * r1) < 1e-9


def test_unscaled_reward_batch():
    out = unscaled_reward_batch(["80", "20", "garbage"], C.FAMILY_A)
    assert abs(out[0]) < 1e-9
    assert out[1] < 0
    assert out[2] == C.UNPARSABLE_PENALTY


def test_build_family_datasets_sizes_and_family_labels():
    train_a, train_b = build_family_datasets(seed=0)
    assert len(train_a) == C.N_TRAIN_PER_FAMILY
    assert len(train_b) == C.N_TRAIN_PER_FAMILY
    assert all(r["family"] == C.FAMILY_A for r in train_a)
    assert all(r["family"] == C.FAMILY_B for r in train_b)
    # prompt text is one of the paraphrases, with no family-revealing content
    for r in train_a + train_b:
        text = r["prompt"][0]["content"]
        assert text in C.PARAPHRASES
        assert "family" not in text.lower()
        assert "A" != text and "B" != text  # sanity: not literally leaking a label string


def test_build_train_dataset_scale_column():
    train_a, train_b = build_family_datasets(seed=0)
    rows = build_train_dataset(train_a[:5], train_b[:5], k=10, seed=0)
    assert len(rows) == 10
    for r in rows:
        expected_scale = 1.0 if r["family"] == C.FAMILY_A else 10.0
        assert r["scale"] == expected_scale


def test_determinism():
    d1 = build_family_datasets(seed=0)
    d2 = build_family_datasets(seed=0)
    assert [r["prompt"] for r in d1[0]] == [r["prompt"] for r in d2[0]]


def test_micro_batch_and_accum_no_split_at_default_config():
    # The redesigned task uses 8-token completions, cheap enough to run the whole
    # generation batch in one backward -- MICRO_BATCH_COMPLETIONS is set >= the default
    # PROMPTS_PER_STEP*G, so no splitting should occur at the default operating point.
    micro, accum = C.micro_batch_and_accum(C.PROMPTS_PER_STEP, C.G)
    assert micro == C.PROMPTS_PER_STEP * C.G
    assert accum == 1


def test_micro_batch_and_accum_still_splits_when_target_is_small():
    micro, accum = C.micro_batch_and_accum(16, 8, target_micro_batch_completions=16)
    assert micro * accum == 128
    assert micro % 8 == 0
    assert micro == 16 and accum == 8
