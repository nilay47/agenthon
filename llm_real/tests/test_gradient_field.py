"""CPU-only, GPU-free unit tests for gradient_field.py's pure logic: prompt selection, the
TRL-completion-mask replication, advantage formulas, and the Gram-matrix bootstrap (checked
against a brute-force reference). The TRL loss-equivalence check itself (validate_loss_equivalence)
needs a real model forward/backward pass and is validated separately via an end-to-end dry run
against the tiny random-weight checkpoint, not repeated here."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gradient_field import (  # noqa: E402
    _agg_dot,
    _cosine,
    _resample_counts,
    apply_loss_eq_settings,
    bootstrap_stats,
    build_completion_tensors,
    build_gram_matrices,
    compute_advantages,
    grpo_overrides_for_settings,
    select_measurement_prompts,
)


def test_select_measurement_prompts_seeded_and_distinct():
    fake_train = [dict(question=f"q{i}", answer=f"reasoning\n#### {i}") for i in range(200)]
    p1 = select_measurement_prompts(fake_train, n=50, seed=999)
    p2 = select_measurement_prompts(fake_train, n=50, seed=999)
    assert [p["gold"] for p in p1] == [p["gold"] for p in p2]  # deterministic
    assert len(set(p["gold"] for p in p1)) == 50  # all distinct


def test_build_completion_tensors_trims_inclusive_of_eos():
    eos, pad = 99, 0
    rows = torch.tensor([
        [1, 2, 99, 7, 7],  # eos at index 2 -> keep [1,2,99]
        [3, 4, 5, 6, 7],   # no eos -> keep the whole row
    ])
    ids, mask = build_completion_tensors(rows, eos_token_id=eos, pad_token_id=pad)
    assert ids.shape == (2, 5)  # padded to the longest row (5)
    assert mask[0].tolist() == [1, 1, 1, 0, 0]  # inclusive of the eos token itself
    assert mask[1].tolist() == [1, 1, 1, 1, 1]
    assert ids[0, :3].tolist() == [1, 2, 99]
    assert ids[0, 3:].tolist() == [pad, pad]


def test_compute_advantages_dr_grpo_is_just_centering():
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
    adv = compute_advantages(rewards, "dr_grpo")
    assert torch.allclose(adv, rewards - rewards.mean())


def test_compute_advantages_grpo_uses_ddof1_bessel_correction():
    # Matches TRL's own nanstd exactly (Bessel-corrected, ddof=1) -- confirmed against
    # trl/trainer/utils.py source. torch.std's default unbiased=True is this directly.
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
    adv = compute_advantages(rewards, "grpo")
    expected_std = rewards.std(unbiased=True)
    expected = (rewards - rewards.mean()) / (expected_std + 1e-4)
    assert torch.allclose(adv, expected)


def test_compute_advantages_grpo_single_sample_no_crash():
    rewards = torch.tensor([5.0])
    adv = compute_advantages(rewards, "grpo")
    assert torch.allclose(adv, torch.tensor([0.0]))  # centered to 0, std=0 -> 0/(0+eps)=0


def test_gram_matrix_bootstrap_matches_brute_force():
    rng = np.random.default_rng(42)
    n_prompts, d = 10, 50
    gA = rng.normal(size=(n_prompts, d)).astype(np.float32)
    gB = rng.normal(size=(n_prompts, d)).astype(np.float32)
    hA = rng.normal(size=(n_prompts, d)).astype(np.float32)
    hB = rng.normal(size=(n_prompts, d)).astype(np.float32)

    grams = build_gram_matrices(dict(gA=gA, gB=gB, hA=hA, hB=hB))
    stats = bootstrap_stats(grams, n_prompts, n_bootstrap=50, seed=1)

    def cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    U_A_dr, U_B_dr = gA.mean(axis=0), gB.mean(axis=0)
    U_dr = (U_A_dr + U_B_dr) / 2
    U_A_grpo, U_B_grpo = hA.mean(axis=0), hB.mean(axis=0)
    U_grpo = (U_A_grpo + U_B_grpo) / 2

    angle_brute = float(np.degrees(np.arccos(np.clip(cos(U_dr, U_grpo), -1, 1))))
    cos_ab_brute = cos(U_A_dr, U_B_dr)
    ratio_dr_brute = np.linalg.norm(U_B_dr) / np.linalg.norm(U_A_dr)
    ratio_grpo_brute = np.linalg.norm(U_B_grpo) / np.linalg.norm(U_A_grpo)

    assert abs(stats["angle_deg"]["point"] - angle_brute) < 1e-3
    assert abs(stats["cos_AB_dr"]["point"] - cos_ab_brute) < 1e-5
    assert abs(stats["ratio_dr"]["point"] - ratio_dr_brute) < 1e-5
    assert abs(stats["ratio_grpo"]["point"] - ratio_grpo_brute) < 1e-5
    assert abs(stats["R"]["point"] - ratio_grpo_brute / ratio_dr_brute) < 1e-5


def test_gram_matrix_bootstrap_resample_matches_brute_force():
    rng = np.random.default_rng(7)
    n_prompts, d = 8, 20
    gA = rng.normal(size=(n_prompts, d)).astype(np.float32)
    gB = rng.normal(size=(n_prompts, d)).astype(np.float32)
    grams = build_gram_matrices(dict(gA=gA, gB=gB))

    counts = _resample_counts(n_prompts, np.random.default_rng(99))
    idx = np.repeat(np.arange(n_prompts), counts.astype(int))
    assert len(idx) == n_prompts

    norm_sq_gram = _agg_dot(counts, grams[("gA", "gA")], n_prompts)
    U_A_resampled = gA[idx].mean(axis=0)
    assert abs(norm_sq_gram - float(np.dot(U_A_resampled, U_A_resampled))) < 1e-4


def test_bootstrap_stats_default_args_resolve_live_not_at_def_time():
    """Regression test: finite_g_curve/bootstrap_stats used to bind n_bootstrap/g_values etc.
    as default ARGUMENT values (evaluated once at function-definition time), so overriding the
    module constant afterward had no effect -- the same class of bug as a `from config import X`
    stale-binding import. Defaults must resolve from the module globals inside the function
    body so a monkeypatch before calling actually takes effect."""
    import gradient_field as gf

    rng = np.random.default_rng(0)
    n_prompts, d = 4, 10
    gA = rng.normal(size=(n_prompts, d)).astype(np.float32)
    grams = build_gram_matrices(dict(gA=gA, gB=gA, hA=gA, hB=gA))

    original = gf.N_BOOTSTRAP
    try:
        gf.N_BOOTSTRAP = 17
        stats = bootstrap_stats(grams, n_prompts)  # no explicit n_bootstrap -- must pick up the monkeypatch
        assert stats["n_bootstrap"] == 17
    finally:
        gf.N_BOOTSTRAP = original


def test_cosine_helper():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    assert abs(_cosine(a, b) - 1.0) < 1e-9
    c = np.array([0.0, 1.0, 0.0])
    assert abs(_cosine(a, c)) < 1e-9
    d = np.array([-1.0, 0.0, 0.0])
    assert abs(_cosine(a, d) + 1.0) < 1e-9


def test_grpo_overrides_for_settings():
    assert grpo_overrides_for_settings(frozenset()) == {}
    assert grpo_overrides_for_settings(frozenset({"lora_dropout_zero"})) == {}
    assert grpo_overrides_for_settings(frozenset({"no_grad_checkpointing"})) == {}
    assert grpo_overrides_for_settings(frozenset({"deterministic_algorithms"})) == {}
    assert grpo_overrides_for_settings(frozenset({"fp32"})) == {"bf16": False, "fp16": False}
    combined = grpo_overrides_for_settings(frozenset({"fp32", "no_grad_checkpointing"}))
    assert combined == {"bf16": False, "fp16": False}


def test_apply_loss_eq_settings_mutates_config_and_returns_originals():
    import config as C
    import gradient_field as gf

    original_dtype, original_ckpt = C.MODEL_DTYPE, C.GRADIENT_CHECKPOINTING
    try:
        saved = apply_loss_eq_settings(frozenset())
        assert saved == dict(MODEL_DTYPE=original_dtype, GRADIENT_CHECKPOINTING=original_ckpt)
        assert C.MODEL_DTYPE == original_dtype  # baseline: no mutation
        assert C.GRADIENT_CHECKPOINTING == original_ckpt

        apply_loss_eq_settings(frozenset({"fp32"}))
        assert C.MODEL_DTYPE is torch.float32

        C.MODEL_DTYPE = original_dtype  # reset before the next trial, as diagnose_and_validate does
        apply_loss_eq_settings(frozenset({"no_grad_checkpointing"}))
        assert C.GRADIENT_CHECKPOINTING is False
    finally:
        C.MODEL_DTYPE, C.GRADIENT_CHECKPOINTING = original_dtype, original_ckpt


def test_build_trainer_gradient_checkpointing_matches_config_not_hardcoded():
    """Regression test for the real bug this diagnosis surfaced: build_trainer() used to
    hardcode gradient_checkpointing=False in its GRPOConfig regardless of
    config.GRADIENT_CHECKPOINTING, so the replay trainer silently used a DIFFERENT setting than
    the real trainer (run_one, which reads config.GRADIENT_CHECKPOINTING) -- a genuine source of
    real-vs-replay divergence on GPU (gradient checkpointing recomputes activations during
    backward; combined with non-deterministic kernels, a mismatched recompute path can diverge
    from a straight single forward pass), even though it never showed up on CPU."""
    import inspect

    import gradient_field as gf

    src = inspect.getsource(gf.build_trainer)
    assert "gradient_checkpointing=False" not in src
    assert "gradient_checkpointing=C.GRADIENT_CHECKPOINTING" in src


def test_diagnose_handles_crash_and_isolates_deterministic_flag(monkeypatch):
    """Regression test for two bugs this diagnosis surfaced: (1) a candidate that raises (e.g.
    deterministic_algorithms hitting an op without a deterministic kernel) must not crash the
    whole diagnostic -- later candidates still get tested; (2) torch.use_deterministic_algorithms
    must not leak from a crashed/non-winning trial into a LATER, unrelated winning candidate that
    never asked for it."""
    import gradient_field as gf

    calls = []

    def fake_validate(seed, rel_tol=1e-5, settings=frozenset()):
        calls.append(settings)
        if "deterministic_algorithms" in settings:
            torch.use_deterministic_algorithms(True)  # simulate the real function turning it on
            raise RuntimeError("simulated: op has no deterministic kernel on this backend")
        if settings == frozenset({"fp32"}):
            return dict(passed=True, relative_diff=0.0, cosine=1.0, settings=sorted(settings))
        return dict(passed=False, relative_diff=1.0, cosine=0.0, settings=sorted(settings))

    original_was_deterministic = torch.are_deterministic_algorithms_enabled()
    monkeypatch.setattr(gf, "validate_loss_equivalence", fake_validate)
    try:
        winning, table = gf.diagnose_and_validate_loss_equivalence(seed=0)
        assert winning == frozenset({"fp32"})  # found despite a crashing candidate elsewhere in the list
        assert len(calls) == len(gf.LOSS_EQ_CANDIDATE_SETTINGS)  # every candidate was tried, none skipped
        crashed_rows = [r for r in table if "error" in r]
        assert len(crashed_rows) >= 1
        # the winning candidate ("fp32") doesn't include determinism -- it must not be left on
        # just because a LATER-in-the-list crashed candidate turned it on and never cleaned up.
        assert torch.are_deterministic_algorithms_enabled() == original_was_deterministic
    finally:
        torch.use_deterministic_algorithms(original_was_deterministic)


def test_pad_and_cat_pads_shorter_chunks_to_global_max():
    from gradient_field import _pad_and_cat

    a = torch.tensor([[1, 2, 3], [4, 5, 6]])       # (2, 3)
    b = torch.tensor([[7, 8]])                      # (1, 2) -- shorter, needs padding
    out = _pad_and_cat([a, b], pad_value=0)
    assert out.shape == (3, 3)
    assert out[0].tolist() == [1, 2, 3]
    assert out[1].tolist() == [4, 5, 6]
    assert out[2].tolist() == [7, 8, 0]  # padded with 0 on the right


def test_per_group_gradient_microbatched_is_invariant_to_chunk_size(monkeypatch):
    """dr_grpo's loss normalizer divides by the CALLING batch's own row count, so a realistic
    per_group_gradient returns roughly the SAME value regardless of how large a chunk it's
    called on (each row contributes about equally, already normalized away) -- verified here
    with a constant fake per_group_gradient. per_group_gradient_microbatched's (chunk_size/g)
    rescaling must then make the TOTAL identical whether the group is processed as one chunk
    or split into several -- this is the exact property that makes microbatching for memory
    safety mathematically exact rather than an approximation."""
    import gradient_field as gf

    const_grad = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    def fake_per_group_gradient(trainer, model, prompt_ids, prompt_mask, completion_ids, completion_mask,
                                 advantages, device):
        return const_grad.copy()

    monkeypatch.setattr(gf, "per_group_gradient", fake_per_group_gradient)

    g = 8
    prompt_ids = torch.zeros(g, 4, dtype=torch.long)
    prompt_mask = torch.ones(g, 4, dtype=torch.long)
    completion_ids = torch.zeros(g, 4, dtype=torch.long)
    completion_mask = torch.ones(g, 4, dtype=torch.long)
    advantages = torch.zeros(g)

    whole = gf.per_group_gradient_microbatched(None, None, prompt_ids, prompt_mask, completion_ids,
                                                completion_mask, advantages, torch.device("cpu"), micro_batch_size=8)
    split_in_4s = gf.per_group_gradient_microbatched(None, None, prompt_ids, prompt_mask, completion_ids,
                                                      completion_mask, advantages, torch.device("cpu"), micro_batch_size=4)
    split_in_2s = gf.per_group_gradient_microbatched(None, None, prompt_ids, prompt_mask, completion_ids,
                                                      completion_mask, advantages, torch.device("cpu"), micro_batch_size=2)
    np.testing.assert_allclose(whole, const_grad)
    np.testing.assert_allclose(split_in_4s, const_grad)
    np.testing.assert_allclose(split_in_2s, const_grad)


def test_item1_results_save_and_load_roundtrip(tmp_path):
    from gradient_field import load_item1_results, save_item1_results

    out_dir = str(tmp_path)
    grams = {("gA", "gA"): np.array([[1.0, 2.0], [2.0, 4.0]]),
              ("gA", "gB"): np.array([[0.5, 0.5], [0.5, 0.5]])}
    stats = dict(angle_deg=dict(point=15.0, mean=15.2, ci=[10.0, 20.0]))
    per_prompt = [dict(reward_a_mean=1.0, reward_b_mean=2.0, reward_a_std=0.1, reward_b_std=0.2,
                        mean_n_tokens=50.0, mean_n_steps=3.0)]

    save_item1_results(out_dir, grams, stats, median_std_ratio_a_over_b=0.5,
                        per_prompt_diagnostics=per_prompt, n_prompts=2, g_main=8)

    loaded = load_item1_results(out_dir, expected_n_prompts=2, expected_g_main=8)
    assert loaded is not None
    loaded_grams, loaded_stats, loaded_ratio, loaded_per_prompt = loaded
    np.testing.assert_allclose(loaded_grams[("gA", "gA")], grams[("gA", "gA")])
    np.testing.assert_allclose(loaded_grams[("gA", "gB")], grams[("gA", "gB")])
    assert loaded_stats == stats
    assert loaded_ratio == 0.5
    assert loaded_per_prompt == per_prompt


def test_item1_results_mismatch_triggers_recompute_not_silent_reuse(tmp_path):
    from gradient_field import load_item1_results, save_item1_results

    out_dir = str(tmp_path)
    save_item1_results(out_dir, {("gA", "gA"): np.array([[1.0]])}, dict(), 0.5, [], n_prompts=128, g_main=8)

    # same file exists, but a DIFFERENT n_prompts is now configured -> must not reuse stale data
    assert load_item1_results(out_dir, expected_n_prompts=64, expected_g_main=8) is None
    assert load_item1_results(out_dir, expected_n_prompts=128, expected_g_main=16) is None
    assert load_item1_results(out_dir, expected_n_prompts=128, expected_g_main=8) is not None


def test_item1_results_missing_file_returns_none(tmp_path):
    from gradient_field import load_item1_results

    assert load_item1_results(str(tmp_path), expected_n_prompts=128, expected_g_main=8) is None
