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
    _resample_counts,
    bootstrap_stats,
    build_completion_tensors,
    build_gram_matrices,
    compute_advantages,
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
