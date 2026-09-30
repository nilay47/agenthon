"""PILOT: Dr.GRPO (scale_rewards='none'), k=1 and k=10, 1 seed, 150 steps each. Reports
per-family eval curves and wall-clock per run, checks the pass criterion, and STOPS --
the full 7-config x 2-3-seed sweep is gated on an explicit human OK after reading this.

Usage (on a CUDA machine, e.g. Colab A100):
    cd llm && python pilot.py
"""
import json
import os

import config as C
from run import run_one

PILOT_SEED = 0


def progress(history, family):
    if len(history) < 2:
        return None
    return history[-1][family] - history[0][family]


def check_pass(result_k1, result_k10):
    h1, h10 = result_k1["eval_history"], result_k10["eval_history"]
    if len(h1) < 2 or len(h10) < 2:
        return False, "not enough eval checkpoints logged"
    improves_k1 = progress(h1, C.FAMILY_A) > 0 and progress(h1, C.FAMILY_B) > 0
    improves_k10 = progress(h10, C.FAMILY_A) > 0 and progress(h10, C.FAMILY_B) > 0
    b_minus_a_k1 = progress(h1, C.FAMILY_B) - progress(h1, C.FAMILY_A)
    b_minus_a_k10 = progress(h10, C.FAMILY_B) - progress(h10, C.FAMILY_A)
    shifts_toward_b = b_minus_a_k10 > b_minus_a_k1
    passed = improves_k1 and improves_k10 and shifts_toward_b
    detail = (f"k=1 improves both: {improves_k1} (dA={progress(h1,C.FAMILY_A):+.4f}, dB={progress(h1,C.FAMILY_B):+.4f})\n"
              f"k=10 improves both: {improves_k10} (dA={progress(h10,C.FAMILY_A):+.4f}, dB={progress(h10,C.FAMILY_B):+.4f})\n"
              f"B-A progress, k=1: {b_minus_a_k1:+.4f}  k=10: {b_minus_a_k10:+.4f}  "
              f"shifts toward B: {shifts_toward_b}")
    return passed, detail


if __name__ == "__main__":
    print("=== PILOT: Dr.GRPO (scale_rewards='none'), k=1 and k=10, seed=0, 150 steps ===\n")
    result_k1 = run_one("drgrpo", k=1, seed=PILOT_SEED)
    print()
    result_k10 = run_one("drgrpo", k=10, seed=PILOT_SEED)

    passed, detail = check_pass(result_k1, result_k10)
    print("\n=== Pilot eval curves ===")
    for label, res in [("k=1", result_k1), ("k=10", result_k10)]:
        print(f"\nDr.GRPO {label} (wall_clock={res['wall_clock_s']:.1f}s):")
        for e in res["eval_history"]:
            print(f"  step={e['step']:>4}  A={e[C.FAMILY_A]:.4f}  B={e[C.FAMILY_B]:.4f}")

    print("\n=== Pass criterion ===")
    print(detail)
    print(f"\nPILOT {'PASSED' if passed else 'FAILED'}")

    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(C.RESULTS_DIR, "pilot_summary.json"), "w") as f:
        json.dump(dict(passed=passed, detail=detail,
                        k1_wall_clock_s=result_k1["wall_clock_s"], k10_wall_clock_s=result_k10["wall_clock_s"],
                        k1_eval_history=result_k1["eval_history"], k10_eval_history=result_k10["eval_history"]),
                  f, indent=2)
    print(f"\nwrote {C.RESULTS_DIR}/pilot_summary.json")
    if not passed:
        print("\nPilot did not pass -- STOPPING here per instructions. Adjust tasks before spending the full budget.")
    else:
        print("\nPilot passed. STOPPING here per instructions -- waiting for explicit OK before running the full 7-config sweep.")
