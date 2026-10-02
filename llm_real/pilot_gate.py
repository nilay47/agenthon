"""Step 2: PILOT GATE (1 seed). Runs Dr.GRPO and GRPO on the GSM8K conflicting-grader mix and
checks whether Dr.GRPO's lack of per-group reward normalization visibly pulls it toward the
verbose (0-10, "reasoning rubric") grader relative to GRPO -- PASS if Dr.GRPO's final mean
completion length exceeds GRPO's by at least PILOT_GATE_MIN_LENGTH_SHIFT_FRAC (15%).

This is a GATE: the notebook stops here regardless of outcome. A FAIL here means either the
predicted effect doesn't show up on this real task (not just the synthetic one llm/ already
validated), or the pilot's signal is too weak/noisy (e.g. accuracy ~0, so neither grader has
much to differentiate on) -- diagnostics are printed either way, comparing against the Step-1
baseline's numbers if results/baseline_eval.json exists.

Usage: cd llm_real && python pilot_gate.py
"""
import json
import os

import config as C
from run import run_one


def _result_path(out_dir, method, seed):
    return os.path.join(out_dir, f"{method}_seed{seed}.json")


def _final_eval(result):
    return result["eval_history"][-1] if result["eval_history"] else None


def run_pilot_gate(seed=0, out_dir=C.RESULTS_DIR, skip_existing=True):
    results = {}
    wall_clocks = {}
    for method in ["drgrpo", "grpo"]:
        path = _result_path(out_dir, method, seed)
        if skip_existing and os.path.exists(path):
            print(f"=== {method}_seed{seed}: SKIPPED (result already exists at {path}) ===")
            with open(path) as f:
                res = json.load(f)
        else:
            print(f"\n=== {method}_seed{seed} ===")
            res = run_one(method, seed=seed, out_dir=out_dir)
        results[method] = res
        wall_clocks[method] = res["wall_clock_s"]

    finals = {m: _final_eval(results[m]) for m in results}
    for m, e in finals.items():
        if e is None:
            raise RuntimeError(f"{m}_seed{seed} has no logged eval_history entries -- "
                                f"MAX_STEPS is probably smaller than EVAL_EVERY")

    len_drgrpo, len_grpo = finals["drgrpo"]["mean_n_tokens"], finals["grpo"]["mean_n_tokens"]
    frac_shift = (len_drgrpo - len_grpo) / len_grpo if len_grpo > 0 else float("nan")
    passed = bool(frac_shift >= C.PILOT_GATE_MIN_LENGTH_SHIFT_FRAC)

    print(f"\n=== PILOT GATE (seed={seed}) ===")
    print(f"Dr.GRPO: final mean completion length = {len_drgrpo:.2f} tokens  "
          f"(accuracy={finals['drgrpo']['accuracy']:.3f}, wall_clock={wall_clocks['drgrpo']:.1f}s)")
    print(f"GRPO:    final mean completion length = {len_grpo:.2f} tokens  "
          f"(accuracy={finals['grpo']['accuracy']:.3f}, wall_clock={wall_clocks['grpo']:.1f}s)")
    print(f"relative shift (drgrpo-grpo)/grpo = {frac_shift:.3f}  (require >= {C.PILOT_GATE_MIN_LENGTH_SHIFT_FRAC})")
    print(f"\n{'PASS' if passed else 'FAIL'}")

    diagnostics = None
    if not passed:
        print("\nDiagnostics:")
        baseline_path = os.path.join(out_dir, "baseline_eval.json")
        if os.path.exists(baseline_path):
            with open(baseline_path) as f:
                baseline = json.load(f)
            base_len = baseline["n_tokens_distribution"]["mean"]
            base_acc = baseline["accuracy"]
            diagnostics = dict(baseline_mean_n_tokens=base_len, baseline_accuracy=base_acc,
                                drgrpo_length_moved_from_baseline=len_drgrpo - base_len,
                                grpo_length_moved_from_baseline=len_grpo - base_len,
                                drgrpo_accuracy_moved_from_baseline=finals["drgrpo"]["accuracy"] - base_acc,
                                grpo_accuracy_moved_from_baseline=finals["grpo"]["accuracy"] - base_acc)
            print(f"  baseline (untrained): mean_n_tokens={base_len:.2f}  accuracy={base_acc:.3f}")
            print(f"  Dr.GRPO length moved by {diagnostics['drgrpo_length_moved_from_baseline']:+.2f} tokens, "
                  f"accuracy moved by {diagnostics['drgrpo_accuracy_moved_from_baseline']:+.3f}")
            print(f"  GRPO length moved by {diagnostics['grpo_length_moved_from_baseline']:+.2f} tokens, "
                  f"accuracy moved by {diagnostics['grpo_accuracy_moved_from_baseline']:+.3f}")
        else:
            print(f"  (no {baseline_path} found -- run baseline_eval.py first for a length/accuracy-moved-at-all check)")

    summary = dict(seed=seed, passed=passed, frac_shift=frac_shift,
                    drgrpo=dict(final=finals["drgrpo"], wall_clock_s=wall_clocks["drgrpo"]),
                    grpo=dict(final=finals["grpo"], wall_clock_s=wall_clocks["grpo"]),
                    diagnostics=diagnostics)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "pilot_gate_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {out_path}")
    return summary


if __name__ == "__main__":
    run_pilot_gate()
