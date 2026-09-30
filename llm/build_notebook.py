"""Generates llm/run_colab.ipynb. Run locally (`python build_notebook.py`) whenever the
notebook's cell content needs to change -- editing the .ipynb JSON by hand is error-prone,
so this script is the source of truth."""
import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

GITHUB_USER = "nilay47"
GITHUB_REPO = "agenthon"


def md(*lines):
    return new_markdown_cell("\n".join(lines))


def code(*lines):
    return new_code_cell("\n".join(lines))


cells = [
    md(
        "# GRPO reward-scaling experiment: Colab A100 runner",
        "",
        "One-click pipeline for the AISTATS LLM experiment: Qwen2.5-0.5B-Instruct + LoRA,",
        "TRL `GRPOTrainer`, comparing GRPO / Dr.GRPO / global normalization reward scaling",
        "under a per-family reward-scale asymmetry (`k`).",
        "",
        "**Run cells top to bottom.** The last cell (full 7-config sweep) is optional and",
        "gated -- only run it after reviewing the pilot's PASS/FAIL verdict.",
        "",
        "All results are saved to Google Drive (`grpo_llm_results/`) after each run, so the",
        "sweep can resume from a Colab disconnect without losing completed work.",
    ),
    md("## 1. Check for a GPU"),
    code(
        "import subprocess",
        "",
        "try:",
        "    import torch",
        "    has_gpu = torch.cuda.is_available()",
        "except ImportError:",
        "    has_gpu = False",
        "",
        "if not has_gpu:",
        "    raise RuntimeError(",
        "        'No GPU detected. Go to Runtime > Change runtime type, select a GPU '",
        "        '(A100 recommended), then Runtime > Restart session and run all cells again.'",
        "    )",
        "",
        "print(subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout)",
    ),
    md("## 2. Config -- edit if your fork/repo name differs"),
    code(
        f'GITHUB_USER = "{GITHUB_USER}"',
        f'GITHUB_REPO = "{GITHUB_REPO}"',
        'DRIVE_RESULTS_DIR = "/content/drive/MyDrive/grpo_llm_results"',
        'REPO_DIR = f"/content/{GITHUB_REPO}"',
    ),
    md("## 3. Mount Google Drive"),
    code(
        "from google.colab import drive",
        "import os",
        "",
        "drive.mount('/content/drive')",
        "os.makedirs(DRIVE_RESULTS_DIR, exist_ok=True)",
        'print(f"Results will be saved to {DRIVE_RESULTS_DIR}")',
    ),
    md(
        "## 4. Clone the repo",
        "",
        "If the repo is **private**, add a `GITHUB_TOKEN` Colab secret first (key icon in the",
        "left sidebar) with a personal access token that has `repo` scope, and grant this",
        "notebook access to it when prompted. If the repo is public, this works with no token.",
    ),
    code(
        "import os",
        "",
        "token = None",
        "try:",
        "    from google.colab import userdata",
        "    token = userdata.get('GITHUB_TOKEN')",
        "except Exception:",
        "    pass",
        "",
        "if token:",
        '    clone_url = f"https://{token}@github.com/{GITHUB_USER}/{GITHUB_REPO}.git"',
        "else:",
        '    clone_url = f"https://github.com/{GITHUB_USER}/{GITHUB_REPO}.git"',
        "",
        "if not os.path.exists(REPO_DIR):",
        "    !git clone {clone_url} {REPO_DIR}",
        "else:",
        '    print(f"{REPO_DIR} already exists, skipping clone (pulling latest instead)")',
        "    !cd {REPO_DIR} && git pull",
        "",
        "%cd {REPO_DIR}/llm",
    ),
    md("## 5. Install dependencies"),
    code("!pip install -q -r requirements.txt"),
    md(
        "## 6. Baseline sanity check",
        "",
        "Runs the **untrained** model on both families' held-out sets. If a family WARNs",
        "(reward out of [0.2, 0.7], or group std too low to give GRPO any learning signal),",
        "read the suggested fix, edit `config.py`'s `TASK_VARIANT_A`/`TASK_VARIANT_B`, and",
        "re-run this cell before continuing.",
    ),
    code(
        '!python baseline_eval.py --out_dir "{DRIVE_RESULTS_DIR}"',
        "",
        "import json, os",
        'with open(os.path.join(DRIVE_RESULTS_DIR, "baseline_eval.json")) as f:',
        "    baseline_result = json.load(f)",
    ),
    md(
        "## 7. Pilot: Dr.GRPO, k=1 and k=10",
        "",
        "Runs the two pilot configs **one at a time**, saving each result to Drive",
        "immediately after it finishes (not just at the end), so a disconnect between the",
        "two runs doesn't lose the first one.",
    ),
    code(
        "import sys",
        "sys.path.insert(0, '.')",
        "from run import run_one",
        "from pilot import PILOT_SEED, check_pass",
        "",
        'print("=== PILOT: Dr.GRPO k=1 ===")',
        'pilot_k1 = run_one("drgrpo", k=1, seed=PILOT_SEED, out_dir=DRIVE_RESULTS_DIR)',
        'print(f"saved to {DRIVE_RESULTS_DIR}/drgrpo_k1_seed{PILOT_SEED}.json")',
    ),
    code(
        'print("=== PILOT: Dr.GRPO k=10 ===")',
        'pilot_k10 = run_one("drgrpo", k=10, seed=PILOT_SEED, out_dir=DRIVE_RESULTS_DIR)',
        'print(f"saved to {DRIVE_RESULTS_DIR}/drgrpo_k10_seed{PILOT_SEED}.json")',
        "",
        "pilot_passed, pilot_detail = check_pass(pilot_k1, pilot_k10)",
        "import json, os",
        'with open(os.path.join(DRIVE_RESULTS_DIR, "pilot_summary.json"), "w") as f:',
        "    json.dump(dict(passed=pilot_passed, detail=pilot_detail,",
        '                   k1_eval_history=pilot_k1["eval_history"],',
        '                   k10_eval_history=pilot_k10["eval_history"],',
        '                   k1_wall_clock_s=pilot_k1["wall_clock_s"],',
        '                   k10_wall_clock_s=pilot_k10["wall_clock_s"]), f, indent=2)',
    ),
    md("## 8. Final summary"),
    code(
        "print('=== Baseline ===')",
        "for fam, v in baseline_result['verdicts'].items():",
        "    print(f\"  family {fam}: {'PASS' if v['passed'] else v['detail']}\")",
        "",
        "print('\\n=== Pilot: Dr.GRPO k=1 vs k=10 ===')",
        "for label, res in [('k=1', pilot_k1), ('k=10', pilot_k10)]:",
        "    print(f\"  {label} (wall_clock={res['wall_clock_s']:.1f}s):\")",
        "    for e in res['eval_history']:",
        "        print(f\"    step={e['step']:>4}  A={e['A']:.4f}  B={e['B']:.4f}\")",
        "",
        "print(f'\\n{pilot_detail}')",
        "print('PILOT', 'PASSED' if pilot_passed else 'FAILED')",
    ),
    md(
        "---",
        "## 9. OPTIONAL -- full 7-config sweep",
        "",
        "**Only run this after reviewing the pilot result above.** This trains all 7",
        "configs (GRPO / Dr.GRPO / global-norm x k in {1,10}, plus the sigma-sampling",
        "method at k=10) x 2 seeds -- substantially more GPU time than the pilot.",
        "",
        "Resumable: any `(method, k, seed)` whose result file already exists in",
        "`DRIVE_RESULTS_DIR` is skipped, so if Colab disconnects mid-sweep, just re-run",
        "this cell (after re-running cells 1-5 to reconnect) and it picks up where it left off.",
    ),
    code(
        "from full import run_sweep, build_summary",
        "import json, os",
        "",
        "SEEDS = [0, 1]  # add 2 for a 3rd seed if time allows",
        "full_results = run_sweep(SEEDS, DRIVE_RESULTS_DIR, skip_existing=True)",
        "full_summary = build_summary(full_results, SEEDS)",
        "",
        "for name, s in full_summary.items():",
        "    print(f\"{name}: progress_A={s['progress_A']['mean']:+.4f} \"",
        "          f\"progress_B={s['progress_B']['mean']:+.4f} \"",
        "          f\"B/A_ratio={s['B_over_A_ratio']['mean']:.2f}\")",
        "",
        'with open(os.path.join(DRIVE_RESULTS_DIR, "full_summary.json"), "w") as f:',
        "    json.dump(full_summary, f, indent=2)",
        'print(f"\\nwrote {DRIVE_RESULTS_DIR}/full_summary.json")',
    ),
]

notebook = new_notebook(cells=cells, metadata={
    "accelerator": "GPU",
    "colab": {"gpuType": "A100", "provenance": []},
    "kernelspec": {"display_name": "Python 3", "name": "python3"},
    "language_info": {"name": "python"},
})

nbformat.validate(notebook)
with open("run_colab.ipynb", "w") as f:
    nbformat.write(notebook, f)
print("wrote run_colab.ipynb")
