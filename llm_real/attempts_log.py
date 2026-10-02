"""Append-only experiment log. Each baseline_eval.py run (one entry per model it tries) and
each pilot_gate.py run appends ONE entry to DRIVE_RESULTS_DIR/attempts.json -- the config
snapshot, the pre-registered prediction, and the outcome -- so a failed design attempt stays
reported rather than silently overwritten by the next redesign's results. This is the ONE file
the notebook's "delete old results" cell deliberately preserves across wipes."""
import json
import os

# Fixed across attempts -- the theory this whole pilot is testing, stated once so it can't be
# quietly re-worded to fit whatever happened after the fact. Each logged entry freezes a COPY
# of this string at call time, so updating it here only affects FUTURE attempts -- past
# entries in attempts.json keep whatever prediction was live when they were written.
PRE_REGISTERED_PREDICTION = (
    "GRPO's per-group reward normalization is predicted to suppress grader B's (terse "
    "verifier's) pull on the policy relative to Dr.GRPO, which lacks that normalization -- so "
    "Dr.GRPO's completions should end up LONGER (pulled toward grader A's reasoning rubric) "
    "than GRPO's. Gate: PASS iff Dr.GRPO's final mean completion length >= "
    "config.PILOT_GATE_MIN_DRGRPO_TO_GRPO_LENGTH_RATIO x GRPO's."
)


def config_snapshot(C, model_name=None):
    """A snapshot of the config knobs that actually vary across redesign attempts. `model_name`
    overrides C.MODEL_NAME -- baseline_eval.py may be snapshotting a specific model it just
    tried (primary or fallback), not necessarily C.MODEL_NAME's current value."""
    return dict(
        model_name=model_name if model_name is not None else C.MODEL_NAME,
        model_name_fallback=C.MODEL_NAME_FALLBACK,
        max_completion_length=C.MAX_COMPLETION_LENGTH,
        max_steps=C.MAX_STEPS,
        prompts_per_step=C.PROMPTS_PER_STEP,
        g=C.G,
        lr=C.LR,
        terse_smooth_denom=C.TERSE_SMOOTH_DENOM,
        reasoning_max_score=C.REASONING_MAX_SCORE,
        baseline_min_accuracy=C.BASELINE_MIN_ACCURACY,
        pilot_gate_min_drgrpo_to_grpo_length_ratio=C.PILOT_GATE_MIN_DRGRPO_TO_GRPO_LENGTH_RATIO,
        prompt_template=C.PROMPT_TEMPLATE,
    )


def record_attempt(out_dir, stage, config, outcome, prediction=PRE_REGISTERED_PREDICTION):
    """Appends one entry to DRIVE_RESULTS_DIR/attempts.json. Reads the existing log first (if
    any) so this is purely additive -- never truncates or overwrites a prior attempt's entry,
    even one that failed."""
    path = os.path.join(out_dir, "attempts.json")
    if os.path.exists(path):
        with open(path) as f:
            log = json.load(f)
    else:
        log = []
    entry = dict(attempt_number=len(log) + 1, stage=stage, config=config,
                 prediction=prediction, outcome=outcome)
    log.append(entry)
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(log, f, indent=2)
    return entry
