"""
Compiles paper_inputs.md from results/*.json. No new experiments: every number below is
loaded from an existing results/*.json file (source noted inline), or from the two small
deterministic-capture scripts run just before this one (capture_test_diagnostics.py,
capture_fixed_instances.py), which re-record already-computed/deterministic quantities
rather than run anything new.
"""
import json

import numpy as np

R = "results/"


def load(name):
    with open(R + name) as f:
        return json.load(f)


pilot1_summary = load("pilot1_summary.json")
pilot2 = load("pilot2_grpo.json")
fu_a = load("followup_a_b3_sanity.json")
fu_b = load("followup_b_proxy_optima.json")
fu_c = load("followup_c_grpo_misspecified.json")
fu_final = load("followup_final_regret.json")
fu_bin = load("followup_binary_reward.json")
fu_bin_dose = load("followup_binary_dose_response.json")
fu_mlp = load("followup_mlp_regret.json")
fu_mlp_stab = load("followup_mlp_stability.json")
test_diag = load("test_diagnostics.json")
fixed_inst = load("fixed_instances.json")

lines = []


def h(s, level=2):
    if lines and lines[-1] != "":
        lines.append("")
    lines.append("#" * level + " " + s)
    lines.append("")


def p(s=""):
    lines.append(s)


def table(headers, rows):
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    lines.append("")


def f(x, nd=4):
    return f"{x:.{nd}f}"


# ---------------------------------------------------------------------------
h("paper_inputs.md", 1)
p("Compiled inputs for writing the paper: every number in this file is copied from a file "
  "under `results/*.json` (source cited per table/value; field names match the JSON keys "
  "verbatim) or from the named source-code constants in `env.py` / `policy.py` / "
  "`pilot_goodhart.py` / `pilot_grpo.py`. No experiments were run to produce this file -- "
  "`capture_test_diagnostics.py` and `capture_fixed_instances.py` only re-record "
  "already-deterministic quantities (the tests' internal numbers, and the 16 fixed "
  "instances' sampled parameters) that weren't previously saved to `results/`.")
p()

# ===========================================================================
h("a. Setup details")

h("AC / environment parameters", 3)
p("Source: `env.py` module constants.")
table(["parameter", "value"], [
    ["N (steps)", "20"], ["T (horizon)", "1.0"], ["tau = T/N", "0.05"],
    ["gamma (permanent impact)", "0"], ["eps (spread)", "0"],
    ["policy std s (fixed, not learned)", "0.05  (`policy.DEFAULT_S`)"],
])

h("Instance sampler ranges", 3)
p("Source: `env.py`, `sample_instances()`. X, sigma ~ Uniform; lam, eta ~ log-Uniform.")
table(["param", "range"], [
    ["X", "[0.5, 2.0]"], ["sigma", "[0.1, 0.6]"], ["lam", "[0.1, 10.0] (log-uniform)"],
    ["eta", "[0.01, 0.1] (log-uniform)"],
])

h("The 16 fixed instances", 3)
p("Source: `results/fixed_instances.json` (`FIXED_SEED=777`, `pilot_grpo.get_fixed_batch()`; "
  "kappa via `env.kappa_batch`). sigma_r (continuous reward, kappa feature set, at "
  "`theta_true*`): `results/pilot2_grpo.json`, `regret_table[*].sigma_r`. p (binary reward, "
  "phi1, at `theta_true*`): `results/followup_binary_reward.json`, "
  "`per_instance[*].success_rate_theta_true_star`.")
sigma_r_by_inst = {row["instance"]: row["sigma_r"] for row in pilot2["regret_table"]}
p_by_inst = {row["instance"]: row["success_rate_theta_true_star"] for row in fu_bin["per_instance"]}
rows = []
for r in fixed_inst:
    i = r["instance"]
    rows.append([i, f(r["X"], 4), f(r["sigma"], 4), f(r["lam"], 4), f(r["eta"], 4), f(r["kappa"], 4),
                 f(sigma_r_by_inst[i], 5), f(p_by_inst[i], 3)])
table(["instance", "X", "sigma", "lam", "eta", "kappa", "sigma_r (continuous, kappa feat.)", "p (binary, phi1)"], rows)

h("Policy parametrization", 3)
p("`n_k = x_{k-1} * a_k`, `a_k = m_k + s*z_k` (k<N), `a_N=1` forced (true sim). "
  "`m_k = theta . phi(t_k, instance)` (LinearPolicy) or `MLP(phi(t_k, instance))` (MLPPolicy). "
  "Source: `policy.py`, `env.py FEATURE_SETS`.")
table(["feature set", "n_feat", "definition"], [
    ["kappa (spec default)", str(fu_c["kappa"]["n_feat"]), "[1, t/T, (t/T)^2, kappaT, kappaT*t/T]"],
    ["phi1 = time_only", str(fu_c["time_only"]["n_feat"]), "[1, t/T, (t/T)^2]"],
    ["phi2 = raw_params", str(fu_c["raw_params"]["n_feat"]), "[1, t/T, (t/T)^2, log sigma, log lam, log eta]"],
    ["time_scalar (MLP input)", "1", "[t/T]  (MLP supplies its own bias)"],
])
p("MLP policy (Extensions 2): 2 hidden layers x 64, tanh, input=time_scalar. "
  f"Source: `results/followup_mlp_regret.json`, `hidden_sizes`={fu_mlp['hidden_sizes']}.")
p()

# ===========================================================================
h("b. Training details per experiment")

table(["experiment", "estimators", "G", "steps", "optimizer", "LR", "seeds", "init", "grad clip"], [
    ["Pilot 1 (Goodhart)", "RLOO", "16", "400 (eval every 10)", "Adam", "0.05", "[0,1,2]",
     "TWAP + random (linear); random (MLP)", "5.0 (`clip_grad_norm_`)"],
    ["Pilot 2 (GRPO, kappa)", "RLOO, Dr.GRPO, GRPO", "{4, 16}", "1500", "Adam", "0.05", "[0,1,2]", "TWAP", "5.0"],
    ["Follow-up C (phi1/phi2/kappa)", "RLOO, Dr.GRPO, GRPO", "{4, 16}", "1500", "Adam", "0.05", "[0,1,2]", "TWAP", "5.0"],
    ["Follow-up D / final regret (kappa, phi1)", "RLOO, Dr.GRPO, GRPO", "16 only", "1500", "Adam", "0.05",
     "[0,1,2,3,4]", "TWAP", "5.0"],
    ["Extension 1 binary (base)", "RLOO, Dr.GRPO, GRPO", "16", "1500", "Adam", "0.05", "[0,1,2,3,4]", "TWAP", "5.0"],
    ["Extension 1 dose-response", "RLOO, GRPO", "16", "1500", "Adam", "0.05", "[0,1,2,3,4]", "TWAP", "5.0"],
    ["Extension 2 MLP (initial)", "RLOO, Dr.GRPO, GRPO", "16", "1500", "Adam", "0.05", "[0,1,2,3,4]", "small random (std 0.1)", "5.0"],
    ["Extension 2 MLP LR sweep", "RLOO only", "16", "1500", "Adam", "{0.01, 0.02, 0.05}", "[0,1,2]",
     "small random (std 0.1)", "5.0"],
    ["Extension 2 MLP (20 seeds)", "RLOO, Dr.GRPO, GRPO", "16", "1500", "Adam", "0.01 (chosen)", "0..19",
     "small random (std 0.1)", "5.0"],
])
p("Sources: `pilot_goodhart.py` (N_TRAIN_PER_STEP=64 instances/step, G_TRAIN=16, LR=0.05, "
  "N_STEPS=400, SEEDS=[0,1,2]); `pilot_grpo.py` (N_FIXED_INSTANCES=16, FIXED_SEED=777, "
  "N_TRAIN_STEPS=1500, LR=0.05, SEEDS=[0,1,2], G_VALUES=[4,16], SIGMA_R_MC=10000, "
  "N_OUTER_FIXED_POINT=8, N_INNER_PER_OUTER=400); `followup_final.py`, `followup_binary.py`, "
  "`followup_binary_dose.py`, `followup_mlp.py`, `followup_mlp_stability.py` "
  "(SEEDS5=[0,1,2,3,4], FULL_SEEDS=range(20)). Grad clipping "
  "`torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=5.0)` is applied "
  "identically in every training loop (`pilot_goodhart.train_one`, `pilot_grpo.train_estimator`, "
  "`followup_binary.train_estimator_binary`, `followup_mlp.train_estimator_mlp`).")

h("Goodhart bug definitions and magnitudes (Pilot 1)", 3)
p("Source: `pilot_goodhart.py`, `BUGS` dict.")
table(["bug", "magnitudes c"], [
    ["B1 (weak terminal penalty)", "[0.001, 0.01, 0.1]"],
    ["B2 (one-step cost lag, blend in [0,1])", "[0.3, 0.7, 1.0]"],
    ["B3 (per-step impact cap)", "[0.002, 0.005, 0.02]"],
])

h("Binary-reward delta values", 3)
p("delta such that `r_bin = 1[r_true >= r*_inst - delta*|r*_inst|]`, r*_inst = -ac_cost_inst. "
  "Source: `results/followup_binary_reward.json` (base run, pooled target); "
  "`results/followup_binary_dose_response.json` (dose sweep, min-p target).")
table(["run", "target", "delta", "pooled/min p achieved"], [
    ["base", "pooled p in [0.2,0.5]", f(fu_bin["delta"], 4), f(fu_bin["pooled_success_rate_twap_init"], 3) + " (pooled)"],
] + [
    [f"dose {i+1}", f"min p ~ {row['target_min_p']}", f(row["delta"], 4), f(row["min_p"], 3) + " (min)"]
    for i, row in enumerate(fu_bin_dose)
])

h("MLP LR sweep (Extension 2)", 3)
p("Source: `results/followup_mlp_stability.json`, `lr_sweep` (RLOO only, 3 seeds each).")
rows = []
for lr, d in fu_mlp_stab["lr_sweep"].items():
    rows.append([lr, ", ".join(f(x, 4) for x in d["regrets"]), f(d["median"], 4), f(d["iqr"], 4)])
table(["lr", "seed regrets", "median", "IQR"], rows)
p(f"**Chosen LR: {fu_mlp_stab['chosen_lr']}** (lowest median, source: `chosen_lr` field).")
p()

# ===========================================================================
h("c. Tests")

p("Source: `tests/test_env.py` (assertions) and `results/test_diagnostics.json` "
  "(measured values, captured by `capture_test_diagnostics.py` re-running the exact same "
  "deterministic code/seeds as the tests, without changing them).")

td = test_diag
table(["test", "what it checks", "tolerance", "measured"], [
    ["test_exact_J_matches_monte_carlo", "exact_J agrees with a 1e5-rollout MC estimate "
     "(5 instances x 3 random thetas)", td["test_exact_J_matches_monte_carlo"]["tolerance"],
     f"max |diff|/SE = {f(td['test_exact_J_matches_monte_carlo']['max_diff_over_se_ratio'], 3)} (limit 3.0)"],
    ["test_ac_closed_form_is_optimal", "AC closed-form schedule cost vs scipy-optimized "
     "free schedule (5 instances, 6 restarts)", td["test_ac_closed_form_is_optimal"]["tolerance"],
     f"max rel_gap = {td['test_ac_closed_form_is_optimal']['max_rel_gap']:.2e}"],
    ["test_autograd_matches_finite_difference", "autograd gradient of exact_J vs central "
     "finite differences (eps=1e-6, 4 instances)", td["test_autograd_matches_finite_difference"]["tolerance"],
     f"max abs diff = {td['test_autograd_matches_finite_difference']['max_abs_diff']:.2e}, "
     f"max rel err = {td['test_autograd_matches_finite_difference']['max_rel_err']:.2e}"],
    ["test_drgrpo_is_exact_factor_of_rloo", "Dr.GRPO advantage == (G-1)/G * RLOO advantage "
     "exactly (20 groups, G=8)", td["test_drgrpo_is_exact_factor_of_rloo"]["tolerance"],
     f"max abs diff = {td['test_drgrpo_is_exact_factor_of_rloo']['max_abs_diff']:.2e}"],
    ["test_rloo_mean_gradient_matches_exact_gradient", "RLOO's mean estimated policy "
     "gradient (G=4000) vs exact grad of sum_i J_i (6 instances)",
     td["test_rloo_mean_gradient_matches_exact_gradient"]["tolerance"],
     f"rel_err = {f(td['test_rloo_mean_gradient_matches_exact_gradient']['rel_err'], 4)}"],
    ["test_grpo_mean_gradient_matches_inverse_std_weighted_target", "GRPO's mean estimated "
     "gradient (G=4000) vs sum_i grad J_i / sigma_r_i (5 instances)",
     td["test_grpo_mean_gradient_matches_inverse_std_weighted_target"]["tolerance"],
     f"cosine = {f(td['test_grpo_mean_gradient_matches_inverse_std_weighted_target']['cosine'], 4)}"],
    ["test_exact_var_matches_monte_carlo", "closed-form exact_var (Z_k cost-to-go recursion) "
     "vs 1e5-rollout MC sample variance (5 instances x 3 random thetas)",
     td["test_exact_var_matches_monte_carlo"]["tolerance"],
     f"max rel err = {f(td['test_exact_var_matches_monte_carlo']['max_rel_err'], 4)}"],
])
p("All 7 tests pass (`pytest tests/ -q` -> `7 passed`).")
p()

# ===========================================================================
h("d. Results")

h("Main regret table: kappa vs phi1 x {RLOO, Dr.GRPO, GRPO, theta_true*, theta_GRPO*}", 3)
p("G=16, 5 seeds. Mean regret over the 16 instances, 95% CI over seeds (t-interval, df=4). "
  "Source: `results/followup_final_regret.json`, `<feature_set>.mean_regret_summary`.")
rows = []
label = {"rloo": "RLOO", "drgrpo": "Dr.GRPO", "grpo": "GRPO"}
for fs, fs_label in [("kappa", "kappa"), ("time_only", "phi1")]:
    ms = fu_final[fs]["mean_regret_summary"]
    for est in ["rloo", "drgrpo", "grpo"]:
        m = ms[est]
        rows.append([fs_label, label[est], f(m["mean"], 4), f"[{f(m['ci_lo'],4)}, {f(m['ci_hi'],4)}]"])
    rows.append([fs_label, "theta_true* (ref)", f(ms["theta_true_star"]["mean"], 4), "--"])
    rows.append([fs_label, "theta_GRPO* (ref)", f(ms["theta_grpo_star"]["mean"], 4), "--"])
table(["feature set", "estimator", "mean regret", "95% CI"], rows)

h("Theta distances (endpoint to theta_true* and theta_GRPO*) per feature set", 3)
p("G in {4,16}, 3 seeds. Source: `results/followup_c_grpo_misspecified.json`, "
  "`<feature_set>.distance_table` (mean +/- sd over 3 seeds); kappa gap and G=16 "
  f"reference also cross-checked against `results/pilot2_grpo.json` (gap={f(pilot2['dist_true_grpo'],5)}).")
for fs, fs_label in [("kappa", "kappa"), ("time_only", "phi1"), ("raw_params", "phi2")]:
    p(f"**{fs_label}** (n_feat={fu_c[fs]['n_feat']}, gap(theta_true*, theta_GRPO*)={f(fu_c[fs]['gap'],5)}):")
    rows = []
    for row in fu_c[fs]["distance_table"]:
        rows.append([label[row["estimator"]], row["G"],
                     f"{f(row['mean_d_true'],4)} (sd {f(row['sd_d_true'],4)})",
                     f"{f(row['mean_d_grpo'],4)} (sd {f(row['sd_d_grpo'],4)})"])
    table(["estimator", "G", "mean dist to theta_true*", "mean dist to theta_GRPO*"], rows)

h("Per-instance regret table under phi1, sorted by sigma_r", 3)
p("G=16, 5 seeds. predicted gap = regret(theta_GRPO*) - regret(theta_true*) (both deterministic, "
  "no CI). Source: `results/followup_final_regret.json`, `time_only.per_instance`.")
rows = []
for row in fu_final["time_only"]["per_instance"]:
    pred_gap = row["regret_grpo_star"] - row["regret_true_star"]
    rows.append([row["instance"], f(row["sigma_r"], 5), f(row["regret_rloo_mean"], 4),
                 f(row["regret_grpo_mean"], 4), f(pred_gap, 4)])
table(["instance", "sigma_r", "regret RLOO (mean)", "regret GRPO (mean)", "predicted gap (theta_GRPO*-theta_true*)"], rows)

h("phi2 results (3 seeds) and G=4 results", 3)
p("phi2 (raw_params) theta distances already listed above. Regret-terms summary not computed "
  "for phi2 (only theta-distance table exists for phi2; regret-in-CI-terms was only run for "
  "kappa and phi1 in the final-regret follow-up). G=4 rows are included in the theta-distance "
  "tables above for all three feature sets; repeated here for phi1 for convenience. Source: "
  "`results/followup_c_grpo_misspecified.json`, `time_only.distance_table` (G=4 rows).")
rows = []
for row in fu_c["time_only"]["distance_table"]:
    if row["G"] == 4:
        rows.append([label[row["estimator"]], f"{f(row['mean_d_true'],4)} (sd {f(row['sd_d_true'],4)})",
                     f"{f(row['mean_d_grpo'],4)} (sd {f(row['sd_d_grpo'],4)})"])
table(["estimator (phi1, G=4)", "mean dist to theta_true*", "mean dist to theta_GRPO*"], rows)
p(f"phi1 seed spread (SEM) at G=4: {f(fu_c['time_only']['seed_spread_by_G']['4'],4)}; "
  f"at G=16: {f(fu_c['time_only']['seed_spread_by_G']['16'],4)}. "
  f"gap ({f(fu_c['time_only']['gap'],4)}) exceeds spread at G=16: "
  f"{fu_c['time_only']['gap_exceeds_seed_spread']['16']}; at G=4: "
  f"{fu_c['time_only']['gap_exceeds_seed_spread']['4']}. "
  "Source: `results/followup_c_grpo_misspecified.json`, `time_only.seed_spread_by_G` / `gap_exceeds_seed_spread`.")

h("MLP 20-seed results", 3)
p("lr=0.01 (chosen by sweep, see section b). Source: `results/followup_mlp_stability.json`, `summary` "
  "and `grpo_minus_rloo_bootstrap`.")
rows = []
for est in ["rloo", "drgrpo", "grpo"]:
    s = fu_mlp_stab["summary"][est]
    rows.append([label[est], f(s["median"], 4), f"[{f(s['q1'],4)}, {f(s['q3'],4)}]", f(s["iqr"], 4),
                 f(s["outlier_cutoff"], 4), f"{s['n_divergent']}/20"])
table(["estimator", "median regret", "IQR [Q1,Q3]", "IQR width", "outlier cutoff (Q3+1.5*IQR)", "divergent seeds"], rows)
gb = fu_mlp_stab["grpo_minus_rloo_bootstrap"]
p(f"**GRPO - RLOO bootstrap (4000 resamples): mean={f(gb['mean'],4)}, "
  f"95% CI=[{f(gb['ci_lo'],4)}, {f(gb['ci_hi'],4)}]** (excludes 0).")

h("Binary reward: base run + delta sweep", 3)
p("Base run (phi1, G=16, 5 seeds, delta chosen for pooled p in [0.2,0.5]). Source: "
  "`results/followup_binary_reward.json`.")
p(f"delta={f(fu_bin['delta'],4)}, pooled success rate at TWAP init={f(fu_bin['pooled_success_rate_twap_init'],3)}, "
  f"gap(theta_true*, theta_GRPO*_bin)={f(fu_bin['gap'],5)}.")
rows = []
ms = fu_bin["mean_regret_summary"]
for est in ["rloo", "drgrpo", "grpo"]:
    m = ms[est]
    rows.append([label[est], f(m["mean"], 4), f"[{f(m['ci_lo'],4)}, {f(m['ci_hi'],4)}]"])
rows.append(["theta_true* (ref)", f(ms["theta_true_star"]["mean"], 4), "--"])
rows.append(["theta_GRPO*_bin (ref)", f(ms["theta_grpo_star_bin"]["mean"], 4), "--"])
table(["estimator", "mean regret", "95% CI"], rows)

p("Delta sweep (RLOO/GRPO only, 5 seeds each). Source: `results/followup_binary_dose_response.json`.")
rows = []
for row in fu_bin_dose:
    rows.append([row["target_min_p"], f(row["delta"], 4), f(row["min_p"], 3), f(row["max_p"], 3),
                 f(row["weight_spread"], 3), f(row["predicted_gap"], 5),
                 f"{f(row['realized_gap_mean'],4)} [{f(row['realized_gap_ci'][0],4)}, {f(row['realized_gap_ci'][1],4)}]"])
table(["target min p", "delta", "min p", "max p", "weight spread (max/min)", "predicted gap", "realized gap (95% CI)"], rows)

h("Goodhart pilots: bug, magnitude, divergence", 3)
p("Linear policy, TWAP init, 3 seeds. divergence = 'divergence_step is not None' in the "
  "automated detector. Source: `results/pilot1_summary.json`.")
rows = []
for row in pilot1_summary:
    if row["capacity"] == "linear" and row["init"] == "twap":
        rows.append([row["bug"], row["c"], "yes" if row["divergence_step"] is not None else "no",
                     f(row["min_true_regret"], 4), f(row["final_true_regret"], 4)])
table(["bug", "magnitude c", "divergence flagged", "min true regret", "final true regret"], rows)

h("cosine(estimated, exact gradient) vs G, per estimator", 3)
p("kappa feature set, TWAP reference theta, 20 MC repeats per G. Source: "
  "`results/pilot2_grpo.json`, `cosine_vs_G`.")
rows = []
g_list = [r["G"] for r in pilot2["cosine_vs_G"]["rloo"]]
for i, gval in enumerate(g_list):
    row = [gval]
    for est in ["rloo", "drgrpo", "grpo"]:
        r = pilot2["cosine_vs_G"][est][i]
        row.append(f"{f(r['mean_cos'],4)} (se {f(r['se_cos'],4)})")
    rows.append(row)
table(["G", "RLOO cosine", "Dr.GRPO cosine", "GRPO cosine"], rows)
p()

# ===========================================================================
h("e. Known caveats and anomalies")

p("- **hash() seeding bug (fixed).** `pilot_goodhart.train_one`, `pilot_grpo.train_estimator`, "
  "and `pilot_grpo.cosine_vs_G` originally seeded RNGs with Python's built-in `hash()` on "
  "strings/tuples (e.g. `hash(estimator)`), which CPython randomizes per process by default -- "
  "so results were reproducible within one run but not across separate `python pilot_*.py` "
  "invocations. Fixed with a `zlib.crc32`-based `env.stable_hash`; verified two independent "
  "`python pilot_grpo.py` runs now produce byte-identical `results/pilot2_grpo.json`. All "
  "results in this file are post-fix and reproducible.")
p("- **B3 grad-clipping fix.** The first full Pilot-1 sweep, run without gradient clipping, "
  "blew up under bug B3 (theta and true regret reaching ~1e10) because policy outputs `a_k` "
  "are unclipped and the capped proxy cost removes any restoring force once trades exceed the "
  "cap. `clip_grad_norm_(..., max_norm=5.0)` was added to every training loop from that point "
  "on (confirmed identical across all bug types/experiments, see section b).")
p("- **theta_GRPO* ~= theta_true* under the kappa feature set** "
  f"(gap={f(pilot2['dist_true_grpo'],5)}, `results/pilot2_grpo.json`) despite 10-20x sigma_r "
  "heterogeneity across the 16 instances -- not a bug; Follow-up C's ablation (phi1/phi2) "
  f"reproduces a real, ~8x larger gap ({f(fu_c['time_only']['gap'],4)} / "
  f"{f(fu_c['raw_params']['gap'],4)} vs {f(fu_c['kappa']['gap'],4)}) that clears the G=16 "
  "seed-noise floor, supporting the reading that kappa-aware features suppress the bias by "
  "letting one shared theta fit every instance well.")
p("- **Binary reward structurally suppresses the same bias** even under phi1: "
  "`Var[r_bin]=p(1-p)` is bounded in [0,0.25], giving only ~1.3-3.2x weight spread across the "
  "delta sweep vs ~20x for continuous sigma_r (`results/followup_binary_dose_response.json`); "
  "predicted gap stays ~0 throughout.")
p("- **Extreme-delta binary training is unstable**, not a bias signal: at target min "
  "p~0.03 the realized GRPO-RLOO gap has a 95% CI of "
  f"[{f(fu_bin_dose[2]['realized_gap_ci'][0],4)}, {f(fu_bin_dose[2]['realized_gap_ci'][1],4)}] "
  "(`results/followup_binary_dose_response.json`, third row) -- a sparse-reward optimization "
  "failure mode (most G=16 groups see constant reward), not evidence about the fixed-point "
  "mechanism.")
p("- **MLP learning-rate sensitivity.** At lr=0.05 (tuned for the 3-parameter linear policy), "
  "MLP training for the ~4300-parameter network was highly unstable (median regret 0.48 at "
  "lr=0.05 vs 0.053 at lr=0.01, `results/followup_mlp_stability.json` `lr_sweep`), initially "
  "making the GRPO-vs-RLOO comparison inconclusive (`results/followup_mlp_regret.json`, pooled "
  "95% CI [-1.07, 1.07]). At the properly-scaled lr=0.01 with 20 seeds the same bias reappears "
  "cleanly (bootstrap CI excludes 0, see section d).")
p("- **Dr.GRPO theta-distance instability under binary reward.** In the binary-reward base run, "
  "Dr.GRPO's mean distance to `theta_true*`/`theta_GRPO*_bin` has an unusually wide CI "
  "(`results/followup_binary_reward.json`, `distance_table`) driven by one seed landing far "
  "from the others in theta-space despite a reasonable regret -- likely parameter "
  "non-identifiability in the 3-dim linear phi1 space (different theta vectors can produce "
  "near-identical induced schedules). Regret-based comparisons (the primary metric requested "
  "throughout) are unaffected.")
p("- **p (binary, at theta_true*) saturates near 1.0 for all 16 instances** in the setup table "
  "(section a) because theta_true* is already close to optimal relative to the (TWAP-calibrated, "
  "relatively loose) base delta=1.875 -- this is why the binary-reward fixed-point iteration "
  "for the base run is seeded from TWAP init, not theta_true*, matching where delta was "
  "actually calibrated (`followup_binary.py`, see inline comment).")
p()

# ===========================================================================
h("f. Exact-variance recomputation of GRPO fixed points")

p("`env.exact_var(policy, batch, s, phi, price_noise)` (new, in `env.py`) computes the "
  "exact closed-form variance of the true reward via the cost-to-go recursion "
  "`Z_N=c; Z_k=H_k+B_k*Z_{k+1}` (c=eta/tau, q=lam*sigma^2*tau, B_k=(1-a_k)^2, "
  "H_k=c*a_k^2+q*B_k), giving `Var(r)=X^4*(E[Z_1^2]-E[Z_1]^2)` (+ `sigma^2*tau*sum E[x_k^2]` "
  "if price noise is on). Verified against 1e5-rollout MC "
  f"(`tests/test_env.py::test_exact_var_matches_monte_carlo`, max rel err = "
  f"{f(test_diag['test_exact_var_matches_monte_carlo']['max_rel_err'],4)}). This section "
  "recomputes every `theta_GRPO*`-derived quantity using exact_var in place of the "
  "previous MC-estimated sigma_r (continuous) or MC-estimated p (binary, now a Gaussian "
  "approximation `p=Phi((E[r]-threshold)/sqrt(Var(r)))` using exact_J and exact_var). "
  "**No new training**: every RLOO/Dr.GRPO/GRPO endpoint reused as-is; only the "
  "deterministic fixed point `theta_GRPO*` (and quantities computed from it) changed. "
  "Source: `results/exact_var_recompute.json` (via `recompute_exact_grpo_star.py`).")

exact_recomp = load("exact_var_recompute.json")


def pct_change(old, new):
    if old is None or abs(old) < 1e-6:
        return "n/a (baseline ~0)"
    return f"{100*(new-old)/abs(old):+.1f}%"


def rel_change(old, new):
    return abs((new - old) / max(abs(old), 1e-9))


rows = []
for fs, fs_label in [("kappa", "kappa"), ("time_only", "phi1"), ("raw_params", "phi2")]:
    d = exact_recomp["continuous"][fs]
    flag_gap = "**FLAG**" if rel_change(d["gap_old"], d["gap_new"]) > 0.01 else ""
    rows.append([fs_label, "gap(theta_true*, theta_GRPO*)", f(d["gap_old"], 5), f(d["gap_new"], 5),
                 pct_change(d["gap_old"], d["gap_new"]), flag_gap])
    flag_regret = "**FLAG**" if rel_change(d["regret_grpo_star_old_mean"], d["regret_grpo_star_new_mean"]) > 0.01 else ""
    rows.append([fs_label, "regret(theta_GRPO*), mean over 16 instances",
                 f(d["regret_grpo_star_old_mean"], 5), f(d["regret_grpo_star_new_mean"], 5),
                 pct_change(d["regret_grpo_star_old_mean"], d["regret_grpo_star_new_mean"]), flag_regret])
table(["feature set", "quantity", "old (MC)", "new (exact)", "% change", "flag (>1%)"], rows)

p("Per-instance theta_GRPO*-endpoint distances (mean over 3 seeds), old (MC sigma_r) vs "
  "new (exact_var), for the largest-gap-change feature set (kappa):")
rows = []
by_est_G = {}
for row in exact_recomp["continuous"]["kappa"]["dist_recompute"]:
    key = (row["estimator"], row["G"])
    by_est_G.setdefault(key, []).append(row)
for (est, gval), rowlist in sorted(by_est_G.items(), key=lambda kv: (kv[0][0], kv[0][1])):
    d_old_mean = np.mean([r["d_grpo_old"] for r in rowlist])
    d_new_mean = np.mean([r["d_grpo_new"] for r in rowlist])
    pct = 100 * (d_new_mean - d_old_mean) / abs(d_old_mean) if abs(d_old_mean) > 1e-9 else float("nan")
    flag = "**FLAG**" if rel_change(d_old_mean, d_new_mean) > 0.01 else ""
    rows.append([label[est], gval, f(d_old_mean, 4), f(d_new_mean, 4), f"{pct:+.1f}%", flag])
table(["estimator", "G", "old mean dist to theta_GRPO*", "new mean dist to theta_GRPO*", "% change", "flag (>1%)"], rows)

p("Binary reward (phi1): predicted regret gap = regret(theta_GRPO*_bin) - regret(theta_true*), "
  "old (MC success rate) vs new (Gaussian approximation using exact_J/exact_var):")
rows = []
for name, d in exact_recomp["binary"].items():
    old_val = d["predicted_gap_old"]
    old_str = f"{old_val:.2e}" if old_val is not None else "n/a"
    rows.append([name, f(d["delta"], 4), old_str, f"{d['predicted_gap_new']:.2e}", "n/a (both ~0)"])
table(["run", "delta", "old predicted gap", "new predicted gap", "% change"], rows)

p()
p("**Summary of what changed by more than 1%:**")
p(f"- kappa's `gap(theta_true*, theta_GRPO*)` moved from "
  f"{f(exact_recomp['continuous']['kappa']['gap_old'],5)} (MC) to "
  f"{f(exact_recomp['continuous']['kappa']['gap_new'],5)} (exact) -- a "
  f"{pct_change(exact_recomp['continuous']['kappa']['gap_old'], exact_recomp['continuous']['kappa']['gap_new'])} "
  "change. This is now much closer to kappa's own G=16 seed-spread SEM "
  f"({fu_c['kappa']['seed_spread_by_G']['16']}), so the original \"clean negative, gap << "
  "seed spread\" conclusion for kappa holds by a much smaller margin than the MC estimate "
  "suggested (gap/spread ~17% under MC vs ~92% under exact) -- still below the spread, but "
  "no longer a wide margin.")
p(f"- phi2's gap changed by "
  f"{pct_change(exact_recomp['continuous']['raw_params']['gap_old'], exact_recomp['continuous']['raw_params']['gap_new'])}, "
  "just over the 1% flag line; phi1's gap changed by "
  f"{pct_change(exact_recomp['continuous']['time_only']['gap_old'], exact_recomp['continuous']['time_only']['gap_new'])}, "
  "not flagged.")
p("- `regret(theta_GRPO*)` (the deterministic reference value, mean over 16 instances) "
  "changed by <1% for all three continuous feature sets -- the fixed point moved further "
  "in theta-space (for kappa) but landed at essentially the same regret, consistent with "
  "the flat/aligned-gradient picture used throughout to explain kappa's small effect size.")
p("- The binary-reward predicted gap is unchanged in substance (both old and new are "
  "~1e-4 to 1e-7, i.e. indistinguishable from 0) across the base run and all 3 dose-response "
  "deltas -- the \"binary reward suppresses the bias\" conclusion is confirmed under the "
  "exact/Gaussian-approximation method too, not just under raw MC.")
p()

h("Was price-noise on in the final runs?", 3)
p("**No.** `price_noise=False` (the default) is used in every training loop and every "
  "`theta_GRPO*`/`theta_true*` computation that feeds into the results reported throughout "
  "this file (`pilot_goodhart.train_one`, `pilot_grpo.train_estimator`, "
  "`followup_c.py`, `followup_final.py`, `followup_binary.py`'s `train_estimator_binary`, "
  "`followup_mlp.py`'s `train_estimator_mlp`, and this section's `recompute_exact_grpo_star.py`). "
  "`price_noise=True` appears exactly once in the whole codebase: a standalone robustness "
  "check in `pilot_grpo.py`'s `__main__` block that computes a separate "
  "`theta_grpo_star_price_noise` "
  f"(`results/pilot2_grpo.json`, dist to theta_true* = "
  f"{f(pilot2['dist_true_grpo_price_noise'],5)}, vs {f(pilot2['dist_true_grpo'],5)} without) "
  "purely to check the kappa clean-negative conclusion was robust to price noise. That "
  "variant was never used for training, for any Follow-up/Extension, or for any number "
  "reported in sections a-e of this file.")
p()

with open("paper_inputs.md", "w") as fh:
    fh.write("\n".join(lines))
print("paper_inputs.md written, lines=", len(lines))
