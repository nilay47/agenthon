# paper_inputs.md

Compiled inputs for writing the paper: every number in this file is copied from a file under `results/*.json` (source cited per table/value; field names match the JSON keys verbatim) or from the named source-code constants in `env.py` / `policy.py` / `pilot_goodhart.py` / `pilot_grpo.py`. No experiments were run to produce this file -- `capture_test_diagnostics.py` and `capture_fixed_instances.py` only re-record already-deterministic quantities (the tests' internal numbers, and the 16 fixed instances' sampled parameters) that weren't previously saved to `results/`.

## a. Setup details

### AC / environment parameters

Source: `env.py` module constants.
| parameter | value |
|---|---|
| N (steps) | 20 |
| T (horizon) | 1.0 |
| tau = T/N | 0.05 |
| gamma (permanent impact) | 0 |
| eps (spread) | 0 |
| policy std s (fixed, not learned) | 0.05  (`policy.DEFAULT_S`) |

### Instance sampler ranges

Source: `env.py`, `sample_instances()`. X, sigma ~ Uniform; lam, eta ~ log-Uniform.
| param | range |
|---|---|
| X | [0.5, 2.0] |
| sigma | [0.1, 0.6] |
| lam | [0.1, 10.0] (log-uniform) |
| eta | [0.01, 0.1] (log-uniform) |

### The 16 fixed instances

Source: `results/fixed_instances.json` (`FIXED_SEED=777`, `pilot_grpo.get_fixed_batch()`; kappa via `env.kappa_batch`). sigma_r (continuous reward, kappa feature set, at `theta_true*`): `results/pilot2_grpo.json`, `regret_table[*].sigma_r`. p (binary reward, phi1, at `theta_true*`): `results/followup_binary_reward.json`, `per_instance[*].success_rate_theta_true_star`.
| instance | X | sigma | lam | eta | kappa | sigma_r (continuous, kappa feat.) | p (binary, phi1) |
|---|---|---|---|---|---|---|---|
| 0 | 1.4166 | 0.1232 | 0.1837 | 0.0257 | 0.3295 | 0.00792 | 1.000 |
| 1 | 1.0742 | 0.5660 | 0.2131 | 0.0209 | 1.8055 | 0.00336 | 1.000 |
| 2 | 1.4001 | 0.3108 | 0.4188 | 0.0800 | 0.7110 | 0.02298 | 1.000 |
| 3 | 1.9453 | 0.1169 | 0.1852 | 0.0118 | 0.4636 | 0.00675 | 1.000 |
| 4 | 0.7942 | 0.2099 | 0.2068 | 0.0639 | 0.3778 | 0.00615 | 1.000 |
| 5 | 1.0056 | 0.3449 | 0.1221 | 0.0593 | 0.4948 | 0.00898 | 1.000 |
| 6 | 1.3527 | 0.2616 | 0.4999 | 0.0786 | 0.6598 | 0.02097 | 1.000 |
| 7 | 1.3094 | 0.1516 | 0.1669 | 0.0520 | 0.2718 | 0.01362 | 1.000 |
| 8 | 1.6041 | 0.5133 | 0.2042 | 0.0576 | 0.9666 | 0.02154 | 1.000 |
| 9 | 1.1462 | 0.1640 | 3.3036 | 0.0672 | 1.1497 | 0.01256 | 1.000 |
| 10 | 0.5863 | 0.2000 | 0.2860 | 0.0548 | 0.4570 | 0.00279 | 1.000 |
| 11 | 0.7491 | 0.1850 | 0.8423 | 0.0437 | 0.8121 | 0.00362 | 1.000 |
| 12 | 1.6361 | 0.5460 | 2.4143 | 0.0281 | 5.0489 | 0.00864 | 1.000 |
| 13 | 1.5199 | 0.4693 | 2.4513 | 0.0641 | 2.9010 | 0.01907 | 1.000 |
| 14 | 1.5523 | 0.4890 | 4.9758 | 0.0281 | 6.4829 | 0.00798 | 1.000 |
| 15 | 0.6579 | 0.1372 | 5.0845 | 0.0225 | 2.0624 | 0.00135 | 1.000 |

### Policy parametrization

`n_k = x_{k-1} * a_k`, `a_k = m_k + s*z_k` (k<N), `a_N=1` forced (true sim). `m_k = theta . phi(t_k, instance)` (LinearPolicy) or `MLP(phi(t_k, instance))` (MLPPolicy). Source: `policy.py`, `env.py FEATURE_SETS`.
| feature set | n_feat | definition |
|---|---|---|
| kappa (spec default) | 5 | [1, t/T, (t/T)^2, kappaT, kappaT*t/T] |
| phi1 = time_only | 3 | [1, t/T, (t/T)^2] |
| phi2 = raw_params | 6 | [1, t/T, (t/T)^2, log sigma, log lam, log eta] |
| time_scalar (MLP input) | 1 | [t/T]  (MLP supplies its own bias) |

MLP policy (Extensions 2): 2 hidden layers x 64, tanh, input=time_scalar. Source: `results/followup_mlp_regret.json`, `hidden_sizes`=[64, 64].

## b. Training details per experiment

| experiment | estimators | G | steps | optimizer | LR | seeds | init | grad clip |
|---|---|---|---|---|---|---|---|---|
| Pilot 1 (Goodhart) | RLOO | 16 | 400 (eval every 10) | Adam | 0.05 | [0,1,2] | TWAP + random (linear); random (MLP) | 5.0 (`clip_grad_norm_`) |
| Pilot 2 (GRPO, kappa) | RLOO, Dr.GRPO, GRPO | {4, 16} | 1500 | Adam | 0.05 | [0,1,2] | TWAP | 5.0 |
| Follow-up C (phi1/phi2/kappa) | RLOO, Dr.GRPO, GRPO | {4, 16} | 1500 | Adam | 0.05 | [0,1,2] | TWAP | 5.0 |
| Follow-up D / final regret (kappa, phi1) | RLOO, Dr.GRPO, GRPO | 16 only | 1500 | Adam | 0.05 | [0,1,2,3,4] | TWAP | 5.0 |
| Extension 1 binary (base) | RLOO, Dr.GRPO, GRPO | 16 | 1500 | Adam | 0.05 | [0,1,2,3,4] | TWAP | 5.0 |
| Extension 1 dose-response | RLOO, GRPO | 16 | 1500 | Adam | 0.05 | [0,1,2,3,4] | TWAP | 5.0 |
| Extension 2 MLP (initial) | RLOO, Dr.GRPO, GRPO | 16 | 1500 | Adam | 0.05 | [0,1,2,3,4] | small random (std 0.1) | 5.0 |
| Extension 2 MLP LR sweep | RLOO only | 16 | 1500 | Adam | {0.01, 0.02, 0.05} | [0,1,2] | small random (std 0.1) | 5.0 |
| Extension 2 MLP (20 seeds) | RLOO, Dr.GRPO, GRPO | 16 | 1500 | Adam | 0.01 (chosen) | 0..19 | small random (std 0.1) | 5.0 |

Sources: `pilot_goodhart.py` (N_TRAIN_PER_STEP=64 instances/step, G_TRAIN=16, LR=0.05, N_STEPS=400, SEEDS=[0,1,2]); `pilot_grpo.py` (N_FIXED_INSTANCES=16, FIXED_SEED=777, N_TRAIN_STEPS=1500, LR=0.05, SEEDS=[0,1,2], G_VALUES=[4,16], SIGMA_R_MC=10000, N_OUTER_FIXED_POINT=8, N_INNER_PER_OUTER=400); `followup_final.py`, `followup_binary.py`, `followup_binary_dose.py`, `followup_mlp.py`, `followup_mlp_stability.py` (SEEDS5=[0,1,2,3,4], FULL_SEEDS=range(20)). Grad clipping `torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=5.0)` is applied identically in every training loop (`pilot_goodhart.train_one`, `pilot_grpo.train_estimator`, `followup_binary.train_estimator_binary`, `followup_mlp.train_estimator_mlp`).

### Goodhart bug definitions and magnitudes (Pilot 1)

Source: `pilot_goodhart.py`, `BUGS` dict.
| bug | magnitudes c |
|---|---|
| B1 (weak terminal penalty) | [0.001, 0.01, 0.1] |
| B2 (one-step cost lag, blend in [0,1]) | [0.3, 0.7, 1.0] |
| B3 (per-step impact cap) | [0.002, 0.005, 0.02] |

### Binary-reward delta values

delta such that `r_bin = 1[r_true >= r*_inst - delta*|r*_inst|]`, r*_inst = -ac_cost_inst. Source: `results/followup_binary_reward.json` (base run, pooled target); `results/followup_binary_dose_response.json` (dose sweep, min-p target).
| run | target | delta | pooled/min p achieved |
|---|---|---|---|
| base | pooled p in [0.2,0.5] | 1.8751 | 0.364 (pooled) |
| dose 1 | min p ~ 0.3 | 2.3176 | 0.307 (min) |
| dose 2 | min p ~ 0.1 | 1.8402 | 0.103 (min) |
| dose 3 | min p ~ 0.03 | 1.4611 | 0.025 (min) |

### MLP LR sweep (Extension 2)

Source: `results/followup_mlp_stability.json`, `lr_sweep` (RLOO only, 3 seeds each).
| lr | seed regrets | median | IQR |
|---|---|---|---|
| 0.01 | 0.0525, 0.0530, 0.0531 | 0.0530 | 0.0003 |
| 0.02 | 0.0578, 0.0543, 0.0523 | 0.0543 | 0.0027 |
| 0.05 | 0.1005, 0.7275, 0.4813 | 0.4813 | 0.3135 |

**Chosen LR: 0.01** (lowest median, source: `chosen_lr` field).

## c. Tests

Source: `tests/test_env.py` (assertions) and `results/test_diagnostics.json` (measured values, captured by `capture_test_diagnostics.py` re-running the exact same deterministic code/seeds as the tests, without changing them).
| test | what it checks | tolerance | measured |
|---|---|---|---|
| test_exact_J_matches_monte_carlo | exact_J agrees with a 1e5-rollout MC estimate (5 instances x 3 random thetas) | diff <= 3*SE | max |diff|/SE = 1.458 (limit 3.0) |
| test_ac_closed_form_is_optimal | AC closed-form schedule cost vs scipy-optimized free schedule (5 instances, 6 restarts) | rel_gap <= 1e-6 | max rel_gap = -1.17e-11 |
| test_autograd_matches_finite_difference | autograd gradient of exact_J vs central finite differences (eps=1e-6, 4 instances) | rtol=1e-4, atol=1e-6 (np.testing.assert_allclose) | max abs diff = 1.57e-05, max rel err = 2.89e-10 |
| test_drgrpo_is_exact_factor_of_rloo | Dr.GRPO advantage == (G-1)/G * RLOO advantage exactly (20 groups, G=8) | rtol=1e-10 | max abs diff = 4.44e-16 |
| test_rloo_mean_gradient_matches_exact_gradient | RLOO's mean estimated policy gradient (G=4000) vs exact grad of sum_i J_i (6 instances) | rel_err < 0.1 | rel_err = 0.0194 |
| test_grpo_mean_gradient_matches_inverse_std_weighted_target | GRPO's mean estimated gradient (G=4000) vs sum_i grad J_i / sigma_r_i (5 instances) | cosine > 0.9 | cosine = 1.0000 |
| test_exact_var_matches_monte_carlo | closed-form exact_var (Z_k cost-to-go recursion) vs 1e5-rollout MC sample variance (5 instances x 3 random thetas) | diff <= 3*SE(sample_var), SE via delta method | max rel err = 0.0135 |

All 7 tests pass (`pytest tests/ -q` -> `7 passed`).

## d. Results

### Main regret table: kappa vs phi1 x {RLOO, Dr.GRPO, GRPO, theta_true*, theta_GRPO*}

G=16, 5 seeds. Mean regret over the 16 instances, 95% CI over seeds (t-interval, df=4). Source: `results/followup_final_regret.json`, `<feature_set>.mean_regret_summary`.
| feature set | estimator | mean regret | 95% CI |
|---|---|---|---|
| kappa | RLOO | 0.0278 | [0.0254, 0.0302] |
| kappa | Dr.GRPO | 0.0263 | [0.0251, 0.0275] |
| kappa | GRPO | 0.0269 | [0.0243, 0.0296] |
| kappa | theta_true* (ref) | 0.0245 | -- |
| kappa | theta_GRPO* (ref) | 0.0242 | -- |
| phi1 | RLOO | 0.0549 | [0.0516, 0.0581] |
| phi1 | Dr.GRPO | 0.0570 | [0.0498, 0.0643] |
| phi1 | GRPO | 0.0790 | [0.0589, 0.0992] |
| phi1 | theta_true* (ref) | 0.0515 | -- |
| phi1 | theta_GRPO* (ref) | 0.0607 | -- |

### Theta distances (endpoint to theta_true* and theta_GRPO*) per feature set

G in {4,16}, 3 seeds. Source: `results/followup_c_grpo_misspecified.json`, `<feature_set>.distance_table` (mean +/- sd over 3 seeds); kappa gap and G=16 reference also cross-checked against `results/pilot2_grpo.json` (gap=0.00337).
**kappa** (n_feat=5, gap(theta_true*, theta_GRPO*)=0.00337):
| estimator | G | mean dist to theta_true* | mean dist to theta_GRPO* |
|---|---|---|---|
| RLOO | 4 | 0.1084 (sd 0.0306) | 0.1090 (sd 0.0314) |
| RLOO | 16 | 0.0340 (sd 0.0095) | 0.0339 (sd 0.0114) |
| Dr.GRPO | 4 | 0.1011 (sd 0.0146) | 0.1006 (sd 0.0155) |
| Dr.GRPO | 16 | 0.0339 (sd 0.0212) | 0.0334 (sd 0.0207) |
| GRPO | 4 | 0.1461 (sd 0.1454) | 0.1467 (sd 0.1454) |
| GRPO | 16 | 0.0301 (sd 0.0157) | 0.0298 (sd 0.0152) |

**phi1** (n_feat=3, gap(theta_true*, theta_GRPO*)=0.02732):
| estimator | G | mean dist to theta_true* | mean dist to theta_GRPO* |
|---|---|---|---|
| RLOO | 4 | 0.1623 (sd 0.1837) | 0.1542 (sd 0.1870) |
| RLOO | 16 | 0.0596 (sd 0.0240) | 0.0537 (sd 0.0236) |
| Dr.GRPO | 4 | 0.2698 (sd 0.2379) | 0.2724 (sd 0.2262) |
| Dr.GRPO | 16 | 0.0393 (sd 0.0076) | 0.0505 (sd 0.0096) |
| GRPO | 4 | 0.1040 (sd 0.0638) | 0.0995 (sd 0.0638) |
| GRPO | 16 | 0.0522 (sd 0.0174) | 0.0436 (sd 0.0110) |

**phi2** (n_feat=6, gap(theta_true*, theta_GRPO*)=0.02708):
| estimator | G | mean dist to theta_true* | mean dist to theta_GRPO* |
|---|---|---|---|
| RLOO | 4 | 0.0794 (sd 0.0232) | 0.0920 (sd 0.0317) |
| RLOO | 16 | 0.0307 (sd 0.0066) | 0.0386 (sd 0.0143) |
| Dr.GRPO | 4 | 0.1221 (sd 0.0502) | 0.1400 (sd 0.0508) |
| Dr.GRPO | 16 | 0.0378 (sd 0.0017) | 0.0461 (sd 0.0050) |
| GRPO | 4 | 0.0739 (sd 0.0385) | 0.0821 (sd 0.0286) |
| GRPO | 16 | 0.0399 (sd 0.0049) | 0.0325 (sd 0.0185) |

### Per-instance regret table under phi1, sorted by sigma_r

G=16, 5 seeds. predicted gap = regret(theta_GRPO*) - regret(theta_true*) (both deterministic, no CI). Source: `results/followup_final_regret.json`, `time_only.per_instance`.
| instance | sigma_r | regret RLOO (mean) | regret GRPO (mean) | predicted gap (theta_GRPO*-theta_true*) |
|---|---|---|---|---|
| 15 | 0.00129 | 0.0032 | 0.0051 | 0.0005 |
| 1 | 0.00340 | 0.0090 | 0.0111 | -0.0002 |
| 10 | 0.00365 | 0.0123 | 0.0080 | -0.0035 |
| 11 | 0.00457 | 0.0145 | 0.0101 | -0.0039 |
| 4 | 0.00794 | 0.0266 | 0.0171 | -0.0077 |
| 3 | 0.00861 | 0.0290 | 0.0188 | -0.0083 |
| 0 | 0.01025 | 0.0343 | 0.0220 | -0.0099 |
| 5 | 0.01169 | 0.0388 | 0.0253 | -0.0110 |
| 9 | 0.01496 | 0.0458 | 0.0357 | -0.0108 |
| 7 | 0.01750 | 0.0598 | 0.0382 | -0.0173 |
| 13 | 0.02261 | 0.0449 | 0.1367 | 0.0428 |
| 8 | 0.02654 | 0.0830 | 0.0601 | -0.0214 |
| 6 | 0.02723 | 0.0893 | 0.0597 | -0.0249 |
| 2 | 0.02945 | 0.0960 | 0.0649 | -0.0265 |
| 12 | 0.04381 | 0.0970 | 0.2773 | 0.0966 |
| 14 | 0.07124 | 0.1946 | 0.4740 | 0.1524 |

### phi2 results (3 seeds) and G=4 results

phi2 (raw_params) theta distances already listed above. Regret-terms summary not computed for phi2 (only theta-distance table exists for phi2; regret-in-CI-terms was only run for kappa and phi1 in the final-regret follow-up). G=4 rows are included in the theta-distance tables above for all three feature sets; repeated here for phi1 for convenience. Source: `results/followup_c_grpo_misspecified.json`, `time_only.distance_table` (G=4 rows).
| estimator (phi1, G=4) | mean dist to theta_true* | mean dist to theta_GRPO* |
|---|---|---|
| RLOO | 0.1623 (sd 0.1837) | 0.1542 (sd 0.1870) |
| Dr.GRPO | 0.2698 (sd 0.2379) | 0.2724 (sd 0.2262) |
| GRPO | 0.1040 (sd 0.0638) | 0.0995 (sd 0.0638) |

phi1 seed spread (SEM) at G=4: 0.1176; at G=16: 0.0253. gap (0.0273) exceeds spread at G=16: True; at G=4: False. Source: `results/followup_c_grpo_misspecified.json`, `time_only.seed_spread_by_G` / `gap_exceeds_seed_spread`.

### MLP 20-seed results

lr=0.01 (chosen by sweep, see section b). Source: `results/followup_mlp_stability.json`, `summary` and `grpo_minus_rloo_bootstrap`.
| estimator | median regret | IQR [Q1,Q3] | IQR width | outlier cutoff (Q3+1.5*IQR) | divergent seeds |
|---|---|---|---|---|---|
| RLOO | 0.0529 | [0.0524, 0.0533] | 0.0008 | 0.0545 | 4/20 |
| Dr.GRPO | 0.0529 | [0.0525, 0.0539] | 0.0014 | 0.0559 | 4/20 |
| GRPO | 0.0622 | [0.0556, 0.0658] | 0.0102 | 0.0811 | 1/20 |

**GRPO - RLOO bootstrap (4000 resamples): mean=0.0079, 95% CI=[0.0046, 0.0116]** (excludes 0).

### Binary reward: base run + delta sweep

Base run (phi1, G=16, 5 seeds, delta chosen for pooled p in [0.2,0.5]). Source: `results/followup_binary_reward.json`.
delta=1.8751, pooled success rate at TWAP init=0.364, gap(theta_true*, theta_GRPO*_bin)=0.00015.
| estimator | mean regret | 95% CI |
|---|---|---|
| RLOO | 0.0724 | [0.0555, 0.0893] |
| Dr.GRPO | 0.0804 | [0.0602, 0.1006] |
| GRPO | 0.0700 | [0.0524, 0.0877] |
| theta_true* (ref) | 0.0515 | -- |
| theta_GRPO*_bin (ref) | 0.0515 | -- |

Delta sweep (RLOO/GRPO only, 5 seeds each). Source: `results/followup_binary_dose_response.json`.
| target min p | delta | min p | max p | weight spread (max/min) | predicted gap | realized gap (95% CI) |
|---|---|---|---|---|---|---|
| 0.3 | 2.3176 | 0.307 | 0.825 | 1.316 | -0.00000 | -0.0025 [-0.0119, 0.0069] |
| 0.1 | 1.8402 | 0.103 | 0.652 | 1.635 | -0.00000 | 0.0038 [-0.0230, 0.0307] |
| 0.03 | 1.4611 | 0.025 | 0.436 | 3.176 | -0.00000 | -1.1884 [-4.4568, 2.0800] |

### Goodhart pilots: bug, magnitude, divergence

Linear policy, TWAP init, 3 seeds. divergence = 'divergence_step is not None' in the automated detector. Source: `results/pilot1_summary.json`.
| bug | magnitude c | divergence flagged | min true regret | final true regret |
|---|---|---|---|---|
| B1 | 0.001 | yes | 0.1514 | 0.3398 |
| B1 | 0.01 | yes | 0.0956 | 0.3682 |
| B1 | 0.1 | yes | 0.0398 | 0.1131 |
| B2 | 0.3 | no | 0.0261 | 0.0349 |
| B2 | 0.7 | yes | 0.0279 | 0.0340 |
| B2 | 1.0 | yes | 0.1771 | 0.3731 |
| B3 | 0.002 | yes | 0.3194 | 1.4141 |
| B3 | 0.005 | yes | 0.3194 | 1.4763 |
| B3 | 0.02 | yes | 0.3194 | 1.3781 |

### cosine(estimated, exact gradient) vs G, per estimator

kappa feature set, TWAP reference theta, 20 MC repeats per G. Source: `results/pilot2_grpo.json`, `cosine_vs_G`.
| G | RLOO cosine | Dr.GRPO cosine | GRPO cosine |
|---|---|---|---|
| 2 | 0.8308 (se 0.0235) | 0.8614 (se 0.0265) | 0.9589 (se 0.0127) |
| 4 | 0.9444 (se 0.0132) | 0.9307 (se 0.0142) | 0.9807 (se 0.0053) |
| 8 | 0.9580 (se 0.0106) | 0.9617 (se 0.0101) | 0.9942 (se 0.0008) |
| 16 | 0.9747 (se 0.0063) | 0.9823 (se 0.0046) | 0.9954 (se 0.0009) |
| 32 | 0.9884 (se 0.0028) | 0.9924 (se 0.0018) | 0.9983 (se 0.0004) |
| 64 | 0.9965 (se 0.0010) | 0.9962 (se 0.0006) | 0.9992 (se 0.0002) |
| 128 | 0.9972 (se 0.0004) | 0.9974 (se 0.0005) | 0.9996 (se 0.0001) |
| 256 | 0.9993 (se 0.0001) | 0.9988 (se 0.0002) | 0.9998 (se 0.0000) |


## e. Known caveats and anomalies

- **hash() seeding bug (fixed).** `pilot_goodhart.train_one`, `pilot_grpo.train_estimator`, and `pilot_grpo.cosine_vs_G` originally seeded RNGs with Python's built-in `hash()` on strings/tuples (e.g. `hash(estimator)`), which CPython randomizes per process by default -- so results were reproducible within one run but not across separate `python pilot_*.py` invocations. Fixed with a `zlib.crc32`-based `env.stable_hash`; verified two independent `python pilot_grpo.py` runs now produce byte-identical `results/pilot2_grpo.json`. All results in this file are post-fix and reproducible.
- **B3 grad-clipping fix.** The first full Pilot-1 sweep, run without gradient clipping, blew up under bug B3 (theta and true regret reaching ~1e10) because policy outputs `a_k` are unclipped and the capped proxy cost removes any restoring force once trades exceed the cap. `clip_grad_norm_(..., max_norm=5.0)` was added to every training loop from that point on (confirmed identical across all bug types/experiments, see section b).
- **theta_GRPO* ~= theta_true* under the kappa feature set** (gap=0.00337, `results/pilot2_grpo.json`) despite 10-20x sigma_r heterogeneity across the 16 instances -- not a bug; Follow-up C's ablation (phi1/phi2) reproduces a real, ~8x larger gap (0.0273 / 0.0271 vs 0.0034) that clears the G=16 seed-noise floor, supporting the reading that kappa-aware features suppress the bias by letting one shared theta fit every instance well.
- **Binary reward structurally suppresses the same bias** even under phi1: `Var[r_bin]=p(1-p)` is bounded in [0,0.25], giving only ~1.3-3.2x weight spread across the delta sweep vs ~20x for continuous sigma_r (`results/followup_binary_dose_response.json`); predicted gap stays ~0 throughout.
- **Extreme-delta binary training is unstable**, not a bias signal: at target min p~0.03 the realized GRPO-RLOO gap has a 95% CI of [-4.4568, 2.0800] (`results/followup_binary_dose_response.json`, third row) -- a sparse-reward optimization failure mode (most G=16 groups see constant reward), not evidence about the fixed-point mechanism.
- **MLP learning-rate sensitivity.** At lr=0.05 (tuned for the 3-parameter linear policy), MLP training for the ~4300-parameter network was highly unstable (median regret 0.48 at lr=0.05 vs 0.053 at lr=0.01, `results/followup_mlp_stability.json` `lr_sweep`), initially making the GRPO-vs-RLOO comparison inconclusive (`results/followup_mlp_regret.json`, pooled 95% CI [-1.07, 1.07]). At the properly-scaled lr=0.01 with 20 seeds the same bias reappears cleanly (bootstrap CI excludes 0, see section d).
- **Dr.GRPO theta-distance instability under binary reward.** In the binary-reward base run, Dr.GRPO's mean distance to `theta_true*`/`theta_GRPO*_bin` has an unusually wide CI (`results/followup_binary_reward.json`, `distance_table`) driven by one seed landing far from the others in theta-space despite a reasonable regret -- likely parameter non-identifiability in the 3-dim linear phi1 space (different theta vectors can produce near-identical induced schedules). Regret-based comparisons (the primary metric requested throughout) are unaffected.
- **p (binary, at theta_true*) saturates near 1.0 for all 16 instances** in the setup table (section a) because theta_true* is already close to optimal relative to the (TWAP-calibrated, relatively loose) base delta=1.875 -- this is why the binary-reward fixed-point iteration for the base run is seeded from TWAP init, not theta_true*, matching where delta was actually calibrated (`followup_binary.py`, see inline comment).

## f. Exact-variance recomputation of GRPO fixed points

`env.exact_var(policy, batch, s, phi, price_noise)` (new, in `env.py`) computes the exact closed-form variance of the true reward via the cost-to-go recursion `Z_N=c; Z_k=H_k+B_k*Z_{k+1}` (c=eta/tau, q=lam*sigma^2*tau, B_k=(1-a_k)^2, H_k=c*a_k^2+q*B_k), giving `Var(r)=X^4*(E[Z_1^2]-E[Z_1]^2)` (+ `sigma^2*tau*sum E[x_k^2]` if price noise is on). Verified against 1e5-rollout MC (`tests/test_env.py::test_exact_var_matches_monte_carlo`, max rel err = 0.0135). This section recomputes every `theta_GRPO*`-derived quantity using exact_var in place of the previous MC-estimated sigma_r (continuous) or MC-estimated p (binary, now a Gaussian approximation `p=Phi((E[r]-threshold)/sqrt(Var(r)))` using exact_J and exact_var). **No new training**: every RLOO/Dr.GRPO/GRPO endpoint reused as-is; only the deterministic fixed point `theta_GRPO*` (and quantities computed from it) changed. Source: `results/exact_var_recompute.json` (via `recompute_exact_grpo_star.py`).
| feature set | quantity | old (MC) | new (exact) | % change | flag (>1%) |
|---|---|---|---|---|---|
| kappa | gap(theta_true*, theta_GRPO*) | 0.00337 | 0.01801 | +434.6% | **FLAG** |
| kappa | regret(theta_GRPO*), mean over 16 instances | 0.02423 | 0.02426 | +0.1% |  |
| phi1 | gap(theta_true*, theta_GRPO*) | 0.02732 | 0.02741 | +0.3% |  |
| phi1 | regret(theta_GRPO*), mean over 16 instances | 0.06072 | 0.06079 | +0.1% |  |
| phi2 | gap(theta_true*, theta_GRPO*) | 0.02708 | 0.02743 | +1.3% | **FLAG** |
| phi2 | regret(theta_GRPO*), mean over 16 instances | 0.02961 | 0.02958 | -0.1% |  |

Per-instance theta_GRPO*-endpoint distances (mean over 3 seeds), old (MC sigma_r) vs new (exact_var), for the largest-gap-change feature set (kappa):
| estimator | G | old mean dist to theta_GRPO* | new mean dist to theta_GRPO* | % change | flag (>1%) |
|---|---|---|---|---|---|
| Dr.GRPO | 4 | 0.1006 | 0.1020 | +1.4% | **FLAG** |
| Dr.GRPO | 16 | 0.0334 | 0.0423 | +26.8% | **FLAG** |
| GRPO | 4 | 0.1467 | 0.1399 | -4.6% | **FLAG** |
| GRPO | 16 | 0.0298 | 0.0246 | -17.6% | **FLAG** |
| RLOO | 4 | 0.1090 | 0.1038 | -4.7% | **FLAG** |
| RLOO | 16 | 0.0339 | 0.0308 | -9.2% | **FLAG** |

Binary reward (phi1): predicted regret gap = regret(theta_GRPO*_bin) - regret(theta_true*), old (MC success rate) vs new (Gaussian approximation using exact_J/exact_var):
| run | delta | old predicted gap | new predicted gap | % change |
|---|---|---|---|---|
| base | 1.8751 | -3.89e-07 | -3.89e-07 | n/a (both ~0) |
| dose_min_p_0.3 | 2.3176 | -3.89e-07 | -3.89e-07 | n/a (both ~0) |
| dose_min_p_0.1 | 1.8402 | -3.89e-07 | -3.89e-07 | n/a (both ~0) |
| dose_min_p_0.03 | 1.4611 | -3.28e-07 | -3.89e-07 | n/a (both ~0) |


**Summary of what changed by more than 1%:**
- kappa's `gap(theta_true*, theta_GRPO*)` moved from 0.00337 (MC) to 0.01801 (exact) -- a +434.6% change. This is now much closer to kappa's own G=16 seed-spread SEM (0.019666737866135316), so the original "clean negative, gap << seed spread" conclusion for kappa holds by a much smaller margin than the MC estimate suggested (gap/spread ~17% under MC vs ~92% under exact) -- still below the spread, but no longer a wide margin.
- phi2's gap changed by +1.3%, just over the 1% flag line; phi1's gap changed by +0.3%, not flagged.
- `regret(theta_GRPO*)` (the deterministic reference value, mean over 16 instances) changed by <1% for all three continuous feature sets -- the fixed point moved further in theta-space (for kappa) but landed at essentially the same regret, consistent with the flat/aligned-gradient picture used throughout to explain kappa's small effect size.
- The binary-reward predicted gap is unchanged in substance (both old and new are ~1e-4 to 1e-7, i.e. indistinguishable from 0) across the base run and all 3 dose-response deltas -- the "binary reward suppresses the bias" conclusion is confirmed under the exact/Gaussian-approximation method too, not just under raw MC.

### Was price-noise on in the final runs?

**No.** `price_noise=False` (the default) is used in every training loop and every `theta_GRPO*`/`theta_true*` computation that feeds into the results reported throughout this file (`pilot_goodhart.train_one`, `pilot_grpo.train_estimator`, `followup_c.py`, `followup_final.py`, `followup_binary.py`'s `train_estimator_binary`, `followup_mlp.py`'s `train_estimator_mlp`, and this section's `recompute_exact_grpo_star.py`). `price_noise=True` appears exactly once in the whole codebase: a standalone robustness check in `pilot_grpo.py`'s `__main__` block that computes a separate `theta_grpo_star_price_noise` (`results/pilot2_grpo.json`, dist to theta_true* = 0.00587, vs 0.00337 without) purely to check the kappa clean-negative conclusion was robust to price noise. That variant was never used for training, for any Follow-up/Extension, or for any number reported in sections a-e of this file.

## Figures

All figures are in `paper_figs/` as both `.png` and `.pdf` (vector). No new experiments were run to produce the PDFs -- they re-render the exact same `results/*.json` data already used for the PNGs in `figs/`.

- **`paper_figs/01_goodhart_proxy_grid.pdf`** (+ `.png`)
  - Shows: Pilot 1: proxy reward vs training step, one panel per (bug, magnitude), linear vs MLP capacity, mean+range over 3 seeds
  - Axes: x: train step (0-400); y: proxy J (buggy training objective)
  - Suggested caption: "Proxy reward saturates within ~30-50 steps for every bug/magnitude, regardless of policy capacity."
- **`paper_figs/02_goodhart_regret_grid.pdf`** (+ `.png`)
  - Shows: Pilot 1: true regret vs training step, one panel per (bug, magnitude), linear vs MLP capacity, mean+range over 3 seeds
  - Axes: x: train step (0-400); y: true regret (cost minus AC-optimal cost)
  - Suggested caption: "Bug B3 (impact cap) drives true regret to a high plateau almost immediately; B2 stays low except at the full bug magnitude."
- **`paper_figs/03_b3_sanity_stepcurves.pdf`** (+ `.png`)
  - Shows: Follow-up A: B3, per-seed (not aggregated) proxy J and true regret vs training step, one column per magnitude
  - Axes: x: train step (0-400); top row y: proxy J; bottom row y: true regret
  - Suggested caption: "All 3 seeds converge to the same exploit almost identically -- the B3 divergence is deterministic, not noise."
- **`paper_figs/04_b3_sanity_schedules.pdf`** (+ `.png`)
  - Shows: Follow-up A: B3 mean liquidation schedule (x_k/X vs t/T) at training init, true-regret minimum, and end, mean over 3 seeds
  - Axes: x: t/T (0-1); y: mean fraction of initial position remaining, x_k/X
  - Suggested caption: "The trained ("end") policy dumps ~95-100% of the position in the first 5-10% of the horizon, unlike the smooth "init"/"min" schedule."
- **`paper_figs/05_grpo_cosine_vs_G.pdf`** (+ `.png`)
  - Shows: Pilot 2: cosine similarity between each estimator's mean gradient and its exact target, vs group size G
  - Axes: x: G (rollouts/group, log2 scale, 2-256); y: cosine(estimated grad, exact target grad)
  - Suggested caption: "All three estimators converge to cosine~1 as G grows; GRPO's per-group normalization gives it higher cosine at small G."
- **`paper_figs/06_grpo_minus_rloo_regret_vs_sigma_r.pdf`** (+ `.png`)
  - Shows: Per-instance regret(GRPO)-regret(RLOO), predicted (theta_GRPO*-theta_true*) vs realized (5-seed trained, 95% CI), kappa vs phi1
  - Axes: x: sigma_r at theta_true* (per instance); y: regret(GRPO)-regret(RLOO)
  - Suggested caption: "Under phi1 both curves rise with sigma_r and the realized CI excludes 0 for the highest-variance instances; under kappa both stay flat at 0."
- **`paper_figs/07_binary_reward_grpo_minus_rloo.pdf`** (+ `.png`)
  - Shows: Extension 1 base run: per-instance regret(GRPO)-regret(RLOO) under binary reward (phi1), predicted vs realized
  - Axes: x: sqrt(p(1-p)) at theta_true* (binary-reward std, per instance); y: regret(GRPO)-regret(RLOO)
  - Suggested caption: "Both predicted and realized differences collapse near a single point (p saturates near 1 at theta_true*), consistent with the near-zero predicted gap."
- **`paper_figs/08_mlp_vs_linear_grpo_minus_rloo.pdf`** (+ `.png`)
  - Shows: Extension 2 (lr=0.05, 5 seeds): per-instance regret(GRPO)-regret(RLOO) for the MLP policy vs the linear phi1 reference
  - Axes: x: sigma_r (linear phi1's theta_true* reference, per instance); y: regret(GRPO)-regret(RLOO)
  - Suggested caption: "MLP error bars are an order of magnitude wider than linear phi1's, showing training noise (not the bias) dominates at this (later corrected) learning rate."
- **`paper_figs/09_binary_dose_response.pdf`** (+ `.png`)
  - Shows: Extension 1 dose-response: predicted and realized regret(GRPO)-regret(RLOO) vs binary-reward weight spread, 3 delta values
  - Axes: x: weight spread max(1/sqrt(p(1-p)))/min(...) at TWAP init (log scale); y: regret gap
  - Suggested caption: "Predicted gap stays ~0 across the whole (modest, <=3.2x) spread range; realized gap is noisy and destabilizes at the most extreme delta."
- **`paper_figs/10_mlp_stability_20seeds.pdf`** (+ `.png`)
  - Shows: Extension 2 resolution: per-seed regret strip plot (RLOO/Dr.GRPO/GRPO), 20 seeds each, MLP policy at the chosen lr=0.01
  - Axes: x: estimator (categorical); y: regret (per-seed mean over 16 instances); black bars: median and IQR
  - Suggested caption: "At a properly-scaled learning rate, GRPO's regret distribution sits measurably above RLOO/Dr.GRPO's, with fewer outlier seeds."
