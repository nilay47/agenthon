"""
Deterministically regenerates the 16 fixed instances (same FIXED_SEED=777 used everywhere
in Pilot 2 / Follow-ups B-D) and records their raw sampled parameters (X, sigma, lam, eta)
plus the closed-form kappa -- none of this involves training or simulation, it's the same
deterministic sampler call already used throughout, just written to results/ for citation.
"""
import json

import numpy as np

from env import kappa_batch
from pilot_grpo import get_fixed_batch

batch = get_fixed_batch()
kappa = kappa_batch(batch)

rows = []
for i in range(batch.B):
    rows.append(dict(instance=i, X=float(batch.X[i]), sigma=float(batch.sigma[i]),
                      lam=float(batch.lam[i]), eta=float(batch.eta[i]), kappa=float(kappa[i])))

with open("results/fixed_instances.json", "w") as f:
    json.dump(rows, f, indent=2)

for r in rows:
    print(r)
