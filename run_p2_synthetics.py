"""Phase-2 synthetics: extend Phase-1 results with hymba-with-nsa family.

Adds MQAR/Dyck-2/parity numbers for the 4 new variants introduced for the
v5 architecture pivot (P2 screen winner + Aquino-Michaels gate ablation +
ASA alternation). Re-runs the 4 P1 variants we want to compare against
under identical conditions.

Reads existing phase1_synthetics.json and appends; writes
phase2_synthetics_2026-05-12.json.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from train_synthetic import TASK_CFG, train_one

OUT = "/opt/docker/LLM/HSPMN/results/phase2_synthetics_2026-05-12.json"
VARIANTS_NEW = [
    "hymba-with-nsa",
    "hymba-with-nsa-gated",
    "hymba-with-nsa-randgate",
    "hymba-with-asa",
]
TASKS = list(TASK_CFG.keys())  # mqar, dyck2, parity

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
results = []
for task in TASKS:
    for variant in VARIANTS_NEW:
        tag = f"{task} :: {variant}"
        print(f"\n=== {tag} ===", flush=True)
        t0 = time.time()
        try:
            res = train_one(task, variant, steps=2000, lr=3e-4, device=device,
                            seed=42)
            res["wall_s"] = time.time() - t0
            print(f"  → {res}", flush=True)
            results.append(res)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}", flush=True)
            results.append({"task": task, "variant": variant,
                            "error": f"{type(e).__name__}: {e}",
                            "wall_s": time.time() - t0})
        # Incremental save so a crash mid-sweep does not lose results.
        with open(OUT, "w") as f:
            json.dump(results, f, indent=2)

print(f"\nWrote {OUT}")
