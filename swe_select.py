#!/usr/bin/env python3
"""Pick N clean, recent sympy instances from SWE-bench Lite and write their specs."""
import json, os, sys
import pandas as pd

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
df = pd.read_parquet(os.path.expanduser("~/swe/lite.parquet"))
sub = df[df.repo == "sympy/sympy"].sort_values("created_at", ascending=False)
out = []
for _, r in sub.iterrows():
    out.append({
        "instance_id": r.instance_id,
        "repo": r.repo,
        "base_commit": r.base_commit,
        "problem_statement": r.problem_statement,
        "test_patch": r.test_patch,
        "gold_patch": r.patch,
        "FAIL_TO_PASS": json.loads(r.FAIL_TO_PASS),
        "PASS_TO_PASS": json.loads(r.PASS_TO_PASS)[:6],
    })
    if len(out) >= N:
        break
json.dump(out, open(os.path.expanduser("~/swe/instances.json"), "w"))
print("wrote", len(out), "instances:", [o["instance_id"] for o in out])
