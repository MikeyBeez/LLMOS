#!/usr/bin/env python3
"""Pick N SWE-bench Lite instances, stratified across repos so a small pilot
isn't monoculture (previously we ran only sympy which masked env-setup issues
that only surfaced on astropy / django / matplotlib).

    ~/swebench-venv/bin/python swe_lite_select.py [N]
"""
import json, os, sys, collections
import pandas as pd

N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
df = pd.read_parquet(os.path.expanduser("~/swe/lite.parquet"))
print(f"loaded SWE-bench Lite: {len(df)} instances across {df.repo.nunique()} repos")

# Stratify: newest first per repo, take a proportional slice.
per_repo = max(1, N // df.repo.nunique())
out = []
for repo, grp in df.groupby("repo"):
    grp = grp.sort_values("created_at", ascending=False)
    for _, r in grp.head(per_repo).iterrows():
        out.append({
            "instance_id":       r.instance_id,
            "repo":              r.repo,
            "base_commit":       r.base_commit,
            "problem_statement": r.problem_statement,
            "test_patch":        r.test_patch,
            "gold_patch":        r.patch,
            "FAIL_TO_PASS":      json.loads(r.FAIL_TO_PASS),
            "PASS_TO_PASS":      json.loads(r.PASS_TO_PASS)[:6],
        })
        if len(out) >= N:
            break
    if len(out) >= N:
        break

json.dump(out, open(os.path.expanduser("~/swe/instances.json"), "w"))
print(f"wrote {len(out)} instances -> ~/swe/instances.json")
by_repo = collections.Counter(o["repo"] for o in out)
print(f"repo distribution: {dict(by_repo)}")
