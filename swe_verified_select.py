#!/usr/bin/env python3
"""Load SWE-bench Verified (full 500 human-verified instances) and write
per-instance specs to disk. Mirrors swe_select.py's output format so
swe_agent.py runs against Verified unchanged.

    ~/swebench-venv/bin/python swe_verified_select.py [N]

If N is given, take a stratified sample across repos; else write all 500.
"""
import json, os, sys, collections
from datasets import load_dataset

N = int(sys.argv[1]) if len(sys.argv) > 1 else 500
OUT = os.path.expanduser("~/swe/verified_instances.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
print(f"loaded SWE-bench Verified: {len(ds)} instances")

# Stratify by repo so a subset stays diverse
by_repo = collections.defaultdict(list)
for i, row in enumerate(ds):
    by_repo[row["repo"]].append(i)

if N >= len(ds):
    idxs = list(range(len(ds)))
else:
    # proportional-per-repo sampling
    idxs = []
    per_repo = max(1, N // len(by_repo))
    for repo, pool in by_repo.items():
        idxs.extend(pool[:per_repo])
    idxs = idxs[:N]

out = []
for i in idxs:
    row = ds[int(i)]
    out.append({
        "instance_id":       row["instance_id"],
        "repo":              row["repo"],
        "base_commit":       row["base_commit"],
        "problem_statement": row["problem_statement"],
        "test_patch":        row["test_patch"],
        "gold_patch":        row["patch"],
        "FAIL_TO_PASS":      json.loads(row["FAIL_TO_PASS"]),
        "PASS_TO_PASS":      json.loads(row["PASS_TO_PASS"])[:6],
    })

json.dump(out, open(OUT, "w"))
print(f"wrote {len(out)} verified instances -> {OUT}")
print(f"repo distribution: {dict(collections.Counter(o['repo'] for o in out))}")
