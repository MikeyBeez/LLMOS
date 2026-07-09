#!/usr/bin/env python3
"""Pick a stratified Hendrycks MATH subset (default: 7 subjects x 7 levels-mix = ~49).
Reads EleutherAI/hendrycks_math test split via HuggingFace datasets; writes
~/math/instances.json. Uses ~/swebench-venv/bin/python."""
import json, os, random, sys
from datasets import load_dataset

SUBJECTS = [
    ("algebra",              "algebra"),
    ("counting_and_probability", "counting_and_probability"),
    ("geometry",             "geometry"),
    ("intermediate_algebra", "intermediate_algebra"),
    ("number_theory",        "number_theory"),
    ("prealgebra",           "prealgebra"),
    ("precalculus",          "precalculus"),
]
PER = int(sys.argv[1]) if len(sys.argv) > 1 else 7   # per subject
OUT = os.path.expanduser("~/math/instances.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

random.seed(20260708)
picked = []
for name, config in SUBJECTS:
    ds = load_dataset("EleutherAI/hendrycks_math", config, split="test")
    # stratify roughly across levels 1-5: bucket by level, take proportional sample
    by_level = {}
    for i, row in enumerate(ds):
        by_level.setdefault(row.get("level", "?"), []).append(i)
    ordered = sorted(by_level.keys())
    per_lvl = max(1, PER // len(ordered))
    idxs = []
    for lvl in ordered:
        pool = by_level[lvl]
        idxs.extend(random.sample(pool, min(per_lvl, len(pool))))
    idxs = idxs[:PER]
    for i in idxs:
        row = ds[int(i)]
        picked.append({
            "id": f"{name}-{i}",
            "subject": name,
            "level": row.get("level"),
            "problem": row["problem"],
            "solution": row["solution"],
            "answer": row.get("answer") or row["solution"],   # boxed answer if present
        })

with open(OUT, "w") as f:
    json.dump(picked, f, indent=1)
print(f"wrote {len(picked)} MATH instances across {len(SUBJECTS)} subjects -> {OUT}")
