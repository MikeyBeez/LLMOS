#!/usr/bin/env python3
"""Pick a stratified MMLU-Pro subset (default: 10 subjects x 5 = 50).
TIGER-Lab/MMLU-Pro test split via HuggingFace. Writes ~/mmlu_pro/instances.json.

Uses ~/swebench-venv/bin/python."""
import json, os, random, sys, collections
from datasets import load_dataset

# 10 categories from MMLU-Pro's 14 (drop 'other' and skew toward diverse domains)
SUBJECTS = [
    "biology", "business", "chemistry", "computer science", "economics",
    "engineering", "health", "history", "law", "math",
]
PER = int(sys.argv[1]) if len(sys.argv) > 1 else 5
OUT = os.path.expanduser("~/mmlu_pro/instances.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

random.seed(20260708)
ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")

by_cat = collections.defaultdict(list)
for i, row in enumerate(ds):
    by_cat[row["category"]].append(i)

picked = []
for subj in SUBJECTS:
    pool = by_cat.get(subj, [])
    if not pool:
        print(f"WARN: no rows for category {subj!r}; available: {sorted(by_cat)}")
        continue
    idxs = random.sample(pool, min(PER, len(pool)))
    for i in idxs:
        row = ds[int(i)]
        picked.append({
            "id": f"{subj.replace(' ','_')}-{row.get('question_id', i)}",
            "subject": subj,
            "question": row["question"],
            "options": row["options"],       # list of up to 10 strings
            "answer": row["answer"],         # letter A-J
            "answer_index": row.get("answer_index"),
        })

with open(OUT, "w") as f:
    json.dump(picked, f, indent=1)
print(f"wrote {len(picked)} MMLU-Pro instances across {len(SUBJECTS)} subjects -> {OUT}")
