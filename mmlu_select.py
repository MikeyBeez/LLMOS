#!/usr/bin/env python3
"""Pick a stratified MMLU subset (default: 10 subjects x 5 questions = 50).
Reads the cais/mmlu test split via HuggingFace datasets; writes ~/mmlu/instances.json.

Uses ~/swebench-venv/bin/python (has `datasets`)."""
import json, os, random, sys
from datasets import load_dataset

SUBJECTS = [
    "high_school_mathematics", "high_school_physics", "high_school_chemistry",
    "high_school_biology", "high_school_us_history", "philosophy",
    "professional_psychology", "college_computer_science", "formal_logic", "jurisprudence",
]
PER = int(sys.argv[1]) if len(sys.argv) > 1 else 5
OUT = os.path.expanduser("~/mmlu/instances.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

random.seed(20260708)
picked = []
for subj in SUBJECTS:
    ds = load_dataset("cais/mmlu", subj, split="test")
    idxs = random.sample(range(len(ds)), min(PER, len(ds)))
    for i in idxs:
        row = ds[int(i)]
        picked.append({
            "id": f"{subj}-{i}",
            "subject": subj,
            "question": row["question"],
            "choices": row["choices"],           # list of 4 strings
            "answer": "ABCD"[row["answer"]],     # int 0-3 -> letter
        })

with open(OUT, "w") as f:
    json.dump(picked, f, indent=1)
print(f"wrote {len(picked)} MMLU instances across {len(SUBJECTS)} subjects -> {OUT}")
