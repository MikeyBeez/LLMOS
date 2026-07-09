#!/usr/bin/env python3
"""LLMOS on MMLU (subset). One process per question.
Model: ornith:35b @ num_ctx=65536. Single-turn RETURN(letter).

    PYTHONPATH=~/Code/LLMOS python3 mmlu_agent.py [N]

Reads ~/mmlu/instances.json (from mmlu_select.py); writes ~/mmlu/results.json.
"""
import json, os, re, sys, time

sys.path.insert(0, os.path.expanduser("~/Code/LLMOS"))
from llmos.store import Store
from llmos.kernel import Kernel
from llmos.cpu import OllamaCPU as _CPU

HOST   = "http://127.0.0.1:11434"       # ollama /api/generate (applies chatml
                                        # template, which our earlier probes
                                        # showed is essential for ornith to
                                        # close reasoning cleanly)
MODEL  = "ornith:35b"
NUMCTX = 65536                          # ollama at 128K needs its own tuning; 64K matches v3
BUDGET = 6                              # generous: model may PLAN once then RETURN
INST   = os.path.expanduser("~/mmlu/instances.json")
OUT    = os.path.expanduser("~/mmlu/results.json")
STORE  = os.path.expanduser("~/mmlu/store.db")

os.makedirs(os.path.dirname(STORE), exist_ok=True)

def prompt_for(inst):
    q = inst["question"]
    labels = "ABCD"
    body = "\n".join(f"{labels[i]}. {c}" for i, c in enumerate(inst["choices"]))
    return (
        f"Answer the following multiple-choice question. RETURN a single letter "
        f"A, B, C, or D as the result — nothing else in the result field.\n\n"
        f"Question: {q}\n\n{body}"
    )

_LETTER = re.compile(r"\b([ABCD])\b")

def extract_letter(result):
    if not result:
        return None
    s = str(result).strip().upper()
    if s in "ABCD":
        return s
    m = _LETTER.search(s)
    return m.group(1) if m else None


def run_one(kernel, inst):
    goal = prompt_for(inst)
    pid = kernel.spawn(goal, budget=BUDGET)
    t0 = time.time()
    kernel.run()
    pcb = kernel.procs[pid]
    letter = extract_letter(pcb.result)
    return {
        "id": inst["id"],
        "subject": inst["subject"],
        "gold": inst["answer"],
        "raw": pcb.result,
        "letter": letter,
        "correct": letter == inst["answer"],
        "budget_used": BUDGET - pcb.budget,
        "seconds": round(time.time() - t0, 1),
        "status": pcb.status.value,
    }


def main():
    with open(INST) as f:
        instances = json.load(f)
    if len(sys.argv) > 1:
        instances = instances[: int(sys.argv[1])]

    store = Store(STORE)
    cpu = _CPU(model=MODEL, host=HOST, num_predict=1024, num_ctx=NUMCTX, keep_alive="24h")
    kernel = Kernel(store, cpu, project="mmlu")
    kernel.boot()

    results = []
    correct = 0
    for i, inst in enumerate(instances, 1):
        r = run_one(kernel, inst)
        results.append(r)
        correct += int(r["correct"])
        print(f"[{i:>3}/{len(instances)}] {inst['subject']:<28} gold={r['gold']} pred={r['letter']} "
              f"{'OK' if r['correct'] else '.'}  {r['seconds']}s", flush=True)
        with open(OUT, "w") as f:
            json.dump({"n": i, "correct": correct, "score": correct / i, "results": results}, f, indent=1)

    print(f"\nMMLU subset ({len(instances)}): {correct}/{len(instances)} = {correct/len(instances):.1%}")


if __name__ == "__main__":
    main()
