#!/usr/bin/env python3
"""LLMOS on MMLU-Pro (subset). One process per question. Retry-on-none.
Model: ornith:35b @ num_ctx=65536. 10 choices A-J.

    PYTHONPATH=~/Code/LLMOS python3 mmlu_pro_agent.py [N]

Reads ~/mmlu_pro/instances.json; writes ~/mmlu_pro/results.json.

Retry policy: if the first attempt produces pcb.result=None (empty RETURN with
no letter extractable from prose), spawn a second process with a stricter prompt
and a bumped CPU seed so the sampling path differs.
"""
import json, os, re, sys, time

sys.path.insert(0, os.path.expanduser("~/Code/LLMOS"))
from llmos.store import Store
from llmos.kernel import Kernel
from llamacpp_cpu import LlamaCppCPU

HOST   = "http://127.0.0.1:8080"        # llama-server (run_llamacpp_moe.sh)
MODEL  = "ornith:35b"
NUMCTX = 131072                          # 128K
BUDGET = 8                              # generous: PLAN/calc/RETURN
INST   = os.path.expanduser("~/mmlu_pro/instances.json")
OUT    = os.path.expanduser("~/mmlu_pro/results.json")
STORE  = os.path.expanduser("~/mmlu_pro/store.db")

os.makedirs(os.path.dirname(STORE), exist_ok=True)

LETTERS = "ABCDEFGHIJ"


def prompt_for(inst, strict=False):
    q = inst["question"]
    opts = inst["options"]
    body = "\n".join(f"{LETTERS[i]}. {opts[i]}" for i in range(len(opts)))
    letters = LETTERS[: len(opts)]
    stricture = ""
    if strict:
        stricture = (
            "\n\nIMPORTANT: your previous attempt failed to emit a valid answer. "
            "You MUST end this reply with exactly one JSON object of the form "
            f'{{\"op\":\"RETURN\",\"args\":{{\"result\":\"X\"}}}} where X is one of '
            f'{list(letters)}. No prose after the JSON. No empty args.'
        )
    return (
        f"Answer the following multiple-choice question. RETURN a single letter "
        f"({', '.join(list(letters))}) as the result — nothing else in the result "
        f"field.\n\nQuestion: {q}\n\n{body}"
        + stricture
    )


_LETTER = re.compile(r"\b([A-J])\b")

def extract_letter(result):
    if not result:
        return None
    s = str(result).strip().upper()
    if len(s) == 1 and s in LETTERS:
        return s
    m = _LETTER.search(s)
    return m.group(1) if m else None


def run_one(kernel, inst, strict=False):
    goal = prompt_for(inst, strict=strict)
    pid = kernel.spawn(goal, budget=BUDGET)
    t0 = time.time()
    kernel.run()
    pcb = kernel.procs[pid]
    return {
        "pid": pid,
        "raw": pcb.result,
        "letter": extract_letter(pcb.result),
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
    cpu = LlamaCppCPU(model=MODEL, host=HOST, num_predict=4096, num_ctx=NUMCTX, keep_alive="24h")
    kernel = Kernel(store, cpu, project="mmlu_pro")
    kernel.boot()

    results = []
    correct = 0
    retries = 0
    for i, inst in enumerate(instances, 1):
        r = run_one(kernel, inst, strict=False)
        r["retried"] = False
        if r["letter"] is None:
            # bump seed so the sampling path differs, then retry with strict prompt
            cpu.seed = (cpu.seed or 0) + 17
            r2 = run_one(kernel, inst, strict=True)
            r2["retried"] = True
            retries += 1
            if r2["letter"] is not None:
                r = r2   # keep the successful retry
            else:
                r = {**r, "retry": r2}   # keep original but attach retry trace
            cpu.seed = 0    # reset to canonical
        ok = (r.get("letter") == inst["answer"])
        correct += int(ok)
        results.append({
            "id":       inst["id"],
            "subject":  inst["subject"],
            "gold":     inst["answer"],
            "raw":      r.get("raw"),
            "letter":   r.get("letter"),
            "correct":  ok,
            "retried":  r.get("retried", False),
            "budget":   r.get("budget_used"),
            "seconds":  r.get("seconds"),
        })
        tag = "(retry)" if r.get("retried") else ""
        print(f"[{i:>3}/{len(instances)}] {inst['subject']:<18} gold={inst['answer']:<2} "
              f"pred={str(r.get('letter')):<5} {'OK' if ok else '. '} {tag:<8} {r.get('seconds')}s",
              flush=True)
        with open(OUT, "w") as f:
            json.dump({"n": i, "correct": correct, "score": correct / i,
                       "retries": retries, "results": results}, f, indent=1)

    print(f"\nMMLU-Pro subset ({len(instances)}): {correct}/{len(instances)} = {correct/len(instances):.1%} "
          f"(retries: {retries})")


if __name__ == "__main__":
    main()
