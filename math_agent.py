#!/usr/bin/env python3
"""LLMOS on Hendrycks MATH (subset). One process per problem.
Model: ornith:35b @ num_ctx=65536. Multi-step with dev.calc.

    PYTHONPATH=~/Code/LLMOS python3 math_agent.py [N]

Reads ~/math/instances.json (from math_select.py); writes ~/math/results.json.
Scoring: normalize (strip \\boxed{}, LaTeX, %/$, whitespace) and equality.
"""
import json, os, re, sys, time

sys.path.insert(0, os.path.expanduser("~/Code/LLMOS"))
from llmos.store import Store
from llmos.kernel import Kernel
from llmos.cpu import OllamaCPU as _CPU

HOST   = "http://127.0.0.1:11434"       # ollama /api/generate + chatml template
MODEL  = "ornith:35b"
NUMCTX = 65536
BUDGET = 20                             # multi-step: model uses calc + RETURN
INST   = os.path.expanduser("~/math/instances.json")
OUT    = os.path.expanduser("~/math/results.json")
STORE  = os.path.expanduser("~/math/store.db")

os.makedirs(os.path.dirname(STORE), exist_ok=True)


# --- MATH answer normalization (best-effort) ---------------------------
_BOX = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
_FRAC = re.compile(r"\\frac\{([^{}]+)\}\{([^{}]+)\}")
_SQRT = re.compile(r"\\sqrt\{([^{}]+)\}")
_STRIP = re.compile(r"[\s\$,]|\\left|\\right|\\!|\\,|\\;|\\:|\\ ")

def norm(s):
    if s is None:
        return None
    s = str(s)
    # extract inside \boxed{} if present
    m = _BOX.search(s)
    if m:
        s = m.group(1)
    # frac{a}{b} -> a/b, sqrt{a} -> sqrt(a)
    s = _FRAC.sub(lambda m: f"({m.group(1)})/({m.group(2)})", s)
    s = _SQRT.sub(lambda m: f"sqrt({m.group(1)})", s)
    # % and $ and thin spaces and commas
    s = _STRIP.sub("", s)
    # dollar/degree/percent labels
    s = s.replace("^\\circ", "").replace("\\%", "").replace("^{\\circ}", "").replace("\\pi", "pi")
    # try numeric equivalence
    try:
        v = float(s)
        return f"{v:.6g}"
    except Exception:
        pass
    return s.strip().lower()


def prompt_for(inst, strict=False):
    stricture = ""
    if strict:
        stricture = (
            "\n\nIMPORTANT: your previous attempt did not produce a valid answer. "
            "You MUST RETURN the numeric or symbolic answer in the result field "
            "this time, even if you are unsure. Do not leave result empty. "
            "Do not RETURN reasoning text; RETURN the answer value only "
            "(e.g. `26` or `1/2` or `sqrt(34)`)."
        )
    return (
        f"Solve this math problem. Do ALL arithmetic by CALLing the calc syscall "
        f"(pass expressions verbatim, do not compute yourself). When you have the "
        f"final answer, RETURN it as the result — just the answer, no reasoning, "
        f"no LaTeX except if the answer requires it (e.g. a fraction like 3/4 is fine "
        f"as `3/4`; a boxed answer is fine as `42`).\n\n"
        f"Problem: {inst['problem']}"
        + stricture
    )


def run_one(kernel, inst, strict=False):
    goal = prompt_for(inst, strict=strict)
    pid = kernel.spawn(goal, budget=BUDGET)
    t0 = time.time()
    kernel.run()
    pcb = kernel.procs[pid]
    pred = pcb.result
    ok = norm(pred) == norm(inst["answer"])
    calc_calls = sum(1 for s in pcb.context
                     if s.get("op") == "CALL" and (s.get("args") or {}).get("name") == "calc")
    return {
        "id": inst["id"],
        "subject": inst["subject"],
        "level": inst["level"],
        "gold": inst["answer"],
        "gold_norm": norm(inst["answer"]),
        "pred": pred,
        "pred_norm": norm(pred),
        "correct": ok,
        "budget_used": BUDGET - pcb.budget,
        "calc_calls": calc_calls,
        "seconds": round(time.time() - t0, 1),
        "status": pcb.status.value,
    }


def main():
    with open(INST) as f:
        instances = json.load(f)
    if len(sys.argv) > 1:
        instances = instances[: int(sys.argv[1])]

    store = Store(STORE)
    cpu = _CPU(model=MODEL, host=HOST, num_predict=2048, num_ctx=NUMCTX, keep_alive="24h")
    kernel = Kernel(store, cpu, project="math")
    kernel.boot()

    results = []
    correct = 0
    retries = 0
    for i, inst in enumerate(instances, 1):
        r = run_one(kernel, inst, strict=False)
        r["retried"] = False
        # retry-on-none: if the model produced no answer at all (empty RETURN
        # or unparseable prose), bump seed and retry with strict prompt.
        if r["pred"] is None or str(r["pred"]).strip() == "" or "SCHEMA VALIDATION" in str(r["pred"]):
            cpu.seed = (cpu.seed or 0) + 17
            r2 = run_one(kernel, inst, strict=True)
            r2["retried"] = True
            retries += 1
            if r2["pred"] is not None and str(r2["pred"]).strip() and "SCHEMA VALIDATION" not in str(r2["pred"]):
                r = r2
            cpu.seed = 0
        results.append(r)
        correct += int(r["correct"])
        tag = "(retry)" if r.get("retried") else ""
        print(f"[{i:>3}/{len(instances)}] {inst['subject']:<24} L{inst['level']} "
              f"gold={str(r['gold_norm'])[:20]:<20} pred={str(r['pred_norm'])[:20]:<20} "
              f"{'OK' if r['correct'] else '.'} {tag:<8} "
              f"calc={r['calc_calls']} steps={r['budget_used']} {r['seconds']}s", flush=True)
        with open(OUT, "w") as f:
            json.dump({"n": i, "correct": correct, "score": correct / i,
                       "retries": retries, "results": results}, f, indent=1)

    print(f"\nMATH subset ({len(instances)}): {correct}/{len(instances)} = {correct/len(instances):.1%} "
          f"(retries: {retries})")


if __name__ == "__main__":
    main()
