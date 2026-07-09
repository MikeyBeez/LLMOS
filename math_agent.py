#!/usr/bin/env python3
"""LLMOS on Hendrycks MATH (subset), NATIVE tool-calling.

Two tools:
  - calc(expr):     evaluate an arithmetic expression via the calc device
                    (understands factorial, binomial, gcd/lcm, trig, pi/e, mod)
  - finish(answer): commit the final answer

Ornith emits <tool_call> XML that ollama's qwen3 parser surfaces as
OpenAI-style tool_calls. Sampling per ornith model card: T=0.6, top_p=0.95.

    PYTHONPATH=~/Code/LLMOS python3 math_agent.py [N]
"""
import json, os, re, sys, time

sys.path.insert(0, os.path.expanduser("~/Code/LLMOS"))
from llmos.store import Store
from llmos.kernel import Kernel
from tool_call_cpu import ToolCallCPU

HOST   = "http://127.0.0.1:11434"
MODEL  = "ornith:35b"
NUMCTX = 65536
NUMPRED = 4096       # multi-step reasoning; more head-room than MMLU
TEMP   = 0.6
BUDGET = 20
INST   = os.path.expanduser("~/math/instances.json")
OUT    = os.path.expanduser("~/math/results.json")
STORE  = os.path.expanduser("~/math/store.db")

os.makedirs(os.path.dirname(STORE), exist_ok=True)


TOOLS = [
    {"type": "function", "function": {
        "name": "calc",
        "description": ("Evaluate a math expression EXACTLY. Understands +-*/// % ** ^, "
                        "factorial (5! or (3+2)!), binomial C(n,k), permutations P(n,k), "
                        "gcd, lcm, sqrt, sin/cos/tan/arcsin/arccos/arctan (radians), "
                        "exp/log/ln, pi/e/tau, `n mod m`."),
        "parameters": {"type": "object",
                       "properties": {"expr": {"type": "string",
                                                "description": "the expression to evaluate"}},
                       "required": ["expr"]}}},
    {"type": "function", "function": {
        "name": "finish",
        "description": "Commit your final answer. Pass the numeric or symbolic answer as `answer`.",
        "parameters": {"type": "object",
                       "properties": {"answer": {"type": "string",
                                                  "description": "final answer, e.g. `26` or `1/2` or `sqrt(34)`"}},
                       "required": ["answer"]}}},
]
TOOL2SYS = {"calc": "calc", "finish": "RETURN"}

SYSTEM = (
    "You are an expert solving math problems. Use the `calc` tool for every "
    "arithmetic step (pass expressions verbatim; the tool handles factorial, "
    "binomial, trig, gcd/lcm, etc.). When you have the final answer, call "
    "`finish` with it. Do NOT compute arithmetic in your head."
)


# --- MATH answer normalization ------------------------------------------
_BOX  = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
_TEXT = re.compile(r"\\text\{([^{}]*)\}")
_FRAC = re.compile(r"\\d?frac\{([^{}]+)\}\{([^{}]+)\}")
_SQRT = re.compile(r"\\sqrt\{([^{}]+)\}")
_STRIP = re.compile(r"[\s\$,]|\\left|\\right|\\!|\\,|\\;|\\:|\\ ")
_PAREN_ATOM = re.compile(r"\(\s*(-?\d+(?:\.\d+)?(?:/\d+)?)\s*\)")

def norm(s):
    if s is None: return None
    s = str(s)
    m = _BOX.search(s)
    if m: s = m.group(1)
    s = _TEXT.sub(r"\1", s)
    s = _FRAC.sub(lambda m: f"({m.group(1)})/({m.group(2)})", s)
    s = _SQRT.sub(lambda m: f"sqrt({m.group(1)})", s)
    s = _STRIP.sub("", s)
    s = s.replace("^\\circ", "").replace("\\%", "").replace("^{\\circ}", "").replace("\\pi", "pi")
    s = s.replace("\\mathbf{", "").replace("\\mathrm{", "").replace("}", "")
    for _ in range(4):
        s = _PAREN_ATOM.sub(r"\1", s)
    try: return f"{float(s):.6g}"
    except Exception: pass
    return s.strip().lower()


def eq_math(a, b):
    na, nb = norm(a), norm(b)
    if na is None or nb is None: return False
    if na == nb: return True
    try:
        return abs(float(na) - float(nb)) < 1e-6
    except Exception: pass
    try:
        import sympy as sp
        ea = sp.sympify(str(a).replace("^", "**"), evaluate=True)
        eb = sp.sympify(str(b).replace("^", "**"), evaluate=True)
        return sp.simplify(ea - eb) == 0
    except Exception: return False


def prompt_for(inst):
    return f"Problem: {inst['problem']}"


def run_one(kernel, inst):
    goal = prompt_for(inst)
    pid = kernel.spawn(goal, budget=BUDGET)
    t0 = time.time()
    kernel.run()
    pcb = kernel.procs[pid]
    pred = pcb.result
    # If the model called finish, pred is a dict-ish; extract 'answer' key.
    if isinstance(pred, dict):
        pred = pred.get("answer", pred.get("result", pred))
    ok = eq_math(pred, inst["answer"])
    calc_calls = sum(1 for s in pcb.context
                     if s.get("op") == "CALL" and (s.get("args") or {}).get("name") == "calc")
    return {
        "id": inst["id"], "subject": inst["subject"], "level": inst["level"],
        "gold": inst["answer"], "gold_norm": norm(inst["answer"]),
        "pred": pred, "pred_norm": norm(pred), "correct": ok,
        "budget_used": BUDGET - pcb.budget, "calc_calls": calc_calls,
        "seconds": round(time.time() - t0, 1),
        "status": pcb.status.value,
    }


def main():
    with open(INST) as f:
        instances = json.load(f)
    if len(sys.argv) > 1:
        instances = instances[: int(sys.argv[1])]

    store = Store(STORE)
    cpu = ToolCallCPU(tools=TOOLS, tool2sys=TOOL2SYS, system_prompt=SYSTEM,
                     model=MODEL, host=HOST, temperature=TEMP,
                     num_predict=NUMPRED, num_ctx=NUMCTX, keep_alive="24h")
    kernel = Kernel(store, cpu, project="math")
    kernel.boot()

    results, correct = [], 0
    for i, inst in enumerate(instances, 1):
        r = run_one(kernel, inst)
        results.append(r); correct += int(r["correct"])
        print(f"[{i:>3}/{len(instances)}] {inst['subject']:<24} L{inst['level']} "
              f"gold={str(r['gold_norm'])[:20]:<20} pred={str(r['pred_norm'])[:20]:<20} "
              f"{'OK' if r['correct'] else '.'} "
              f"calc={r['calc_calls']} steps={r['budget_used']} {r['seconds']}s",
              flush=True)
        with open(OUT, "w") as f:
            json.dump({"n": i, "correct": correct, "score": correct / i,
                       "results": results}, f, indent=1)
    print(f"\nMATH subset ({len(instances)}): {correct}/{len(instances)} = {correct/len(instances):.1%}")


if __name__ == "__main__":
    main()
