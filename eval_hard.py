#!/usr/bin/env python3
"""HARD multi-step quality benchmark for the LLMOS CPU (ornith:35b).

Each goal forces real planning: several dependent steps, memory used as scratch,
and arithmetic / logic / string reasoning done in the model's head one instruction
at a time — with a deterministic, checkable final answer. This measures what the
model AND the operating system can actually do, not just whether the loop closes.

    PYTHONPATH=. python3 -u eval_hard.py
"""
from __future__ import annotations

import os
import re
import sys
import tempfile
import time

sys.path.insert(0, os.path.expanduser("~/Code/LLMOS"))
from llmos.store import Store
from llmos.kernel import Kernel
from llmos.cpu import OllamaCPU

MODEL = "ornith:35b"
HOST = "http://127.0.0.1:11435"
NUM_PREDICT = 1024      # give the reasoning model room to think each step
BUDGET = 16
TS_RE = re.compile(r"(20\d\d)-(\d\d)-(\d\d)T(\d\d):(\d\d)")


def _num(v):
    try:
        return float(str(v).strip().strip("'\""))
    except Exception:
        return None


def _mem(store):
    return {k: store.mem_read("mem", k) for k in store.mem_list("mem")}


def c_product(store, pcb):
    m = _mem(store)
    hit = any(_num(v) == 42 for v in m.values())
    return hit, f"mem={m}"


def c_prime(store, pcb):
    m = _mem(store)
    v = str(m.get("verdict", "")).lower()
    return ("prime" in v and "composite" not in v), f"verdict={m.get('verdict')!r} mem={m}"


def c_squares(store, pcb):
    m = _mem(store)
    want = {"sq1": 1, "sq2": 4, "sq3": 9, "sq4": 16}
    ok = all(_num(m.get(k)) == want[k] for k in want)
    return ok, f"got={{k:m.get(k) for k in want}} -> {{ {', '.join(k+'='+str(m.get(k)) for k in want)} }}"


def c_parity(store, pcb):
    m = _mem(store)
    ts = None
    for v in m.values():
        mt = TS_RE.search(str(v))
        if mt:
            ts = mt
            break
    if not ts:
        return False, f"no timestamp stored; mem={m}"
    hour = int(ts.group(4))
    expected = "even" if hour % 2 == 0 else "odd"
    got = str(m.get("parity", "")).lower().strip()
    return (got == expected), f"hour={hour} expected={expected} got={got!r} mem={m}"


def c_reverse(store, pcb):
    m = _mem(store)
    rev = str(m.get("reversed", "")).lower().strip().strip("'\"")
    length_ok = _num(m.get("length")) == 5
    return (rev == "somll" and length_ok), f"reversed={m.get('reversed')!r} length={m.get('length')!r}"


def c_seq(store, pcb):
    m = _mem(store)
    return (_num(m.get("step2")) == 21), f"step2={m.get('step2')!r} mem={m}"


GOALS = [
    ("Store 6 under memory key a and 7 under key b. Then compute a times b and "
     "store the product under key product. Finally return the product.", c_product),

    ("Store the number 17 under key n. Decide whether n is a prime number. Store the "
     "word 'prime' under key verdict if it is prime, otherwise store 'composite'. "
     "Return the verdict.", c_prime),

    ("Store the square of 1 under key sq1, the square of 2 under key sq2, the square "
     "of 3 under key sq3, and the square of 4 under key sq4. Then return all four values.", c_squares),

    ("Get the current time and store it under key t. Look at the hour of that time, and "
     "store 'even' or 'odd' under key parity depending on whether the hour number is even "
     "or odd. Return the parity.", c_parity),

    ("Store the word 'llmos' under key word. Reverse the letters of the word and store the "
     "reversed string under key reversed. Also store the number of letters under key length. "
     "Return the reversed word and its length.", c_reverse),

    ("Store 100 under key start. Subtract 58 from it and store the result under key step1. "
     "Then halve step1 and store the result under key step2. Return step2.", c_seq),
]


def run_goal(goal, checker):
    db = tempfile.mktemp(suffix=".db")
    store = Store(db)
    cpu = OllamaCPU(model=MODEL, host=HOST, num_predict=NUM_PREDICT, log=lambda *a: None)
    k = Kernel(store, cpu, log=lambda *a: None)
    k.boot()
    pid = k.spawn(goal, budget=BUDGET)
    t0 = time.time()
    k.run()
    dt = time.time() - t0
    pcb = k.procs[pid]
    ok, note = checker(store, pcb)
    rows = store.metrics_rows()
    faults = sum(r["fault"] or 0 for r in rows)
    retries = sum(r["retries"] or 0 for r in rows)
    ops = ">".join(r["op"] for r in rows)
    store.close()
    if os.path.exists(db):
        os.unlink(db)
    return {"ok": ok, "note": note, "steps": pcb.pc, "faults": faults,
            "retries": retries, "dt": dt, "ops": ops, "result": str(pcb.result)[:120]}


def main():
    print(f"HARD EVAL  model={MODEL}  num_predict={NUM_PREDICT}  budget={BUDGET}", flush=True)
    passed = 0
    tot_steps = tot_faults = tot_retries = 0.0
    t_all = time.time()
    for i, (goal, checker) in enumerate(GOALS, 1):
        r = run_goal(goal, checker)
        passed += r["ok"]
        tot_steps += r["steps"]
        tot_faults += r["faults"]
        tot_retries += r["retries"]
        print(f"\n[{i}] {'PASS' if r['ok'] else 'FAIL'}  {r['dt']:5.1f}s  steps={r['steps']} "
              f"faults={r['faults']} retries={r['retries']}", flush=True)
        print(f"    goal: {goal[:90]}...", flush=True)
        print(f"    ops : {r['ops']}", flush=True)
        print(f"    chk : {r['note']}", flush=True)
        print(f"    ret : {r['result']!r}", flush=True)
    print(f"\n===== SCORE {MODEL}: {passed}/{len(GOALS)} correct  "
          f"(steps {int(tot_steps)}, faults {int(tot_faults)}, retries {int(tot_retries)}, "
          f"{time.time() - t_all:.0f}s total) =====", flush=True)
    print("### HARD EVAL DONE ###", flush=True)


if __name__ == "__main__":
    main()
