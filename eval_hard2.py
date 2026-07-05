#!/usr/bin/env python3
"""HARDER multi-step benchmark for the LLMOS CPU (ornith:35b) — deeper reasoning,
longer chains, conditionals, and computations that are genuinely hard to do one
instruction at a time in your head. Deterministic checks. Runs through the full
stack (contracts, topic routing, metrics).

    PYTHONPATH=. python3 -u eval_hard2.py
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

MODEL, HOST, NUM_PREDICT, BUDGET = "ornith:35b", "http://127.0.0.1:11435", 1024, 20
TS_RE = re.compile(r"T(\d\d):\d\d")


def _num(v):
    try:
        return float(str(v).strip().strip("'\""))
    except Exception:
        return None


def _mem(store):
    return {k: store.mem_read("mem", k) for k in store.mem_list("mem")}


def c_triangle(store, pcb):
    m = _mem(store)
    v = str(m.get("verdict", "")).lower()
    ok = _num(m.get("sumsq")) == 25 and "right" in v and "not" not in v
    return ok, f"sumsq={m.get('sumsq')} verdict={m.get('verdict')!r}"


def c_fib(store, pcb):
    m = _mem(store)
    want = [1, 1, 2, 3, 5, 8]
    got = [_num(m.get(f"f{i}")) for i in range(1, 7)]
    return (got == want), f"f1..f6={got}"


def c_letters(store, pcb):
    m = _mem(store)
    ok = _num(m.get("letters")) == 15 and str(m.get("first", "")).lower().strip("'\"") == "o" \
        and str(m.get("last", "")).lower().strip("'\"") == "m"
    return ok, f"letters={m.get('letters')} first={m.get('first')!r} last={m.get('last')!r}"


def c_change(store, pcb):
    m = _mem(store)
    ok = _num(m.get("price")) == 3 and _num(m.get("total")) == 21 and _num(m.get("change")) == 29
    return ok, f"price={m.get('price')} total={m.get('total')} change={m.get('change')}"


def c_binary(store, pcb):
    m = _mem(store)
    ok = "101101" in str(m.get("binary", "")) and _num(m.get("ones")) == 4
    return ok, f"binary={m.get('binary')!r} ones={m.get('ones')}"


def c_time(store, pcb):
    m = _mem(store)
    hour = _num(m.get("hour"))
    if hour is None:
        return False, f"no hour stored; mem={list(m)}"
    hour = int(hour)
    exp_ampm = "am" if hour < 12 else "pm"
    exp_next = (hour + 1) % 24
    got_ampm = str(m.get("ampm", "")).lower().strip("'\"")
    ok = (got_ampm == exp_ampm) and (_num(m.get("next_hour")) == exp_next)
    return ok, f"hour={hour} ampm={m.get('ampm')!r}(exp {exp_ampm}) next_hour={m.get('next_hour')}(exp {exp_next})"


GOALS = [
    ("Store 3 under key a, 4 under key b, and 5 under key c. Compute a*a + b*b and store it under "
     "key sumsq. If sumsq equals c*c, store 'right triangle' under key verdict, otherwise store "
     "'not a right triangle'. Return the verdict.", c_triangle),

    ("Compute the first six Fibonacci numbers starting 1, 1 and store them under keys f1, f2, f3, "
     "f4, f5, f6 in order. Return the value of f6.", c_fib),

    ("Take the phrase 'operating system'. Ignoring the space, count the letters and store the count "
     "under key letters. Store the first letter under key first and the last letter under key last. "
     "Return all three.", c_letters),

    ("Apples cost 3 dollars each; store that under key price. A customer buys 7 apples: compute the "
     "total cost and store it under key total. They pay with a 50 dollar bill: compute the change "
     "and store it under key change. Return the change.", c_change),

    ("Convert the decimal number 45 to binary and store the binary digits as a string under key "
     "binary. Count how many 1 bits it has and store that under key ones. Return both.", c_binary),

    ("Get the current time and store the hour (0-23) under key hour. If the hour is less than 12 "
     "store 'AM' under key ampm, otherwise store 'PM'. Store what the hour will be one hour later "
     "(wrapping 23 back to 0) under key next_hour. Return ampm and next_hour.", c_time),
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
            "retries": retries, "dt": dt, "ops": ops, "result": str(pcb.result)[:110]}


def main():
    print(f"HARD EVAL 2  model={MODEL}  num_predict={NUM_PREDICT}  budget={BUDGET}", flush=True)
    passed = 0
    t_all = time.time()
    for i, (goal, checker) in enumerate(GOALS, 1):
        r = run_goal(goal, checker)
        passed += r["ok"]
        print(f"\n[{i}] {'PASS' if r['ok'] else 'FAIL'}  {r['dt']:5.1f}s  steps={r['steps']} "
              f"faults={r['faults']} retries={r['retries']}", flush=True)
        print(f"    goal: {goal[:88]}...", flush=True)
        print(f"    ops : {r['ops']}", flush=True)
        print(f"    chk : {r['note']}", flush=True)
        print(f"    ret : {r['result']!r}", flush=True)
    print(f"\n===== SCORE {MODEL}: {passed}/{len(GOALS)}  ({time.time() - t_all:.0f}s total) =====", flush=True)
    print("### HARD EVAL 2 DONE ###", flush=True)


if __name__ == "__main__":
    main()
