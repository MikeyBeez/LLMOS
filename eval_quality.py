#!/usr/bin/env python3
"""Quality eval for the LLMOS CPU: does the model pick the RIGHT instructions and
reach the RIGHT result — not just complete the loop? Each goal has a checkable
success criterion. We score correctness, steps (efficiency), faults, and retries
across models. Fresh temp DB per goal so memory checks are unambiguous.

    PYTHONPATH=. python3 -u eval_quality.py
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
from llmos.cpu import OllamaCPU, MockCPU
from llmos.programs import PROGRAMS

TS_RE = re.compile(r"20\d\d-\d\d-\d\dT\d\d:\d\d")


def check_time(store, pcb):
    for k in store.mem_list("mem"):
        if TS_RE.search(str(store.mem_read("mem", k))):
            return True, f"saved timestamp at mem/{k}"
    return False, "no timestamp written to memory"


def check_greeting(store, pcb):
    for k in store.mem_list("mem"):
        v = str(store.mem_read("mem", k)).lower()
        if "hello world" in v:
            return True, f"mem/{k} = {v[:40]!r}"
    return False, "greeting text not found in memory"


def check_number(store, pcb):
    for k in store.mem_list("mem"):
        v = str(store.mem_read("mem", k))
        if "42" in v:
            return True, f"mem/{k} = {v}"
    return False, "42 not found in memory"


def check_read(store, pcb):
    r = str(pcb.result)
    return ("42" in r), f"returned result = {r[:60]!r}"


# (goal, checker, seed_mem_or_None) — seed is (key, value) pre-written to memory
GOALS = [
    ("get the current time and save it to memory", check_time, None),
    ("save the text 'hello world' to memory under the key greeting, then finish", check_greeting, None),
    ("store the number 42 in memory under the key answer", check_number, None),
    ("read the value at memory key answer and return it", check_read, ("answer", 42)),
]

MODELS = [
    ("ornith:35b", "http://127.0.0.1:11435"),
    ("qwen2.5:latest", "http://127.0.0.1:11434"),
    ("llama3.1:8b", "http://127.0.0.1:11434"),
]


def run_goal(model, host, goal, checker, seed):
    db = tempfile.mktemp(suffix=".db")
    store = Store(db)
    if seed:
        store.mem_write("mem", seed[0], seed[1])
    cpu = OllamaCPU(model=model, host=host, log=lambda *a: None)
    k = Kernel(store, cpu, log=lambda *a: None)
    k.boot()
    pid = k.spawn(goal, budget=8)
    t0 = time.time()
    k.run()
    dt = time.time() - t0
    pcb = k.procs[pid]
    ok, note = checker(store, pcb)
    rows = store.metrics_rows()
    faults = sum(r["fault"] or 0 for r in rows)
    retries = sum(r["retries"] or 0 for r in rows)
    ops = [r["op"] for r in rows]
    store.close()
    if os.path.exists(db):
        os.unlink(db)
    return {"ok": ok, "note": note, "steps": pcb.pc, "faults": faults,
            "retries": retries, "dt": dt, "ops": ops, "result": str(pcb.result)[:90]}


def main():
    for model, host in MODELS:
        print(f"\n===== {model} =====", flush=True)
        passed = 0
        tot_steps = tot_faults = tot_retries = 0
        for goal, checker, seed in GOALS:
            r = run_goal(model, host, goal, checker, seed)
            passed += r["ok"]
            tot_steps += r["steps"]
            tot_faults += r["faults"]
            tot_retries += r["retries"]
            print(f"[{'PASS' if r['ok'] else 'FAIL'}] {r['dt']:5.1f}s  steps={r['steps']} "
                  f"faults={r['faults']} retries={r['retries']}  ops={'>'.join(r['ops'])}", flush=True)
            print(f"        goal: {goal!r}", flush=True)
            print(f"        {r['note']}   result={r['result']!r}", flush=True)
        print(f"SCORE {model}: {passed}/{len(GOALS)} correct  "
              f"(steps {tot_steps}, faults {tot_faults}, retries {tot_retries})", flush=True)
    print("\n### EVAL DONE ###", flush=True)


if __name__ == "__main__":
    main()
