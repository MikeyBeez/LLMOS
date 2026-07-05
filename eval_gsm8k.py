#!/usr/bin/env python3
"""GSM8K on LLMOS. Runs real grade-school-math word problems two ways: the model
directly (chain-of-thought baseline) and the same model driven through the LLMOS
instruction loop. Exact-match on the final number, the standard GSM8K metric.

    PYTHONPATH=. python3 -u eval_gsm8k.py [N]
"""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import time
import urllib.request

sys.path.insert(0, os.path.expanduser("~/Code/LLMOS"))
from llmos.store import Store
from llmos.kernel import Kernel
from llmos.cpu import OllamaCPU

MODEL, HOST = "ornith:35b", "http://127.0.0.1:11435"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
SAMPLE = os.path.expanduser("~/Code/LLMOS/gsm8k_sample.jsonl")


def extract_num(text):
    nums = re.findall(r"-?\d+\.?\d*", str(text).replace(",", ""))
    return nums[-1] if nums else None


def num_eq(a, b):
    try:
        return abs(float(a) - float(b)) < 1e-6
    except Exception:
        return False


def direct(q):
    prompt = q + "\n\nSolve step by step, then end with 'The answer is <number>.'"
    body = json.dumps({"model": MODEL, "prompt": prompt, "stream": False, "keep_alive": "30m",
                       "options": {"temperature": 0, "seed": 0, "num_predict": 1024}}).encode()
    req = urllib.request.Request(HOST + "/api/generate", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read()).get("response", "")


def llmos(q):
    db = tempfile.mktemp(suffix=".db")
    store = Store(db)
    cpu = OllamaCPU(model=MODEL, host=HOST, num_predict=1024, log=lambda *a: None)
    k = Kernel(store, cpu, log=lambda *a: None)
    k.boot()
    goal = q + " Work out the answer, then RETURN only the final number."
    pid = k.spawn(goal, budget=8)
    k.run()
    res = str(k.procs[pid].result)
    store.close()
    if os.path.exists(db):
        os.unlink(db)
    return res


def main():
    rows = [json.loads(l) for l in open(SAMPLE) if l.strip()][:N]
    d_ok = l_ok = 0
    print(f"GSM8K  model={MODEL}  N={len(rows)}", flush=True)
    t0 = time.time()
    for i, r in enumerate(rows, 1):
        q, gold = r["question"], r["gold"]
        dp = extract_num(direct(q)); do = num_eq(dp, gold); d_ok += do
        lp = extract_num(llmos(q)); lo = num_eq(lp, gold); l_ok += lo
        print(f"[{i:>2}] gold={gold:>7}  direct={str(dp):>8} {'OK' if do else 'x '}   "
              f"llmos={str(lp):>8} {'OK' if lo else 'x '}", flush=True)
    n = len(rows)
    print(f"\n=== GSM8K, {n} problems, {MODEL} ===", flush=True)
    print(f"direct ornith   : {d_ok}/{n} = {100 * d_ok / n:.0f}%", flush=True)
    print(f"ornith on LLMOS : {l_ok}/{n} = {100 * l_ok / n:.0f}%", flush=True)
    print(f"time: {time.time() - t0:.0f}s", flush=True)
    print("### GSM8K DONE ###", flush=True)


if __name__ == "__main__":
    main()
