#!/usr/bin/env python3
"""Smoke test the LlamaCppCPU: one MMLU + one MATH instance. Prints per-step
timings + whether cache_prompt is hitting.

Prereqs:
  ./run_llamacpp.sh    (server on :8080)
  ~/mmlu/instances.json, ~/math/instances.json  (from *_select.py)

    PYTHONPATH=~/Code/LLMOS python3 smoke_llamacpp.py
"""
import json, os, sys, time

sys.path.insert(0, os.path.expanduser("~/Code/LLMOS"))
from llmos.store import Store
from llmos.kernel import Kernel
from llamacpp_cpu import LlamaCppCPU

HOST   = "http://127.0.0.1:8080"
NUMCTX = 131072
GRAMMAR = os.path.expanduser("~/Code/LLMOS/isa.gbnf")


def run_and_report(kernel, cpu, goal, tag, budget):
    pid = kernel.spawn(goal, budget=budget)
    t0 = time.time()
    kernel.run()
    pcb = kernel.procs[pid]
    dt = time.time() - t0
    print(f"\n[{tag}] pid={pid} status={pcb.status.value} result={pcb.result!r} "
          f"steps={budget - pcb.budget} took={dt:.1f}s")
    for step in pcb.context:
        m = cpu.last_meta   # last step's meta only; per-step meta not stored
        print(f"  pc={step['pc']} {step['op']} args={str(step.get('args'))[:60]}"
              f" -> {str(step.get('result'))[:80]}")
    print(f"  final cpu.last_meta = {cpu.last_meta}")


def main():
    store = Store(os.path.expanduser("~/llamacpp_smoke/store.db"))
    os.makedirs(os.path.dirname(store.path), exist_ok=True)
    cpu = LlamaCppCPU(host=HOST, num_ctx=NUMCTX, grammar_path=GRAMMAR,
                       cache_prompt=True, num_predict=1024)
    kernel = Kernel(store, cpu, project="llamacpp_smoke")
    kernel.boot()

    # 1) MMLU: forces the letter-only RETURN. Grammar guarantees a valid RETURN
    #    with non-null result even if the model wants to just reason in prose.
    mmlu = json.load(open(os.path.expanduser("~/mmlu/instances.json")))[0]
    labels = "ABCD"
    body = "\n".join(f"{labels[i]}. {c}" for i, c in enumerate(mmlu["choices"]))
    goal = (f"Answer the following multiple-choice question. RETURN a single "
            f"letter A, B, C, or D as the result — nothing else in the result "
            f"field.\n\nQuestion: {mmlu['question']}\n\n{body}")
    run_and_report(kernel, cpu, goal, f"MMLU:{mmlu['id']} gold={mmlu['answer']}", budget=6)

    # 2) MATH: exercises calc + RETURN. Grammar guarantees RETURN.result is
    #    present, killing the empty-RETURN failure mode from v1/v2.
    math = json.load(open(os.path.expanduser("~/math/instances.json")))[0]
    goal = (f"Solve this math problem. Do ALL arithmetic by CALLing the calc "
            f"syscall (pass expressions verbatim, do not compute yourself). "
            f"When you have the final answer, RETURN it as the result.\n\n"
            f"Problem: {math['problem']}")
    run_and_report(kernel, cpu, goal, f"MATH:{math['id']} gold={math['answer']}", budget=12)


if __name__ == "__main__":
    main()
