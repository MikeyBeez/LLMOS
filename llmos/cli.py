"""LLMOS shell — the (conversational-in-spirit) command surface.

    python3 -m llmos.cli run [goal]                    run with the deterministic MockCPU
    python3 -m llmos.cli run "<goal>" --ollama         run with a real local model as the CPU
    python3 -m llmos.cli ps                            list known processes
    python3 -m llmos.cli replay <pid>                  reconstruct a run's state from its trace

The built-in "hello" program proves the spine deterministically:
PLAN -> CALL(clock) -> WRITE_MEM -> YIELD -> RETURN. With --ollama, a real LLM
emits the instructions instead.
"""
from __future__ import annotations

import argparse
import os
import tempfile

from .cpu import MockCPU, OllamaCPU
from .isa import Instruction, Op
from .kernel import Kernel
from .replay import replay
from .store import Store

DB = os.path.expanduser("~/Code/LLMOS/state/llmos.db")


def hello_program(pcb) -> Instruction:
    """The CPU for the 'hello' goal. Note pc==2: the CPU reads the previous step's
    result out of its own context window to build the next instruction — exactly
    how a real model would carry data forward."""
    pc = pcb.pc
    if pc == 0:
        return Instruction(Op.PLAN, {"text": "read the clock, then persist it to memory"})
    if pc == 1:
        return Instruction(Op.CALL, {"name": "clock.now", "args": {}})
    if pc == 2:
        t = pcb.context[-1]["result"]
        return Instruction(Op.WRITE_MEM, {"ns": "mem", "key": "hello.timestamp", "value": t})
    if pc == 3:
        return Instruction(Op.YIELD, {})
    return Instruction(Op.RETURN, {"result": {"saved": "hello.timestamp"}})


PROGRAMS = {"hello": hello_program}


def _kernel(cpu=None):
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    store = Store(DB)
    return Kernel(store, cpu or MockCPU(PROGRAMS)), store


def cmd_run(args):
    cpu = OllamaCPU(model=args.model) if args.ollama else None
    kernel, store = _kernel(cpu)
    if args.ollama:
        print(f"[cpu] OllamaCPU model={args.model}")
    kernel.boot()
    pid = kernel.spawn(args.goal, budget=args.budget)
    kernel.run()
    pcb = kernel.procs[pid]
    print(f"\n[done] pid={pid} status={pcb.status.value} result={pcb.result}")
    keys = store.mem_list("mem")
    if keys:
        print("[mem] contents:")
        for k in keys:
            print(f"       mem/{k} = {store.mem_read('mem', k)}")
    print(f"[hint] python3 -m llmos.cli replay {pid}")
    store.close()


def cmd_ps(args):
    _, store = _kernel()
    procs = store.list_processes()
    if not procs:
        print("(no processes yet — try: python3 -m llmos.cli run hello)")
    for p in procs:
        print(f"pid={p['pid']:>3} status={p['status']:<8} pc={p['pc']:<3} goal={p['goal']!r}")
    store.close()


def cmd_replay(args):
    _, store = _kernel()
    tmp = tempfile.mktemp(suffix=".db")
    ok, n, applied, diffs = replay(store, args.pid, tmp)
    if os.path.exists(tmp):
        os.unlink(tmp)
    print(f"[replay] pid={args.pid}: {n} instructions, {len(applied)} memory write(s) re-applied")
    print(f"[replay] state reproduced from trace: {'OK' if ok else 'MISMATCH ' + str(diffs)}")
    store.close()


def main():
    ap = argparse.ArgumentParser(prog="llmos")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run a goal")
    r.add_argument("goal", nargs="?", default="hello")
    r.add_argument("--ollama", action="store_true", help="use a real local model as the CPU")
    r.add_argument("--model", default="qwen2.5:latest", help="ollama model tag")
    r.add_argument("--budget", type=int, default=32, help="max instruction cycles")
    r.set_defaults(fn=cmd_run)
    p = sub.add_parser("ps", help="list processes")
    p.set_defaults(fn=cmd_ps)
    rp = sub.add_parser("replay", help="reconstruct a run from its trace")
    rp.add_argument("pid", type=int)
    rp.set_defaults(fn=cmd_replay)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
