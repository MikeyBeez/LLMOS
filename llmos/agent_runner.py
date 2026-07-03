"""LLMOS agent — one real macOS process running on the CPU.

This is the "userland" process. It runs its own CPU (Mock or Ollama) to produce
instructions, but it cannot touch the world: every instruction is sent to the
kernel over a Unix domain socket, the kernel enforces capabilities + executes any
syscall + records the trace, and the result comes back. The model lives out here;
the authority stays in the kernel.

Launched by procd.py as a child process:
    python3 -m llmos.agent_runner --pid P --goal "..." --sock /path --budget N [--ollama --model M]
"""
from __future__ import annotations

import argparse
import json
import os
import socket

from .cpu import MockCPU, OllamaCPU
from .pcb import PCB
from .programs import PROGRAMS


def main():
    ap = argparse.ArgumentParser(prog="llmos-agent")
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--goal", required=True)
    ap.add_argument("--sock", required=True)
    ap.add_argument("--budget", type=int, default=16)
    ap.add_argument("--ollama", action="store_true")
    ap.add_argument("--model", default="qwen2.5:latest")
    a = ap.parse_args()

    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(a.sock)
    rfile = s.makefile("r", encoding="utf-8")

    def send(obj):
        s.sendall((json.dumps(obj) + "\n").encode())

    # announce which real OS process this is (and that the kernel is our parent)
    send({"t": "hello", "pid": a.pid, "ospid": os.getpid(), "ppid": os.getppid()})

    cpu = OllamaCPU(model=a.model) if a.ollama else MockCPU(PROGRAMS)
    pcb = PCB(pid=a.pid, goal=a.goal, budget=a.budget)

    while pcb.budget > 0:
        instr = cpu.step(pcb)                                  # FETCH (in this process)
        send({"t": "step", "op": instr.op.value, "args": instr.args})   # EXECUTE (via kernel)
        resp = json.loads(rfile.readline())
        pcb.context.append({"pc": pcb.pc, "op": instr.op.value, "args": instr.args, "result": resp["result"]})
        pcb.pc += 1
        pcb.budget -= 1
        if resp.get("done"):
            break

    s.close()


if __name__ == "__main__":
    main()
