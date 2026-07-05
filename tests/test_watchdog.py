"""Watchdog test: a runaway process that never RETURNs is TERMINATED when it
exhausts its budget — not re-queued (which would spin forever). The budget is the
hard-cap safety net from the architecture.

Run from the repo root:  PYTHONPATH=. python3 tests/test_watchdog.py
"""
import os
import tempfile

from llmos.store import Store
from llmos.kernel import Kernel
from llmos.cpu import MockCPU
from llmos.isa import Instruction, Op


def spin(pcb):
    return Instruction(Op.PLAN, {"text": "never finishes"})   # a program with no RETURN


def main():
    db = tempfile.mktemp(suffix=".db")
    store = Store(db)
    kernel = Kernel(store, MockCPU({"spin": spin}), log=lambda *a: None)
    kernel.boot()
    pid = kernel.spawn("spin", budget=5)
    kernel.run()                     # must not hang
    pcb = kernel.procs[pid]
    assert pcb.status.value == "KILLED", f"runaway should be KILLED, got {pcb.status.value}"
    assert pcb.pc == 5, f"should run exactly its budget of 5 steps, ran {pcb.pc}"
    store.close()
    if os.path.exists(db):
        os.unlink(db)
    print("ALL WATCHDOG TESTS PASSED")


if __name__ == "__main__":
    main()
