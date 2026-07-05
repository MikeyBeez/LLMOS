"""Eviction test: once a paged-in item's subtask is done, EVICT drops it from the
context window (RAM) while it stays in the store (disk) and the trace (audit).
This is the swap-out half of the page-fault memory model.

Run from the repo root:  PYTHONPATH=. python3 tests/test_evict.py
"""
import os
import tempfile

from llmos.store import Store
from llmos.kernel import Kernel
from llmos.cpu import MockCPU
from llmos.isa import Instruction, Op

BIG = "PAGED-IN PROTOCOL CONTENT " * 40   # something worth evicting from the window


def program(pcb):
    pc = pcb.pc
    if pc == 0:
        return Instruction(Op.WRITE_MEM, {"key": "protocol", "value": BIG})
    if pc == 1:
        return Instruction(Op.READ_MEM, {"key": "protocol"})   # page it into the window
    if pc == 2:
        return Instruction(Op.EVICT, {"key": "protocol"})      # piece done -> evict it
    return Instruction(Op.RETURN, {"result": "done"})


def main():
    db = tempfile.mktemp(suffix=".db")
    store = Store(db)
    kernel = Kernel(store, MockCPU({"task": program}), log=lambda *a: None)
    kernel.boot()
    pid = kernel.spawn("task", budget=8)
    kernel.run()
    pcb = kernel.procs[pid]

    # the paged-in READ_MEM span is gone from the window (RAM freed)
    read_spans = [c for c in pcb.context if c["op"] == "READ_MEM" and c["args"].get("key") == "protocol"]
    assert not read_spans, "the evicted paged-in span should be gone from the window"
    assert "protocol" not in pcb.working_set, "evicted key should leave the working set"

    # but it is still on disk (eviction frees RAM, not disk)
    assert store.mem_read("mem", "protocol") == BIG, "evicted key must remain in the store"

    # and the full history is still in the trace (audit intact)
    ops = [r["op"] for r in store.trace_read(pid)]
    assert "READ_MEM" in ops and "EVICT" in ops, ops
    assert pcb.result == "done"

    store.close()
    if os.path.exists(db):
        os.unlink(db)
    print("ALL EVICT TESTS PASSED")


if __name__ == "__main__":
    main()
