"""Smoke test for the LLMOS v0.1 kernel spine.

Run from the repo root:  PYTHONPATH=. python3 tests/test_hello.py

Covers: a process runs to completion; the memory write lands; the trace records
the exact opcode sequence; replay reconstructs state; and a capability-denied
write faults without persisting (the trust boundary holds).
"""
import os
import tempfile

from llmos.cpu import MockCPU
from llmos.isa import Instruction, Op
from llmos.kernel import Kernel
from llmos.pcb import Status
from llmos.replay import replay
from llmos.store import Store


def hello(pcb):
    if pcb.pc == 0:
        return Instruction(Op.CALL, {"name": "clock.now", "args": {}})
    if pcb.pc == 1:
        return Instruction(Op.WRITE_MEM, {"key": "t", "value": pcb.context[-1]["result"]})
    return Instruction(Op.RETURN, {"result": "ok"})


def denied(pcb):
    if pcb.pc == 0:
        return Instruction(Op.WRITE_MEM, {"key": "x", "value": 1})
    return Instruction(Op.RETURN, {"result": "done"})


def main():
    tmp = tempfile.mktemp(suffix=".db")
    store = Store(tmp)
    k = Kernel(store, MockCPU({"hello": hello, "denied": denied}), log=lambda *a: None)
    k.boot()

    # 1. happy path
    pid = k.spawn("hello")
    k.run()
    assert k.procs[pid].status == Status.DONE, "process should finish"
    ts = store.mem_read("mem", "t")
    assert ts is not None, "timestamp should be written to memory"
    ops = [r["op"] for r in store.trace_read(pid)]
    assert ops == ["CALL", "WRITE_MEM", "RETURN"], f"unexpected trace: {ops}"

    # 2. replay reconstructs state
    rtmp = tempfile.mktemp(suffix=".db")
    ok, n, applied, diffs = replay(store, pid, rtmp)
    if os.path.exists(rtmp):
        os.unlink(rtmp)
    assert ok, f"replay mismatch: {diffs}"

    # 3. the trust boundary: a write without mem.write capability must fault + not persist
    pid2 = k.spawn("denied", capabilities={"dev.clock"})
    k.run()
    tr2 = store.trace_read(pid2)
    write_rows = [r for r in tr2 if r["op"] == "WRITE_MEM"]
    assert write_rows and "error" in (write_rows[0]["result"] or {}), "denied write should fault"
    assert store.mem_read("mem", "x") is None, "denied write must not persist"

    store.close()
    os.unlink(tmp)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
