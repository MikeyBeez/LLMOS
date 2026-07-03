"""Ask-channel tests: a sandboxed process REQUESTs a capability; a pluggable
Authority grants or denies; and a tainted process cannot regain a privileged cap.

Run from the repo root:  PYTHONPATH=. python3 tests/test_authority.py
"""
import tempfile

from llmos.authority import DenyAuthority, PolicyAuthority
from llmos.cpu import MockCPU
from llmos.isa import Instruction, Op
from llmos.kernel import Kernel
from llmos.store import Store

SANDBOX = {"dev.clock", "mem.read", "fs.read"}   # deliberately no mem.write


def wants_write(pcb):
    if pcb.pc == 0:
        return Instruction(Op.REQUEST, {"capability": "mem.write", "reason": "persist result"})
    if pcb.pc == 1:
        if not (pcb.context[-1]["result"] or {}).get("granted"):
            return Instruction(Op.RETURN, {"result": "blocked"})
        return Instruction(Op.WRITE_MEM, {"key": "k", "value": "v"})
    return Instruction(Op.RETURN, {"result": "done"})


def run_case(authority):
    store = Store(tempfile.mktemp(suffix=".db"))
    k = Kernel(store, MockCPU({"w": wants_write}), log=lambda *a: None, authority=authority)
    k.boot()
    k.spawn("w", capabilities=set(SANDBOX))
    k.run()
    val = store.mem_read("mem", "k")
    store.close()
    return val


def main():
    # granted -> the write goes through
    assert run_case(PolicyAuthority(grant={"mem.write"})) == "v", "granted request should let the write through"
    # denied -> the write never happens
    assert run_case(DenyAuthority()) is None, "denied request must block the write"
    assert run_case(PolicyAuthority(deny={"mem.write"})) is None, "explicit deny must block"

    # a tainted process must not regain a privileged cap, even if the authority would grant it
    store = Store(tempfile.mktemp(suffix=".db"))
    k = Kernel(store, MockCPU({"t": wants_write}), log=lambda *a: None,
               authority=PolicyAuthority(grant={"mem.write"}))
    k.boot()
    pid = k.spawn("t", capabilities=set(SANDBOX))
    k.procs[pid].tainted = True                       # simulate a prior untrusted ingest
    k.run()
    req_row = k.store.trace_read(pid)[0]
    assert "denied" in (req_row["result"] or {}), "tainted process must be auto-denied privileged caps"
    assert store.mem_read("mem", "k") is None, "tainted process must not write"
    store.close()

    print("ALL AUTHORITY TESTS PASSED")


if __name__ == "__main__":
    main()
