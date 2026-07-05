"""Goal-contract tests: the kernel refuses a RETURN while a required step is
unfinished, and lets it through once the step is done. Deterministic (MockCPU).

Run from the repo root:  PYTHONPATH=. python3 tests/test_contract.py
"""
import os
import tempfile

from llmos.store import Store
from llmos.kernel import Kernel
from llmos.cpu import MockCPU
from llmos.isa import Instruction, Op


def lazy_program(pcb):
    """A CPU that tries to RETURN before writing the required key 'foo'. After the
    kernel traps that premature RETURN, it notices the trap in its window, writes
    foo, and returns properly — exactly the recovery a real model should do."""
    wrote = any(c["op"] == "WRITE_MEM" for c in pcb.context)
    trapped = any(isinstance(c["result"], dict) and c["result"].get("trap") for c in pcb.context)
    if wrote:
        return Instruction(Op.RETURN, {"result": "ok"})
    if trapped:
        return Instruction(Op.WRITE_MEM, {"key": "foo", "value": "done"})
    return Instruction(Op.RETURN, {"result": "premature"})   # skips the required step


def main():
    goal = "save something under key foo"
    db = tempfile.mktemp(suffix=".db")
    store = Store(db)
    kernel = Kernel(store, MockCPU({goal: lazy_program}), log=lambda *a: None)
    kernel.boot()
    pid = kernel.spawn(goal, budget=8)

    # the contract was derived from the goal
    assert kernel.procs[pid].contract == {"required_keys": ["foo"]}, kernel.procs[pid].contract

    kernel.run()
    pcb = kernel.procs[pid]

    # the premature RETURN was trapped at least once
    traps = [c for c in pcb.context if isinstance(c["result"], dict) and c["result"].get("trap")]
    assert traps, "expected the premature RETURN to be trapped by the kernel"

    # the required key now exists and the process finished cleanly with the real result
    assert "foo" in store.mem_list("mem"), "required key 'foo' must exist before RETURN"
    assert pcb.status.value == "DONE"
    assert pcb.result == "ok", f"expected the post-recovery result, got {pcb.result!r}"

    # a goal that names no key gets no contract and returns immediately
    g2 = "just say hi"
    assert Kernel._derive_contract(g2) == {}
    kernel.cpu.programs[g2] = lambda pcb: Instruction(Op.RETURN, {"result": "hi"})
    pid2 = kernel.spawn(g2, budget=4)
    kernel.run()
    assert kernel.procs[pid2].result == "hi"

    store.close()
    if os.path.exists(db):
        os.unlink(db)
    print("ALL CONTRACT TESTS PASSED")


if __name__ == "__main__":
    main()
