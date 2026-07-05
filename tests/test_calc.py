"""Calculator-device tests. The point of the device is to take arithmetic away from
the stochastic CPU and give it to deterministic code — including order of operations.
No separate PEMDAS protocol is needed: the AST evaluator respects precedence by
construction (Python's own grammar), so 2+3*4 is 14, not 20.

Run from the repo root:  PYTHONPATH=. python3 tests/test_calc.py
"""
import os
import tempfile

from llmos.store import Store
from llmos.kernel import Kernel
from llmos.cpu import MockCPU
from llmos.syscall import SyscallTable
from llmos.isa import Instruction, Op


class FakePCB:
    def __init__(self):
        self.capabilities = {"dev.calc"}
        self.pid = 1


def main():
    db = tempfile.mktemp(suffix=".db")
    store = Store(db)
    st = SyscallTable(store)

    # order of operations (PEMDAS) is handled by the evaluator, not the model
    cases = {
        "2+3*4": 14,            # multiply before add
        "(2+3)*4": 20,          # parentheses first
        "2**3+1": 9,            # exponent before add
        "10-2-3": 5,            # left-to-right subtraction
        "2+3*4-1": 13,
        "100/4/5": 5,
        "34800/240": 145,       # the exact division the model got wrong (145, not 70)
        "(6*6000-1200)/(20*12)": 145,   # all of problem 8 in one expression
    }
    for expr, exp in cases.items():
        r = st.dispatch(FakePCB(), "calc", {"expr": expr})
        assert r.get("value") == exp, f"{expr} -> {r} (expected {exp})"

    # the evaluator is safe: no names, calls, or imports
    r = st.dispatch(FakePCB(), "calc", {"expr": "__import__('os').system('echo hi')"})
    assert "error" in r, r

    # and it works through the kernel: a program that offloads the division to calc
    def prog(pcb):
        if pcb.pc == 0:
            return Instruction(Op.CALL, {"name": "calc", "args": {"expr": "34800/240"}})
        return Instruction(Op.RETURN, {"result": pcb.context[-1]["result"]["value"]})

    kernel = Kernel(store, MockCPU({"divide": prog}), log=lambda *a: None)
    kernel.boot()
    pid = kernel.spawn("divide", budget=6)
    kernel.run()
    assert kernel.procs[pid].result == 145, kernel.procs[pid].result

    store.close()
    if os.path.exists(db):
        os.unlink(db)
    print("ALL CALC / PEMDAS TESTS PASSED")


if __name__ == "__main__":
    main()
