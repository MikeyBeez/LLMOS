"""Idle-time scheduling tests: while no foreground process is ready, the kernel
spends the otherwise-idle CPU on background work — cataloging what just finished,
and running background processes on a CHEAPER CPU (big.LITTLE for LLM processors).

Run from the repo root:  PYTHONPATH=. python3 tests/test_idle.py
"""
import os
import tempfile

from llmos.store import Store
from llmos.kernel import Kernel
from llmos.cpu import MockCPU
from llmos.programs import PROGRAMS
from llmos.isa import Instruction, Op


class CheapCPU:
    """Stand-in for a small fast model (e.g. llama on the mac) used for idle work."""
    model = "cheap-bg"

    def __init__(self):
        self.last_meta = {}

    def step(self, pcb):
        self.last_meta = {}
        return Instruction(Op.RETURN, {"result": "bg-reflected"})


def main():
    db = tempfile.mktemp(suffix=".db")
    store = Store(db)
    cheap = CheapCPU()
    kernel = Kernel(store, MockCPU(PROGRAMS), log=lambda *a: None, bg_cpu=cheap)
    kernel.boot()

    # 1. a foreground process is auto-cataloged during idle time
    pid = kernel.spawn("ping", budget=8)
    assert store.mem_read("catalog", f"proc-{pid}") is None      # nothing yet
    kernel.run()
    assert kernel.procs[pid].result == "pong"
    cat = store.mem_read("catalog", f"proc-{pid}")
    assert cat is not None, "expected an idle-time catalog entry for the finished process"
    assert cat["result"] == "pong" and "ping" in cat["wrote"], cat

    # 2. a background process runs during idle time on the CHEAP cpu
    bpid = kernel.spawn("summarize what just happened", budget=4, background=True)
    kernel.run()
    assert kernel.procs[bpid].result == "bg-reflected"

    # 3. prove the split: the background pid ran on the cheap bg CPU, the
    #    foreground pid ran on the primary CPU.
    models = {r["pid"]: r["model"] for r in store.metrics_rows()}
    assert models.get(bpid) == "cheap-bg", models
    assert models.get(pid) is None, models          # MockCPU primary has no model tag

    # 4. a background process is NOT itself cataloged (no runaway housekeeping)
    assert store.mem_read("catalog", f"proc-{bpid}") is None

    store.close()
    if os.path.exists(db):
        os.unlink(db)
    print("ALL IDLE / CURATION / HETEROGENEOUS-CPU TESTS PASSED")


if __name__ == "__main__":
    main()
