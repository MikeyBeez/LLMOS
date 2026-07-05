"""Project-scoping tests: the project is the top-level category, applied first.
Two projects can share a topic name ('kernel') that means different things; working
in one project must not pull the other's memory into the window.

Run from the repo root:  PYTHONPATH=. python3 tests/test_project.py
"""
import os
import tempfile

from llmos.store import Store
from llmos.kernel import Kernel
from llmos.cpu import MockCPU


def paged(pcb):
    return {c["args"]["key"] for c in pcb.context if c["op"] == "READ_MEM"}


def main():
    db = tempfile.mktemp(suffix=".db")
    store = Store(db)
    # same topic word 'kernel' in two different projects
    store.mem_write("mem", "kernel_loop", "the deterministic fetch-decode loop", topic="kernel", project="LLMOS")
    store.mem_write("mem", "evict_op", "drop paged-in spans from the window", topic="kernel", project="LLMOS")
    store.mem_write("mem", "gemm_kernel", "a CUDA kernel for matrix multiply", topic="kernel", project="HRS")

    # working in the LLMOS project
    kernel = Kernel(store, MockCPU({}), log=lambda *a: None, project="LLMOS")
    kernel.boot()
    pid = kernel.spawn("explain the kernel loop and evict", budget=8)
    pcb = kernel.procs[pid]

    assert pcb.topic == "kernel", pcb.topic
    # only LLMOS-project 'kernel' memory is loaded; the HRS one is NOT
    assert paged(pcb) == {"kernel_loop", "evict_op"}, paged(pcb)
    assert "gemm_kernel" not in paged(pcb)

    # the same store, entered as the HRS project, sees only the HRS 'kernel' memory
    hrs = Kernel(store, MockCPU({}), log=lambda *a: None, project="HRS")
    hrs.boot()
    pid2 = hrs.spawn("what does the kernel do", budget=8)
    assert paged(hrs.procs[pid2]) == {"gemm_kernel"}, paged(hrs.procs[pid2])

    store.close()
    if os.path.exists(db):
        os.unlink(db)
    print("ALL PROJECT-SCOPING TESTS PASSED")


if __name__ == "__main__":
    main()
