"""Topic-dependency tests: when a topic uses another topic, the index records it,
and loading the topic also loads its dependency's context.

Run from the repo root:  PYTHONPATH=. python3 tests/test_topic_deps.py
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
    store.mem_write("mem", "black_scholes", "an option-pricing model", topic="derivatives")
    store.mem_write("mem", "normal_dist", "the gaussian distribution", topic="statistics")
    store.mem_write("mem", "roux", "a thickener", topic="cooking")   # unrelated topic

    kernel = Kernel(store, MockCPU({}), log=lambda *a: None)
    kernel.boot()

    # derivatives depends on statistics
    kernel.link_topics("derivatives", "statistics")

    pid = kernel.spawn("price this option with black scholes", topic="derivatives", budget=8)
    pcb = kernel.procs[pid]

    # loading 'derivatives' also pulled in its dependency 'statistics'
    assert paged(pcb) == {"black_scholes", "normal_dist"}, paged(pcb)
    # but NOT the unrelated topic
    assert "roux" not in paged(pcb)

    # the dependency is recorded in the index
    e = store.mem_read("topic_index", "derivatives")
    assert e.get("uses") == ["statistics"], e

    store.close()
    if os.path.exists(db):
        os.unlink(db)
    print("ALL TOPIC-DEPENDENCY TESTS PASSED")


if __name__ == "__main__":
    main()
