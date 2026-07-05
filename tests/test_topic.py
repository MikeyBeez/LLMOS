"""Topic routing tests: a process loads only the context for its topic, and when
the conversation switches topics the old topic is evicted and the new one paged in.

Mikey's example: talking about real estate, then particle physics — you don't need
the real-estate context while doing physics, and vice versa.

Run from the repo root:  PYTHONPATH=. python3 tests/test_topic.py
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
    # two clearly separate topics
    store.mem_write("mem", "cap_rate", "NOI divided by price", topic="real_estate")
    store.mem_write("mem", "mortgage", "a loan secured by property", topic="real_estate")
    store.mem_write("mem", "quark", "a fundamental particle", topic="particle_physics")
    store.mem_write("mem", "boson", "a force-carrying particle", topic="particle_physics")

    kernel = Kernel(store, MockCPU({}), log=lambda *a: None)
    kernel.boot()

    # talking about real estate -> only real-estate context is loaded
    pid = kernel.spawn("what is a good cap rate for a rental property", budget=8)
    pcb = kernel.procs[pid]
    assert pcb.topic == "real_estate", pcb.topic
    assert paged(pcb) == {"cap_rate", "mortgage"}, paged(pcb)
    assert "quark" not in paged(pcb) and "boson" not in paged(pcb)   # physics NOT loaded

    # the conversation shifts to particle physics -> evict real estate, load physics
    kernel.switch_topic(pcb, "particle_physics")
    assert pcb.topic == "particle_physics"
    assert paged(pcb) == {"quark", "boson"}, paged(pcb)
    assert "cap_rate" not in paged(pcb) and "mortgage" not in paged(pcb)  # real estate evicted

    # and a fresh goal classifies to physics on its own
    pid2 = kernel.spawn("explain what a boson is", budget=8)
    assert kernel.procs[pid2].topic == "particle_physics", kernel.procs[pid2].topic
    assert paged(kernel.procs[pid2]) == {"quark", "boson"}

    # the evicted topic still lives on disk (nothing was lost)
    assert store.mem_read("mem", "cap_rate") == "NOI divided by price"

    store.close()
    if os.path.exists(db):
        os.unlink(db)
    print("ALL TOPIC-ROUTING TESTS PASSED")


if __name__ == "__main__":
    main()
