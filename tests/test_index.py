"""Topic-index tests: a new prompt is checked against the index and FITS an
existing topic when it can, only minting a new topic when nothing fits — so topics
don't proliferate. The index accumulates each topic's prompts and responses.

Run from the repo root:  PYTHONPATH=. python3 tests/test_index.py
"""
import os
import tempfile

from llmos.store import Store
from llmos.kernel import Kernel
from llmos.cpu import MockCPU


def main():
    db = tempfile.mktemp(suffix=".db")
    store = Store(db)
    kernel = Kernel(store, MockCPU({}), log=lambda *a: None)
    kernel.boot()

    t1 = kernel.route_topic("how do I refinance my mortgage on a rental property")
    t2 = kernel.route_topic("what is a fair mortgage rate for a rental property")   # should FIT t1
    t3 = kernel.route_topic("explain quark boson particle physics interactions")     # should be NEW

    idx = {t: store.mem_read("topic_index", t) for t in store.mem_list("topic_index")}

    assert t1 == t2, f"related prompts should share a topic: {t1} vs {t2}"
    assert t3 != t1, f"unrelated prompt should get its own topic: {t3} vs {t1}"
    assert len(idx) == 2, f"expected 2 topics (fit, not proliferate), got {list(idx)}"
    assert len(idx[t1]["prompts"]) == 2, idx[t1]["prompts"]
    assert len(idx[t3]["prompts"]) == 1, idx[t3]["prompts"]

    # responses attach to the topic's list
    kernel.record_response(t1, "how do I refinance my mortgage", "Refinance by comparing lenders...")
    e = store.mem_read("topic_index", t1)
    assert e["entries"] and e["entries"][0]["response"].startswith("Refinance"), e

    store.close()
    if os.path.exists(db):
        os.unlink(db)
    print("ALL TOPIC-INDEX TESTS PASSED")


if __name__ == "__main__":
    main()
