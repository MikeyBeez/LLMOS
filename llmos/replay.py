"""Replay — the payoff of a single-writer trace.

Because the kernel records every instruction and its arguments, the ledger is a
complete record of a run. Re-applying the recorded write stream to a fresh store
reproduces byte-identical memory state, without touching any nondeterministic
device (the clock's value was captured in the write's args at record time).

This is what turns a stochastic CPU into something auditable: you may not be able
to reproduce the *thinking*, but you can always reproduce the *computation* from
its trace.
"""
from __future__ import annotations

from .store import Store


def replay(orig_store: Store, pid: int, tmp_path: str):
    rows = orig_store.trace_read(pid)
    fresh = Store(tmp_path)
    applied = []
    for row in rows:
        if row["op"] == "WRITE_MEM":
            a = row["args"]
            fresh.mem_write(a.get("ns", "mem"), a["key"], a.get("value"), "replay")
            applied.append((a.get("ns", "mem"), a["key"]))

    ok = True
    diffs = []
    for ns, key in applied:
        if fresh.mem_read(ns, key) != orig_store.mem_read(ns, key):
            ok = False
            diffs.append(f"{ns}/{key}")
    fresh.close()
    return ok, len(rows), applied, diffs
