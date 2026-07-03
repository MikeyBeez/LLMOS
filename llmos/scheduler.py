"""Scheduler — decides which ready process gets the next (expensive) inference cycle.

v0.1: cooperative, FIFO ready-queue. A process runs until it YIELDs or RETURNs,
with a hard per-process budget as the preemption safety net. Preemptive policies
(and, on the hosted runtime, SIGSTOP/SIGCONT over real macOS processes) layer on
top of this same interface later.
"""
from __future__ import annotations

from collections import deque


class Scheduler:
    def __init__(self):
        self.ready: deque[int] = deque()

    def add(self, pid: int) -> None:
        self.ready.append(pid)

    def next(self) -> int | None:
        return self.ready.popleft() if self.ready else None

    def has_work(self) -> bool:
        return bool(self.ready)
