"""Process Control Block — the serializable state that makes a process checkpointable.

An LLMOS process is an agent with its own goal, program counter, capability set,
and a context window (its RAM). The PCB is everything needed to stop it, persist
it, and resume it later.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


class Status(str, Enum):
    NEW = "NEW"
    READY = "READY"
    RUNNING = "RUNNING"
    YIELDED = "YIELDED"
    DONE = "DONE"
    KILLED = "KILLED"


@dataclass
class PCB:
    pid: int
    goal: str
    ppid: int | None = None
    pc: int = 0                                    # program counter
    status: Status = Status.NEW
    capabilities: set[str] = field(default_factory=set)
    budget: int = 32                               # instruction cycles before preemption
    working_set: list[str] = field(default_factory=list)   # memory keys paged in
    context: list[dict] = field(default_factory=list)      # the window: recent step records
    result: Any = None
    tainted: bool = False               # has untrusted data entered this process's window?
    contract: dict = field(default_factory=dict)   # required postconditions (keys that must exist before RETURN)
    contract_tries: int = 0                         # how many times the kernel has re-trapped a premature RETURN
    background: bool = False                        # runs in idle time (may use a cheaper CPU)
    topic: str = "general"                          # which topic's context this process loads

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        d["capabilities"] = sorted(self.capabilities)
        return d
