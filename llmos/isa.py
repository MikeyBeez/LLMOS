"""The LLMOS instruction set (ISA).

One instruction is one intent the CPU emits per inference cycle. A program is a
sequence of these. Opcodes are deliberately few for v0.1; more get added as the
machine grows.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Op(str, Enum):
    PLAN = "PLAN"            # reason / set a plan; no world effect
    CALL = "CALL"            # syscall to a device (capability-checked)
    READ_MEM = "READ_MEM"    # page a memory key into the window
    WRITE_MEM = "WRITE_MEM"  # persist a value to memory (capability-checked)
    SPAWN = "SPAWN"          # create a child process
    REQUEST = "REQUEST"      # ask the authority (human/policy) to grant a capability
    YIELD = "YIELD"          # hand the CPU back to the scheduler
    RETURN = "RETURN"        # finish, with a result


@dataclass
class Instruction:
    op: Op
    args: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"op": self.op.value, "args": self.args}

    @staticmethod
    def from_dict(d: dict) -> "Instruction":
        return Instruction(Op(d["op"]), d.get("args", {}) or {})
