"""Built-in demo programs for the MockCPU (deterministic goals).

A program is a callable(pcb) -> Instruction: given the process's context window
so far, it returns the next instruction. The callable form lets the 'CPU' read
earlier results out of its window to build the next instruction, exactly as a
real model would. Shared by the single-process CLI and the out-of-process agent.
"""
from __future__ import annotations

import os

from .isa import Instruction, Op

_EXAMPLES = os.path.expanduser("~/Code/LLMOS/examples")
TRUSTED_FILE = os.path.join(_EXAMPLES, "trusted", "config.txt")
UNTRUSTED_FILE = os.path.join(_EXAMPLES, "untrusted", "note.txt")


def hello_program(pcb) -> Instruction:
    pc = pcb.pc
    if pc == 0:
        return Instruction(Op.PLAN, {"text": "read the clock, then persist it to memory"})
    if pc == 1:
        return Instruction(Op.CALL, {"name": "clock.now", "args": {}})
    if pc == 2:
        t = pcb.context[-1]["result"]           # read the CALL result from the window
        return Instruction(Op.WRITE_MEM, {"ns": "mem", "key": "hello.timestamp", "value": t})
    if pc == 3:
        return Instruction(Op.YIELD, {})
    return Instruction(Op.RETURN, {"result": {"saved": "hello.timestamp"}})


def ping_program(pcb) -> Instruction:
    pc = pcb.pc
    if pc == 0:
        return Instruction(Op.PLAN, {"text": "answer with pong"})
    if pc == 1:
        return Instruction(Op.WRITE_MEM, {"ns": "mem", "key": "ping", "value": "pong"})
    return Instruction(Op.RETURN, {"result": "pong"})


def readgood_program(pcb) -> Instruction:
    """Reads a trusted config file and stores a value from it. Succeeds."""
    pc = pcb.pc
    if pc == 0:
        return Instruction(Op.CALL, {"name": "fs.read", "args": {"path": TRUSTED_FILE}})
    if pc == 1:
        data = pcb.context[-1]["result"] or {}
        return Instruction(Op.WRITE_MEM, {"key": "config", "value": data.get("content", "").strip()})
    return Instruction(Op.RETURN, {"result": "loaded trusted config"})


def readbad_program(pcb) -> Instruction:
    """Simulates a naive CPU that reads attacker-controlled content and 'obeys' an
    instruction embedded in it. The kernel revokes privileges the moment the
    untrusted data is read, so the follow-on WRITE_MEM is denied at the boundary —
    the injection fails even though the CPU was fooled."""
    pc = pcb.pc
    if pc == 0:
        return Instruction(Op.CALL, {"name": "fs.read", "args": {"path": UNTRUSTED_FILE}})
    if pc == 1:
        return Instruction(Op.WRITE_MEM, {"key": "secret", "value": "exfiltrated-by-injection"})
    return Instruction(Op.RETURN, {"result": "done"})


PROGRAMS = {
    "hello": hello_program,
    "ping": ping_program,
    "readgood": readgood_program,
    "readbad": readbad_program,
}
