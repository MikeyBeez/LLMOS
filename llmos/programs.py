"""Built-in demo programs for the MockCPU (deterministic goals).

A program is a callable(pcb) -> Instruction: given the process's context window
so far, it returns the next instruction. The callable form lets the 'CPU' read
earlier results out of its window to build the next instruction, exactly as a
real model would. Shared by the single-process CLI and the out-of-process agent.
"""
from __future__ import annotations

from .isa import Instruction, Op


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


PROGRAMS = {
    "hello": hello_program,
    "ping": ping_program,
}
