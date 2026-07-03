"""The CPU — the execution unit. One step() call is one instruction cycle.

The CPU reads a process's context and emits the next instruction. It is a
swappable device driver:

  MockCPU    deterministic; runs an authored program. Lets us test the entire
             machine without a stochastic model.
  ReplayCPU  re-emits the instructions recorded in a trace, so a stochastic run
             becomes deterministically replayable.
  OllamaCPU  a real local model (via Ollama), temperature 0 + fixed seed.
"""
from __future__ import annotations

import json

from .isa import Instruction, Op


class MockCPU:
    """programs maps goal -> either a list[Instruction] or a callable(pcb)->Instruction.
    The callable form lets the 'CPU' read prior results from the window to build the
    next instruction, which is exactly what a real model does."""

    def __init__(self, programs: dict):
        self.programs = programs

    def step(self, pcb) -> Instruction:
        prog = self.programs.get(pcb.goal)
        if prog is None:
            return Instruction(Op.RETURN, {"result": f"no program for goal: {pcb.goal}"})
        if callable(prog):
            return prog(pcb)
        if pcb.pc >= len(prog):
            return Instruction(Op.RETURN, {"result": "program exhausted"})
        return prog[pcb.pc]


class ReplayCPU:
    """Re-emits the exact instruction stream from a recorded trace."""

    def __init__(self, trace_rows: list[dict]):
        self.rows = trace_rows

    def step(self, pcb) -> Instruction:
        if pcb.pc >= len(self.rows):
            return Instruction(Op.RETURN, {"result": "trace exhausted"})
        row = self.rows[pcb.pc]
        return Instruction(Op(row["op"]), row["args"])


class OllamaCPU:
    """Real CPU. Prompts a local model and parses structured JSON output into one
    instruction. The model must be taught the ISA (future work); wired here so the
    driver exists and can be pointed at the mini or pop's GPU."""

    def __init__(self, model: str = "llama3.2", host: str = "http://localhost:11434", seed: int = 0):
        self.model = model
        self.host = host
        self.seed = seed

    def step(self, pcb) -> Instruction:
        import urllib.request

        body = json.dumps({
            "model": self.model,
            "prompt": self._build_prompt(pcb),
            "stream": False,
            "format": "json",
            "options": {"temperature": 0, "seed": self.seed},
        }).encode()
        req = urllib.request.Request(
            self.host + "/api/generate", data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            resp = json.loads(r.read())
        d = json.loads(resp.get("response", "{}"))
        return Instruction(Op(d["op"]), d.get("args", {}) or {})

    def _build_prompt(self, pcb) -> str:
        return (
            "You are the CPU of LLMOS. Emit the SINGLE next instruction as JSON: "
            '{"op": one of PLAN|CALL|READ_MEM|WRITE_MEM|SPAWN|YIELD|RETURN, "args": {...}}.\n'
            f"Goal: {pcb.goal}\n"
            f"Steps so far: {json.dumps(pcb.context)}\n"
            "Next instruction:"
        )
