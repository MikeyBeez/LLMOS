"""The CPU — the execution unit. One step() call is one instruction cycle.

The CPU reads a process's context and emits the next instruction. It is a
swappable device driver:

  MockCPU    deterministic; runs an authored program. Lets us test the entire
             machine without a stochastic model.
  ReplayCPU  re-emits the instructions recorded in a trace, so a stochastic run
             becomes deterministically replayable.
  OllamaCPU  a real local model (via Ollama). Gives the model exactly ONE chance
             to fix an invalid instruction, then fails closed.

Every CPU exposes `last_meta` after each step(): a small dict of timing/token
info the kernel records into the metrics table. For the deterministic CPUs it is
empty (the kernel still times the call itself); for OllamaCPU it carries the
model's token counts and generation time, so we can see where the seconds go.
"""
from __future__ import annotations

import json

from .isa import Instruction, Op, missing_args


class MockCPU:
    """programs maps goal -> either a list[Instruction] or a callable(pcb)->Instruction.
    The callable form lets the 'CPU' read prior results from the window to build the
    next instruction, which is exactly what a real model does."""

    def __init__(self, programs: dict):
        self.programs = programs
        self.model = None
        self.last_meta: dict = {}

    def step(self, pcb) -> Instruction:
        self.last_meta = {}
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
        self.model = None
        self.last_meta: dict = {}

    def step(self, pcb) -> Instruction:
        self.last_meta = {}
        if pcb.pc >= len(self.rows):
            return Instruction(Op.RETURN, {"result": "trace exhausted"})
        row = self.rows[pcb.pc]
        return Instruction(Op(row["op"]), row["args"])


class OllamaCPU:
    """Real CPU. Prompts a local model and parses its JSON output into one instruction.

    Schema handling (Mikey, 2026-07-03): the model gets exactly ONE chance to fix an
    invalid instruction. If the first emission is malformed — not JSON, unknown
    opcode, or missing a required field — the CPU sends a structured correction and
    regenerates once. If it is still invalid, the CPU fails closed: it returns a
    terminating RETURN carrying the error, which the kernel records in the trace.

    keep_alive keeps the model resident between instructions/runs (Ollama unloads
    after 5 min by default; a cold reload is the known latency spike). last_meta
    exposes token counts and generation time for the metrics table.
    """

    def __init__(self, model: str = "qwen2.5:latest", host: str = "http://localhost:11434",
                 seed: int = 0, max_retries: int = 1, log=None, keep_alive: str = "30m",
                 num_predict: int = 200):
        self.model = model
        self.host = host
        self.seed = seed
        self.max_retries = max_retries
        self.log = log or (lambda *a: None)
        self.keep_alive = keep_alive
        self.num_predict = num_predict
        self.last_meta: dict = {}

    def step(self, pcb) -> Instruction:
        reason = None
        raw = None
        retries = 0
        agg = {"prompt_tokens": 0, "eval_tokens": 0, "eval_ms": 0.0, "load_ms": 0.0}
        for attempt in range(self.max_retries + 1):
            out = self._generate(pcb, correction=reason)
            raw, meta = out if isinstance(out, tuple) else (out, {})
            for k in agg:
                agg[k] += (meta.get(k) or 0)
            instr, err = self._parse_and_validate(raw)
            if err is None:
                self.last_meta = {**agg, "retries": retries}
                return instr
            reason = err
            retries += 1
            if attempt < self.max_retries:
                self.log(f"[cpu] rejected: {err} — giving the model one chance to fix it")
        # the one chance is spent and it's still invalid: throw the error (fail closed)
        self.log(f"[cpu] still invalid after retry: {reason} — failing closed")
        self.last_meta = {**agg, "retries": retries}
        return Instruction(Op.RETURN, {"result": "SCHEMA VALIDATION FAILED", "error": reason, "raw": raw})

    def _parse_and_validate(self, raw):
        """Return (Instruction, None) if valid, else (None, human-readable reason)."""
        try:
            d = json.loads(raw)
        except (json.JSONDecodeError, TypeError) as e:
            return None, f"output was not valid JSON ({e})"
        if not isinstance(d, dict) or "op" not in d:
            return None, "output must be a JSON object with an 'op' field"
        try:
            op = Op(str(d["op"]).strip().upper())
        except ValueError:
            return None, f"unknown op {d.get('op')!r}; must be one of {[o.value for o in Op]}"
        args = d.get("args") or {}
        if not isinstance(args, dict):
            return None, "'args' must be an object"
        miss = missing_args(op, args)
        if miss:
            return None, f"{op.value} requires field(s) {miss}"
        return Instruction(op, args), None

    def _generate(self, pcb, correction=None):
        """Return (response_text, meta). meta carries token counts and durations
        (ms) pulled straight from Ollama's own timing fields."""
        import urllib.request
        try:
            body = json.dumps({
                "model": self.model,
                "prompt": self._build_prompt(pcb, correction),
                "stream": False,
                "format": "json",
                "keep_alive": self.keep_alive,
                "options": {"temperature": 0, "seed": self.seed, "num_predict": self.num_predict},
            }).encode()
            req = urllib.request.Request(
                self.host + "/api/generate", data=body,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=180) as r:
                resp = json.loads(r.read())
            meta = {
                "prompt_tokens": resp.get("prompt_eval_count"),
                "eval_tokens": resp.get("eval_count"),
                "eval_ms": (resp.get("eval_duration") or 0) / 1e6,
                "load_ms": (resp.get("load_duration") or 0) / 1e6,
            }
            return resp.get("response", ""), meta
        except Exception as e:
            # device error (model down/timeout): fail closed with a valid terminating instruction
            return json.dumps({"op": "RETURN", "args": {"result": "CPU device error", "error": str(e)}}), {}

    def _build_prompt(self, pcb, correction=None) -> str:
        history = "\n".join(
            f"  step {s['pc']}: {s['op']} {json.dumps(s['args'])} -> {json.dumps(s['result'])}"
            for s in pcb.context
        ) or "  (none yet)"
        head = ""
        if correction:
            head = (f"Your previous output was REJECTED: {correction}.\n"
                    "Return exactly ONE corrected JSON instruction and nothing else.\n\n")
        return head + (
            "You are the CPU of LLMOS. Emit exactly ONE instruction as a single JSON object "
            "and nothing else.\n\n"
            "Instruction set (pick one op; required args shown):\n"
            '  {"op":"PLAN","args":{"text":"..."}}\n'
            '  {"op":"CALL","args":{"name":"clock.now","args":{}}}   (requires: name)\n'
            '  {"op":"WRITE_MEM","args":{"key":"...","value":<any>}}  (requires: key)\n'
            '  {"op":"READ_MEM","args":{"key":"..."}}                (requires: key)\n'
            '  {"op":"RETURN","args":{"result":<any>}}\n\n'
            "Available syscalls for CALL: clock.now (returns the current UTC time).\n"
            "Use earlier step results; do not repeat a completed step; RETURN when the goal is met.\n\n"
            f"GOAL: {pcb.goal}\n"
            f"STEPS SO FAR:\n{history}\n\n"
            "Next instruction (one JSON object):"
        )
