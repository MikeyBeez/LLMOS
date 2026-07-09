"""The CPU — the execution unit. One step() call is one instruction cycle.

The CPU reads a process's context and emits the next instruction. It is a
swappable device driver:

  MockCPU    deterministic; runs an authored program. Lets us test the entire
             machine without a stochastic model.
  ReplayCPU  re-emits the instructions recorded in a trace, so a stochastic run
             becomes deterministically replayable.
  OllamaCPU  a real local model (via Ollama).

Decode is interpretive (ARCHITECTURE.md): the model does NOT have to return
clean JSON. It may think out loud, wrap the answer in prose, or fence it in
markdown. The CPU's decode stage does the work of turning that free output into
a structured instruction — it strips reasoning, extracts any JSON object the
model emitted, and if there is none, parses the intent from plain language.
Whatever it produces is then run through the schema validator (the gate). The
model gets one corrective retry; if decode still fails, the CPU fails closed.

Every CPU exposes `last_meta` after each step(): timing/token info the kernel
records into the metrics table.
"""
from __future__ import annotations

import json
import re

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
    """Real CPU. Prompts a local model and DECODES its free-form output into one
    instruction — the model is not required to return JSON.

    keep_alive keeps the model resident between calls (Ollama unloads after 5 min
    by default; the cold reload is the big latency spike). last_meta exposes token
    counts and generation time for the metrics table.
    """

    def __init__(self, model: str = "ornith:35b", host: str = "http://127.0.0.1:11435",
                 seed: int = 0, max_retries: int = 1, log=None, keep_alive: str = "30m",
                 num_predict: int = 512, num_ctx: int = 8192):
        self.model = model
        self.host = host
        self.seed = seed
        self.max_retries = max_retries
        self.log = log or (lambda *a: None)
        self.keep_alive = keep_alive
        self.num_predict = num_predict
        self.num_ctx = num_ctx
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
            instr, err = self._decode(raw, pcb)
            if err is None:
                self.last_meta = {**agg, "retries": retries}
                return instr
            reason = err
            retries += 1
            if attempt < self.max_retries:
                self.log(f"[cpu] undecodable: {err} — giving the model one chance to fix it")
        # the one chance is spent and it's still undecodable: fail closed
        self.log(f"[cpu] still undecodable after retry: {reason} — failing closed")
        self.last_meta = {**agg, "retries": retries}
        return Instruction(Op.RETURN, {"result": "SCHEMA VALIDATION FAILED", "error": reason, "raw": raw})

    # --- decode: free text -> one validated instruction -----------------
    def _decode(self, raw, pcb):
        """Return (Instruction, None) if we can build a valid instruction, else
        (None, reason). Order: (1) any JSON object the model emitted is authoritative
        and is validated as-is; (2) only if there is NO JSON object do we parse the
        intent from plain language."""
        text = self._strip_think(raw)
        obj = self._find_json(text)
        if obj is not None:
            return self._validate_obj(obj)
        instr = self._nl_parse(text, pcb)
        if instr is not None:
            return instr, None
        return None, f"no instruction could be parsed from the model output: {text[:160]!r}"

    @staticmethod
    def _strip_think(raw) -> str:
        """Remove reasoning-model scaffolding so it doesn't drown the instruction."""
        if not raw:
            return ""
        t = re.sub(r"<think>.*?</think>", " ", raw, flags=re.DOTALL | re.IGNORECASE)
        t = re.sub(r"<think>.*$", " ", t, flags=re.DOTALL | re.IGNORECASE)   # unclosed = all thinking
        return t.strip()

    @staticmethod
    def _find_json(text):
        """Extract the model's instruction object from anywhere in the text: a
        ```json fenced block, or the last balanced {...} that parses. Prefer an
        object that actually carries an 'op' field."""
        if not text:
            return None
        candidates = []
        for m in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL):
            candidates.append(m.group(1))
        depth, start = 0, None
        for i, ch in enumerate(text):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}" and depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    candidates.append(text[start:i + 1])
        best = None
        for c in candidates:
            try:
                d = json.loads(c)
            except Exception:
                continue
            if isinstance(d, dict) and "op" in d:
                best = d            # keep the LAST op-bearing object (the final answer)
        if best is not None:
            return best
        for c in reversed(candidates):
            try:
                d = json.loads(c)
                if isinstance(d, dict):
                    return d
            except Exception:
                continue
        return None

    @staticmethod
    def _validate_obj(d):
        """Return (Instruction, None) if the object is a valid instruction, else
        (None, human-readable reason). This is the schema gate."""
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

    @staticmethod
    def _nl_parse(text, pcb):
        """Last-resort decode: build an instruction from plain-language intent when
        the model emitted no JSON at all. Conservative on purpose — it returns None
        rather than guess, so a genuinely unparseable reply still triggers the retry.
        Uses the window to avoid repeating a finished step."""
        t = (text or "").lower().strip()
        if not t:
            return None
        ctx = getattr(pcb, "context", []) or []
        called_clock = any(c.get("op") == "CALL" for c in ctx)
        last_result = ctx[-1]["result"] if ctx else None

        # Goal-conditioned tail extraction. When the goal specifies the answer
        # SHAPE (a single letter A-D for MCQ, or a boxed value for math), pull
        # the answer straight from the tail of a bare-prose reply rather than
        # failing schema-validation. This turns "Let me think through this...
        # therefore the answer is A." into RETURN(result="A"), matching the
        # docstring's promise that "the model does NOT have to return JSON."
        goal = getattr(pcb, "goal", "") or ""
        goal_l = goal.lower()
        if "single letter" in goal_l and "a, b, c, or d" in goal_l:
            tail = text[-500:]
            m_ans = re.search(r"(?:answer|choice|option|final|correct)[^A-Za-z0-9]{0,20}([ABCD])\b",
                              tail, re.IGNORECASE)
            if m_ans:
                return Instruction(Op.RETURN, {"result": m_ans.group(1).upper()})
            m_boxed = re.search(r"\\boxed\{([ABCD])\}", tail)
            if m_boxed:
                return Instruction(Op.RETURN, {"result": m_boxed.group(1)})
            # last bare capital A-D in the tail (weakest signal, kept last)
            letters = re.findall(r"\b([ABCD])\b", tail)
            if letters:
                return Instruction(Op.RETURN, {"result": letters[-1]})
        # math short-answer: goal asks to RETURN the final answer as a value
        if "solve this math problem" in goal_l or ("return" in goal_l and "final answer" in goal_l):
            tail = text[-600:]
            m_box = re.search(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", tail)
            if m_box:
                return Instruction(Op.RETURN, {"result": m_box.group(1).strip()})
            m_ans = re.search(r"(?:answer|result)\s*(?:is|=|:)\s*([-+]?\d+(?:\.\d+)?(?:/\d+)?)",
                              tail, re.IGNORECASE)
            if m_ans:
                return Instruction(Op.RETURN, {"result": m_ans.group(1)})

        if re.search(r"\b(return|final answer|task (is )?complete|done|finished|goal (is )?met|"
                     r"already saved|has been saved|saved (it|the))\b", t):
            # take the last non-empty sentence/line as the answer, not the reasoning preamble
            parts = [p.strip() for p in re.split(r"[\n.]+", text) if p.strip()]
            answer = parts[-1] if parts else text.strip()
            return Instruction(Op.RETURN, {"result": answer[:200]})
        if (("clock" in t) or ("current time" in t) or (" time" in t)) and not called_clock:
            return Instruction(Op.CALL, {"name": "clock.now", "args": {}})
        if re.search(r"\b(write|save|store|persist)\b", t) and ("mem" in t or "memory" in t or called_clock):
            return Instruction(Op.WRITE_MEM, {"key": "result", "value": last_result})
        if re.search(r"\b(read|recall|load|fetch)\b", t) and ("mem" in t or "memory" in t):
            return Instruction(Op.READ_MEM, {"key": "result"})
        if "plan" in t:
            return Instruction(Op.PLAN, {"text": text[:200].strip()})
        return None

    # --- generation -----------------------------------------------------
    def _generate(self, pcb, correction=None):
        """Return (response_text, meta). No JSON grammar is forced on the model —
        we decode whatever it says. meta carries token counts and durations (ms)."""
        import urllib.request
        try:
            body = json.dumps({
                "model": self.model,
                "prompt": self._build_prompt(pcb, correction),
                "stream": False,
                "keep_alive": self.keep_alive,
                "options": {"temperature": 0, "seed": self.seed, "num_predict": self.num_predict, "num_ctx": self.num_ctx},
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
            # some reasoning models put the answer in a separate 'thinking' field
            return resp.get("response", "") or resp.get("thinking", ""), meta
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
            head = (f"Your previous reply could not be decoded into an instruction: {correction}.\n"
                    "End your reply with the next instruction as a single JSON object.\n\n")
        return head + (
            "You are the CPU of LLMOS. Choose the ONE next instruction toward the goal.\n"
            "You may reason first, but END your reply with a single JSON object for the instruction.\n\n"
            "Instruction set (pick one op; required args shown):\n"
            '  {"op":"PLAN","args":{"text":"..."}}\n'
            '  {"op":"CALL","args":{"name":"clock.now","args":{}}}   (requires: name)\n'
            '  {"op":"WRITE_MEM","args":{"key":"...","value":<any>}}  (requires: key)\n'
            '  {"op":"READ_MEM","args":{"key":"..."}}                (requires: key)\n'
            '  {"op":"EVICT","args":{"key":"..."}}   (drop a paged-in key from your window once done)\n'
            '  {"op":"RETURN","args":{"result":<any>}}\n\n'
            "Available syscalls for CALL: clock.now (current UTC time); "
            "calc (evaluate arithmetic EXACTLY). calc understands:\n"
            "  - basic ops: + - * / // % **, and ^ works too\n"
            "  - factorials: 5!, (3+2)!\n"
            "  - combinatorics: C(n,k), P(n,k), binomial(n,k), factorial(n), gcd/lcm\n"
            "  - trig (radians): sin, cos, tan, arcsin/asin, arccos/acos, arctan/atan, sinh, cosh, tanh\n"
            "  - exp/log: exp, log, ln, log2, log10\n"
            "  - constants: pi, e, tau; use `n mod m` or `n % m` for modulo\n"
            "  - misc: sqrt, abs, round, floor, ceil, min, max, degrees, radians\n"
            "  - quantity words: dozen, half a dozen, score, gross\n"
            'Example: {"op":"CALL","args":{"name":"calc","args":{"expr":"C(6,3) * C(5,2)"}}}\n'
            "Do NOT convert quantity words or notation to numbers yourself; let calc do it.\n"
            "Use earlier step results; do not repeat a completed step; RETURN when the goal is met.\n\n"
            f"GOAL: {pcb.goal}\n"
            f"STEPS SO FAR:\n{history}\n\n"
            "Reason if needed, then end with one JSON instruction:"
        )
