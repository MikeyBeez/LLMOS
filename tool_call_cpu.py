"""ToolCallCPU — drive ornith through its NATIVE tool-calling instead of our
interpretive JSON-ISA.

Per the Ornith-1.0-35B model card (huggingface.co/deepreinforce-ai/Ornith-1.0-35B),
the model is trained to:
  1. Open the assistant turn with a <think>...</think> reasoning block
  2. Emit <tool_call>...</tool_call> XML blocks that a server-side parser
     surfaces as OpenAI-style tool_calls
  3. Answer at temperature 0.6, top_p 0.95, top_k 20

Ollama's --chat-template chatml + qwen3 tool-call parsing does the XML->JSON
work automatically when we hit /api/chat with `tools=[...]`. The kernel still
receives ordinary LLMOS Instructions.

Modeled directly on swe_agent.py's CodingCPU, generalized so MMLU / MATH /
other benchmark runners can plug in their own tool schemas.
"""
import json, os, time, urllib.request, urllib.error

from llmos.cpu import OllamaCPU
from llmos.isa import Instruction, Op


# PHASE_DEADLINE (2026-08-27). Unix ts by which the current agent phase must
# end, set by phase_run before every turn -- the same module-global mechanism
# already used for test_runner and swe_fix_tools, which is why this file gets
# no new machinery.
#
# WHY IT WAS NEEDED HERE AND NOT ONLY THERE. The tool call was made
# deadline-aware on 2026-08-24; the MODEL call never was, and it is the one
# that can block longest. _chat sends stream:False and one blocking urlopen
# with request_timeout=600, inside an anti-truncation loop of up to 3
# max_tokens doublings x 6 HTTP attempts, and phase_run wraps THAT in 3 more
# attempts with 20/40/60s sleeps. Worst case for a SINGLE TURN is ~1900s
# against a 960s segment budget and a 2400s repertoire wall, and the wall
# check only runs BETWEEN turns -- so one hung llama-server eats most of an
# instance and the harness only finds out afterwards.
#
# This is the timer interrupt: a phase can now stop an instruction, not just
# notice after it returned.
PHASE_DEADLINE = None


def _budget_left():
    """Seconds until the phase deadline, or None when no deadline is set."""
    if not PHASE_DEADLINE:
        return None
    return PHASE_DEADLINE - time.time()


class ToolCallCPU(OllamaCPU):
    """Drop-in replacement for OllamaCPU that uses /api/chat + tools.

    tools:        list of OpenAI-format function tool schemas
    tool2sys:     dict mapping tool_name -> LLMOS syscall name
                  (special names 'finish' and 'return' emit RETURN instead of CALL)
    system_prompt: text prepended as the {"role":"system"} message
    """

    def __init__(self, tools, tool2sys, system_prompt="",
                 model="ornith:35b", host="http://127.0.0.1:8080",
                 temperature=0.6, num_predict=2048, num_ctx=65536,
                 seed=0, keep_alive="24h", log=None,
                 request_timeout=600, budget_recent=6000, budget_old=1200,
                 recent_turns=16):
        super().__init__(model=model, host=host, seed=seed, log=log,
                         keep_alive=keep_alive,
                         num_predict=num_predict, num_ctx=num_ctx)
        self.tools = tools
        self.tool2sys = tool2sys
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.request_timeout = request_timeout
        # How much of each prior tool result the model actually gets to see.
        # 1800 (the old fixed head-clip) starved it: pytest back-loads the
        # assertion diff, so the model saw boilerplate and never the reason.
        self.budget_recent = int(os.environ.get("TOOL_BUDGET_RECENT", budget_recent))
        self.budget_old = int(os.environ.get("TOOL_BUDGET_OLD", budget_old))
        self.recent_turns = int(os.environ.get("TOOL_RECENT_TURNS", recent_turns))
        # per-turn accounting of what the clip withheld from the model
        self._clip_drop = 0
        self._clip_events = 0

    # --- Override step() rather than _generate(): tool-calling doesn't go through
    # the interpretive JSON-ISA decode path at all. ---
    def step(self, pcb):
        try:
            msg, meta = self._chat(self._messages(pcb))
        except Exception as e:
            self.last_meta = {"retries": 0}
            return Instruction(Op.RETURN,
                               {"result": "CPU device error", "error": str(e)})
        self.last_meta = meta
        if isinstance(meta, dict):
            # what the model was NOT shown this turn. prompt_tokens says what it
            # got; without this the trace cannot tell a starved agent from a dumb one.
            meta["clipped_chars"] = self._clip_drop
            meta["clipped_msgs"] = self._clip_events
        tcs = msg.get("tool_calls") or []
        if not tcs:
            # Model reasoned but didn't call a tool. Extract thinking, ask the
            # scheduler to give the model another turn with a nudge.
            txt = (msg.get("content") or msg.get("thinking") or "").strip()
            return Instruction(Op.PLAN, {"text": (txt[:400] or "continue")})
        fn = tcs[0].get("function", {})
        tool = fn.get("name", "")
        args = fn.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {"_raw": args}
        # 'finish' and 'return' are conventional names for the terminal tool call.
        target = self.tool2sys.get(tool, "")
        if target in ("RETURN", "finish", "return") or tool in ("finish", "return"):
            # Prefer common result-carrying keys: 'result', 'answer', 'summary'.
            result = args.get("result", args.get("answer", args.get("summary", args)))
            return Instruction(Op.RETURN, {"result": result})
        if not target:
            return Instruction(Op.PLAN,
                               {"text": f"unknown tool {tool!r}; args={args!r}"})
        return Instruction(Op.CALL, {"name": target, "args": args})

    # --- message assembly ------------------------------------------------
    def _messages(self, pcb):
        """Build the conversation from the process context. System prompt + user
        goal + one assistant/tool pair per prior CALL/RETURN step."""
        msgs = []
        self._clip_drop = 0
        self._clip_events = 0
        if self.system_prompt:
            msgs.append({"role": "system", "content": self.system_prompt})
        msgs.append({"role": "user", "content": pcb.goal})
        n = len(pcb.context)
        for idx, s in enumerate(pcb.context):
            recent = (n - idx) <= self.recent_turns
            msgs.extend(self._pair_for(
                s, self.budget_recent if recent else self.budget_old))
        return msgs

    def _clip_result(self, result, budget):
        """Serialize a tool result for the model, keeping BOTH ends.

        A head-only clip is exactly wrong for test output: pytest/unittest put
        the identifying header first but the *diagnosis* -- assertion diff,
        short test summary, the failing line -- last. Keep a head (so the
        identifying fields survive) and a tail (so the reason survives), and
        say out loud that the middle went, so the model can narrow its next
        request instead of blindly re-issuing the same call.
        """
        s = json.dumps(result, default=str)
        if len(s) <= budget:
            return s
        self._clip_drop += len(s) - budget
        self._clip_events += 1
        head = max(200, budget // 4)
        tail = budget - head - 80
        if tail <= 0:
            return s[:budget]
        marker = ("\n...[%d chars elided from the middle; re-read a narrower "
                  "range to see them]...\n" % (len(s) - head - tail))
        return s[:head] + marker + s[-tail:]

    def _pair_for(self, s, budget=None):
        if budget is None:
            budget = self.budget_recent
        op = s.get("op")
        if op == "CALL":
            name = (s.get("args") or {}).get("name", "")
            tool = self._sys2tool(name)
            targs = (s.get("args") or {}).get("args", {}) or {}
            cid = f"c{s['pc']}"
            return [{"role": "assistant", "content": "",
                     "tool_calls": [{"id": cid, "type": "function",
                                     "function": {"name": tool, "arguments": targs}}]},
                    {"role": "tool", "tool_call_id": cid,
                     "content": self._clip_result(s.get("result"), budget)}]
        if op == "PLAN":
            txt = (s.get("args") or {}).get("text", "")
            return [{"role": "assistant", "content": (txt or "")[:600]},
                    {"role": "user",
                     "content": "Call one of the provided tools now."}]
        if op == "RETURN":
            # RETURN closes the process; no further turns
            return []
        return []

    def _sys2tool(self, sysname):
        """Reverse-map from LLMOS syscall name to the tool name the model uses."""
        for tname, sname in self.tool2sys.items():
            if sname == sysname:
                return tname
        # fallback: the model's tool name might already match
        return sysname.replace(".", "_")

    # --- transport: llama-server /v1/chat/completions (no ollama) --------
    @staticmethod
    def _normalize(messages):
        """OpenAI form: assistant tool_call arguments must be JSON strings."""
        out = []
        for m in messages:
            if m.get("role") == "assistant" and m.get("tool_calls"):
                m = dict(m)
                tcs = []
                for tc in m["tool_calls"]:
                    tc = json.loads(json.dumps(tc))  # deep copy
                    fn = tc.get("function", {})
                    if isinstance(fn.get("arguments"), (dict, list)):
                        fn["arguments"] = json.dumps(fn["arguments"])
                    tcs.append(tc)
                m["tool_calls"] = tcs
            out.append(m)
        return out

    def _recover_tool_call(self, content):
        """Recover a tool call from reasoning models that emit bare JSON
        {"name":..,"arguments":..} after </think> instead of the <tool_call> XML
        the server parser expects (e.g. VibeThinker and other non-agentic models)."""
        import re as _re
        txt = content.rsplit("</think>", 1)[-1]
        for h in reversed(list(_re.finditer(r'\{\s*"name"\s*:', txt))):
            start = h.start(); depth = 0
            for i in range(start, len(txt)):
                if txt[i] == "{": depth += 1
                elif txt[i] == "}":
                    depth -= 1
                    if depth == 0:
                        try: obj = json.loads(txt[start:i+1])
                        except Exception: obj = None
                        if isinstance(obj, dict) and "name" in obj and "arguments" in obj:
                            args = obj["arguments"]
                            if not isinstance(args, str): args = json.dumps(args)
                            return [{"id": "call_0", "type": "function",
                                     "function": {"name": obj["name"], "arguments": args}}]
                        break
        return None

    def _chat(self, messages):
        _gemini = "googleapis" in self.host
        _norm = self._normalize(messages)
        # ADAPTIVE ANTI-TRUNCATION (env TRUNC_RETRY, default on). llama.cpp sets
        # finish_reason=="length" when the model hit max_tokens mid-output; for a
        # thinking model at 2048 the tool-call JSON (a patch, or a long check/
        # reproduce script) gets cut off, the call is unparseable, and the turn is
        # wasted. Rather than ask the model to "resend smaller" (which ornith does
        # not reliably do), transparently re-generate with a doubled ceiling up to
        # TRUNC_MAX. Baseline-preserving: only grows on an actual length-stop.
        _retry = os.environ.get("TRUNC_RETRY", "1") == "1"
        _cap = int(os.environ.get("TRUNC_MAX", "8192"))
        _mt = self.num_predict
        _headers = {"Content-Type": "application/json"}
        if _gemini:
            _url = self.host + "/chat/completions"
            _headers["Authorization"] = "Bearer " + os.environ.get("GEMINI_API_KEY", "")
        else:
            _url = self.host + "/v1/chat/completions"
        resp = None; m = {}; _fr = None; _grew = 0
        # RETRIES WAS HARDCODED TO 0 (fixed 2026-08-27, found by preflight.py
        # on its first run). The retry loop below has existed for months and
        # meta reported "retries": 0 unconditionally -- a constant wearing the
        # name of a measurement, the same shape as the patch counter and
        # phase2_reason. If the server starts rate-limiting or flapping, this
        # is the number that says so.
        _retries = 0
        for _grow in range(3):   # initial try + up to 2 doublings
            _payload = {
                "model": self.model, "stream": False,
                "messages": _norm, "tools": self.tools,
                "temperature": self.temperature, "max_tokens": _mt,
            }
            if not _gemini:
                _payload.update({"top_p": 0.95, "top_k": 20, "seed": self.seed})
            body = json.dumps(_payload).encode()
            resp = None
            for _attempt in range(6):
                if _attempt:
                    _retries += 1
                try:
                    req = urllib.request.Request(_url, data=body, headers=_headers)
                    # Never wait past the phase deadline. The floor is 30s
                    # rather than 0 because a request that is cancelled before
                    # the model can answer wastes the turn without ending the
                    # phase; phase_run's own check ends it on the next pass.
                    _to = self.request_timeout
                    _left = _budget_left()
                    if _left is not None:
                        _to = max(30, min(_to, _left))
                    with urllib.request.urlopen(req, timeout=_to) as r:
                        resp = json.loads(r.read())
                    break
                except urllib.error.HTTPError as _e:
                    if _e.code == 429 and _attempt < 5:
                        _nap = min(90, 10 * (_attempt + 1))
                        _left = _budget_left()
                        if _left is not None and _left <= _nap:
                            # Sleeping through the deadline and then asking
                            # again is the worst of both: the phase is over
                            # and we spent its last seconds waiting.
                            raise
                        time.sleep(_nap); continue
                    raise
            _choice = (resp.get("choices") or [{}])[0]
            _fr = _choice.get("finish_reason")
            m = _choice.get("message", {}) or {}
            if _retry and _fr == "length" and _mt < _cap:
                _left = _budget_left()
                if _left is not None and _left <= 0:
                    # A doubled ceiling means a SLOWER regeneration, so the
                    # anti-truncation retry is the last thing that should run
                    # past a deadline. Keep the truncated answer; the caller
                    # is about to end the phase anyway.
                    _fr = "length_deadline"
                    break
                _mt = min(_cap, _mt * 2)
                _grew += 1
                continue
            break
        if not m.get("tool_calls") and m.get("content"):
            _rec = self._recover_tool_call(m["content"])
            if _rec: m["tool_calls"] = _rec
        usage = resp.get("usage") or {}
        timings = resp.get("timings") or {}
        meta = {"prompt_tokens": usage.get("prompt_tokens"),
                "eval_tokens":   usage.get("completion_tokens"),
                "eval_ms": timings.get("predicted_ms", 0),
                "retries": _retries,
                "finish_reason": _fr,
                "trunc_grow": _grew,
                "max_tokens": _mt}
        return m, meta
