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
import json, urllib.request

from llmos.cpu import OllamaCPU
from llmos.isa import Instruction, Op


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
                 request_timeout=600):
        super().__init__(model=model, host=host, seed=seed, log=log,
                         keep_alive=keep_alive,
                         num_predict=num_predict, num_ctx=num_ctx)
        self.tools = tools
        self.tool2sys = tool2sys
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.request_timeout = request_timeout

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
        if self.system_prompt:
            msgs.append({"role": "system", "content": self.system_prompt})
        msgs.append({"role": "user", "content": pcb.goal})
        for s in pcb.context:
            msgs.extend(self._pair_for(s))
        return msgs

    def _pair_for(self, s):
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
                     "content": json.dumps(s.get("result"), default=str)[:1800]}]
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

    def _chat(self, messages):
        body = json.dumps({
            "model": self.model, "stream": False,
            "messages": self._normalize(messages), "tools": self.tools,
            "temperature": self.temperature, "top_p": 0.95, "top_k": 20,
            "seed": self.seed, "max_tokens": self.num_predict,
        }).encode()
        req = urllib.request.Request(
            self.host + "/v1/chat/completions", data=body,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.request_timeout) as r:
            resp = json.loads(r.read())
        m = (resp.get("choices") or [{}])[0].get("message", {}) or {}
        usage = resp.get("usage") or {}
        timings = resp.get("timings") or {}
        meta = {"prompt_tokens": usage.get("prompt_tokens"),
                "eval_tokens":   usage.get("completion_tokens"),
                "eval_ms": timings.get("predicted_ms", 0),
                "load_ms": 0,
                "retries": 0}
        return m, meta
