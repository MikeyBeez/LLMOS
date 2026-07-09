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
                 model="ornith:35b", host="http://127.0.0.1:11434",
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

    # --- transport -------------------------------------------------------
    def _chat(self, messages):
        body = json.dumps({
            "model": self.model, "stream": False, "keep_alive": self.keep_alive,
            "messages": messages, "tools": self.tools,
            "options": {"temperature": self.temperature, "seed": self.seed,
                        "top_p": 0.95, "top_k": 20,
                        "num_ctx": self.num_ctx, "num_predict": self.num_predict},
        }).encode()
        req = urllib.request.Request(
            self.host + "/api/chat", data=body,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.request_timeout) as r:
            resp = json.loads(r.read())
        m = resp.get("message", {}) or {}
        meta = {"prompt_tokens": resp.get("prompt_eval_count"),
                "eval_tokens":   resp.get("eval_count"),
                "eval_ms": (resp.get("eval_duration") or 0) / 1e6,
                "load_ms": (resp.get("load_duration") or 0) / 1e6,
                "retries": 0}
        return m, meta
