"""LlamaCppCPU — swap ollama's /api/generate for llama.cpp's /completion.

Two wins over OllamaCPU:
  1. GBNF grammar forces the model to close its reply with a syntactically valid
     Instruction. Kills the two biggest failure modes at the token layer:
        - RETURN with empty args (grammar requires args.result to be non-null)
        - reasoning-only responses with no JSON (grammar requires an instruction)
  2. cache_prompt=true — llama.cpp keeps the KV cache from the prior /completion
     call and only re-prefills the tokens the client appended. Since LLMOS
     traces grow monotonically (append-only context), this drops per-step
     latency from O(full_context) to O(new_tokens_since_last_step).

Optional: pass draft_model_path at server-startup for speculative decoding
with a small ornith:9b draft — typically 1.5-3x wall-clock speedup on ornith:35b.

Compatible with the existing Kernel: subclasses OllamaCPU so decode / retry /
last_meta behavior is inherited unchanged. Only _generate() is overridden.
"""
import json, os, urllib.request

from llmos.cpu import OllamaCPU
from llmos.isa import Instruction


DEFAULT_GRAMMAR_PATH = os.path.join(os.path.dirname(__file__), "isa.gbnf")


class LlamaCppCPU(OllamaCPU):
    """A CPU that talks directly to llama-server's /completion endpoint.

    The server is started separately (see run_llamacpp.sh). This class assumes
    the server is up and reachable at `host`. Model identity is set at
    server-startup; the `model` param here is only kept for logging parity
    with OllamaCPU.
    """

    def __init__(self, model="ornith:35b", host="http://127.0.0.1:8080",
                 grammar_path=None, cache_prompt=True,
                 num_predict=4096, num_ctx=131072,
                 seed=0, max_retries=1, log=None, keep_alive="24h"):
        super().__init__(model=model, host=host, seed=seed, max_retries=max_retries,
                         log=log, keep_alive=keep_alive,
                         num_predict=num_predict, num_ctx=num_ctx)
        self.grammar_path = grammar_path or DEFAULT_GRAMMAR_PATH
        self.cache_prompt = cache_prompt
        self._grammar_cache = None

    def _grammar(self):
        if self._grammar_cache is None and self.grammar_path and os.path.exists(self.grammar_path):
            with open(self.grammar_path) as f:
                self._grammar_cache = f.read()
        return self._grammar_cache or ""

    def _generate(self, pcb, correction=None):
        """POST to llama.cpp's /completion. Returns (raw_text, meta) matching
        OllamaCPU's contract so the base class's decode/retry path is reused."""
        prompt = self._build_prompt(pcb, correction)
        body = {
            "prompt":        prompt,
            "n_predict":     self.num_predict,
            "temperature":   0.0,
            "seed":          self.seed,
            "cache_prompt":  self.cache_prompt,
            "stream":        False,
            # llama.cpp will only sample tokens allowed by the grammar. This is
            # where the "model literally cannot emit an empty RETURN" property
            # comes from. grammar_lazy=true delays enforcement until the model
            # emits the first '{' — ornith is a thinking-mode model that reasons
            # in prose for hundreds of tokens; without lazy mode, the grammar's
            # `reasoning ::= [^{]*` still requires the grammar-machine to walk
            # every non-'{' token, which slows sampling. Lazy mode ignores the
            # grammar entirely until the trigger, then applies it strictly.
            "grammar":       self._grammar(),
            "grammar_lazy":  True,
            # llama.cpp trigger type enum: 0=WORD, 1=PATTERN, 2=PATTERN_FULL,
            # 3=TOKEN. Type=1 (regex pattern) activates the grammar as soon as
            # the model's output matches "\{" — i.e. the first opening brace.
            "grammar_triggers": [{"type": 1, "value": "\\{"}],
        }
        try:
            data = json.dumps(body).encode()
            req = urllib.request.Request(
                self.host.rstrip("/") + "/completion", data=data,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=300) as r:
                resp = json.loads(r.read())
        except Exception as e:
            # device error: fail closed with a terminating RETURN so the kernel
            # doesn't spin. Matches OllamaCPU's exception path.
            return (
                json.dumps({"op": "RETURN", "args": {"result": "CPU device error",
                                                     "error": str(e)}}),
                {},
            )

        # llama-server's /completion response shape:
        # {"content": "...", "tokens_predicted": N, "tokens_evaluated": M,
        #  "prompt_n": P, "timings": {"predicted_ms":..,"prompt_ms":..}, ...}
        text = resp.get("content", "") or ""
        timings = resp.get("timings", {}) or {}
        meta = {
            "prompt_tokens": resp.get("tokens_evaluated") or resp.get("prompt_n"),
            "eval_tokens":   resp.get("tokens_predicted"),
            "eval_ms":       timings.get("predicted_ms"),
            "load_ms":       timings.get("prompt_ms"),
            "cache_n":       resp.get("cache_n"),          # tokens reused from prompt cache
        }
        return text, meta
