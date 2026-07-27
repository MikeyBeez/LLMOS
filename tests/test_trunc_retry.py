"""Unit test for ToolCallCPU._chat adaptive anti-truncation. Mocks the HTTP
layer so it exercises the real _chat loop without a live llama-server."""
import os, sys, json, io
import urllib.request
sys.path.insert(0, os.path.expanduser("~/Code/LLMOS"))
import tool_call_cpu as TC

TOOLS = [{"type": "function", "function": {"name": "check", "parameters": {}}}]
TOOL2SYS = {"check": "swe.check"}

class _Resp:
    def __init__(self, payload): self._b = json.dumps(payload).encode()
    def read(self): return self._b
    def __enter__(self): return self
    def __exit__(self, *a): return False

def _install(seq, record):
    """seq: list of response dicts to return in order (last repeats)."""
    calls = {"i": 0}
    def fake_urlopen(req, timeout=None):
        body = json.loads(req.data.decode())
        record.append(body["max_tokens"])
        i = min(calls["i"], len(seq) - 1)
        calls["i"] += 1
        return _Resp(seq[i])
    urllib.request.urlopen = fake_urlopen

_ORIG = urllib.request.urlopen

def _cpu():
    return TC.ToolCallCPU(TOOLS, TOOL2SYS, host="http://x", num_predict=2048)

def _good(fr="stop"):
    return {"choices": [{"finish_reason": fr, "message": {"role": "assistant",
            "tool_calls": [{"id": "t", "type": "function",
            "function": {"name": "check", "arguments": "{}"}}]}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}}

def _trunc():
    return {"choices": [{"finish_reason": "length", "message": {"role": "assistant",
            "content": '{"snippet": "print(incompl'}}],  # cut off
            "usage": {"prompt_tokens": 10, "completion_tokens": 2048}}

def test_grows_on_length():
    os.environ["TRUNC_RETRY"] = "1"; os.environ.pop("TRUNC_MAX", None)
    rec = []; _install([_trunc(), _good()], rec)
    m, meta = _cpu()._chat([{"role": "user", "content": "hi"}])
    assert m.get("tool_calls"), ("second (roomier) gen should carry the call", m)
    assert meta["trunc_grow"] == 1, meta
    assert meta["max_tokens"] == 4096, meta
    assert rec == [2048, 4096], ("2nd request must double the ceiling", rec)
    print("PASS grows_on_length")

def test_no_grow_when_not_truncated():
    os.environ["TRUNC_RETRY"] = "1"
    rec = []; _install([_good()], rec)
    m, meta = _cpu()._chat([{"role": "user", "content": "hi"}])
    assert meta["trunc_grow"] == 0 and meta["max_tokens"] == 2048, meta
    assert rec == [2048], rec
    print("PASS no_grow_when_not_truncated")

def test_disabled_does_not_grow():
    os.environ["TRUNC_RETRY"] = "0"
    rec = []; _install([_trunc(), _good()], rec)
    m, meta = _cpu()._chat([{"role": "user", "content": "hi"}])
    assert meta["trunc_grow"] == 0 and meta["max_tokens"] == 2048, meta
    assert rec == [2048], ("retry disabled -> exactly one request", rec)
    print("PASS disabled_does_not_grow")

def test_caps_out():
    os.environ["TRUNC_RETRY"] = "1"; os.environ["TRUNC_MAX"] = "8192"
    rec = []; _install([_trunc()], rec)  # always truncates
    m, meta = _cpu()._chat([{"role": "user", "content": "hi"}])
    assert meta["max_tokens"] == 8192, meta
    assert meta["trunc_grow"] == 2, meta          # 2048->4096->8192, 3 tries total
    assert rec == [2048, 4096, 8192], rec
    print("PASS caps_out")

if __name__ == "__main__":
    try:
        test_grows_on_length()
        test_no_grow_when_not_truncated()
        test_disabled_does_not_grow()
        test_caps_out()
        print("ALL TRUNC TESTS PASS")
    finally:
        urllib.request.urlopen = _ORIG
        os.environ.pop("TRUNC_RETRY", None); os.environ.pop("TRUNC_MAX", None)
