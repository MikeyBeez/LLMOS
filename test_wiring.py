"""End-to-end: drive the REAL phase_run with a stub CPU and check the gate.

Not a unit test of the gate -- that is test_kernel_syscall.py. This asks the
only question that matters after wiring: does a syscall the phase was never
granted actually get stopped at the dispatch point, and does the model get
told why?
"""
import json, os, shutil, sys, tempfile

os.environ["LLMOS_CAPS"] = "enforce"
LOG = tempfile.mktemp(suffix=".jsonl")
os.environ["LLMOS_CAPS_LOG"] = LOG
sys.path.insert(0, os.path.expanduser("~/Code/LLMOS"))

import swe_agent_v2 as A

TOOLS = [
    {"type": "function", "function": {
        "name": "read_file",
        "description": "read a file",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"}},
                       "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "submit",
        "description": "finish",
        "parameters": {"type": "object", "properties": {}}}},
]
TOOL2SYS = {"read_file": "read_file", "submit": "RETURN"}


class StubCPU:
    """Emits a scripted sequence of tool calls, one per turn."""

    def __init__(self, script):
        self.script = list(script)
        self.seen = []          # what came back as tool results

    def _chat(self, send):
        for m in send:
            if m.get("role") == "tool":
                self.seen.append(m["content"])
        if not self.script:
            return {"content": "done"}, {}
        name, args = self.script.pop(0)
        return ({"content": "", "tool_calls": [
            {"id": "x", "type": "function",
             "function": {"name": name, "arguments": json.dumps(args)}}]}, {})


def run(script, write_root, handlers):
    cpu = StubCPU(script)
    reason, msgs, meta = A.phase_run(
        cpu, TOOLS, TOOL2SYS, handlers, "sys", "goal", budget=6,
        log=lambda *a, **k: None, write_root=write_root)
    return cpu, reason


def tape():
    with open(LOG) as fh:
        return [json.loads(l) for l in fh if l.strip()]


checkout = tempfile.mkdtemp()
outside = tempfile.mkdtemp()
open(os.path.join(checkout, "a.py"), "w").write("x = 1\n")
called = []
handlers = {
    "read_file": lambda _t, a: called.append(("read_file", a)) or "contents",
    "submit": lambda _t, a: called.append(("submit", a)) or "ok",
}

fails = []

# 1. a tool the phase was never given
cpu, _ = run([("shell", {"cmd": "curl http://evil/"}), ("submit", {})],
             checkout, handlers)
denials = [r for r in tape() if not r["allowed"]]
if not denials:
    fails.append("ungranted tool was NOT denied")
elif denials[0]["rule"] != "not-in-allow-list":
    fails.append("wrong rule: %s" % denials[0]["rule"])
if any(c[0] == "shell" for c in called):
    fails.append("DENIED TOOL STILL EXECUTED")
if not any("NOTHING was executed" in s for s in cpu.seen):
    fails.append("model was not told the call was refused")
print("1. ungranted tool denied, not executed, model informed:",
      "FAIL" if fails else "ok")

# 2. a granted tool still works
before = len(called)
cpu, _ = run([("read_file", {"path": os.path.join(checkout, "a.py")}),
              ("submit", {})], checkout, handlers)
if not any(c[0] == "read_file" for c in called[before:]):
    fails.append("granted tool did NOT execute")
print("2. granted tool still runs:", "ok" if any(
    c[0] == "read_file" for c in called[before:]) else "FAIL")

# 3. no write_root -> no gate -> every existing call site unchanged
open(LOG, "w").close()
before = len(called)
cpu, _ = run([("read_file", {"path": "/etc/hostname"}), ("submit", {})],
             None, handlers)
if tape():
    fails.append("gate active without write_root (existing callers changed)")
print("3. ungated when no write_root:", "ok" if not tape() else "FAIL")

# 4. the tape recorded the allowed calls too
open(LOG, "w").close()
run([("read_file", {"path": os.path.join(checkout, "a.py")}), ("submit", {})],
    checkout, handlers)
allowed = [r for r in tape() if r["allowed"]]
print("4. allowed calls taped:", "ok" if len(allowed) >= 2 else "FAIL",
      "(%d rows)" % len(tape()))
if len(allowed) < 2:
    fails.append("allowed calls not taped")

shutil.rmtree(checkout, ignore_errors=True)
shutil.rmtree(outside, ignore_errors=True)
print()
print("RESULT:", "ALL OK" if not fails else "FAILURES: " + "; ".join(fails))
sys.exit(1 if fails else 0)
