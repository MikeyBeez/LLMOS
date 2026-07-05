#!/usr/bin/env python3
"""LLMOS on SWE-bench (by-hand, no Docker). For each instance: set up the repo at
its base commit in a venv, let ornith drive an LLMOS process using the fs/shell
devices to produce a patch, then score it (apply the test patch, run FAIL_TO_PASS),
then DELETE the repo and keep only the outcome. Streams one instance at a time.

The CPU uses ornith's NATIVE tool-calling (/api/chat with tools). The model
returns structured tool arguments, so shell commands with embedded quotes no
longer have to survive hand-escaped JSON strings -- the transport that broke the
JSON-ISA scaffold. Each tool_call is mapped to the SAME LLMOS Instruction the
kernel already dispatches, so the kernel, devices, trace, and edit contract are
unchanged.

    PYTHONPATH=~/Code/LLMOS python3 swe_agent.py [N]
"""
import json, os, shutil, subprocess, sys, tempfile, time, urllib.request

sys.path.insert(0, os.path.expanduser("~/Code/LLMOS"))
from llmos.store import Store
from llmos.kernel import Kernel
from llmos.cpu import OllamaCPU
from llmos.isa import Instruction, Op

HOST = "http://127.0.0.1:11434"      # ornith is local on pop
MODEL = "ornith:35b"
WORK = os.path.expanduser("~/swe/work")
BUDGET = 30
FORCE_AFTER = 8   # tool calls with no edit -> push the model to stop exploring and edit

# --- native tool schema (Ollama /api/chat tools) ------------------------
TOOLS = [
    {"type": "function", "function": {
        "name": "shell_exec",
        "description": "Run a shell command from the repo root: grep, python3 -c '...', run a script, pytest, etc. Returns exit_code, stdout, stderr.",
        "parameters": {"type": "object",
                       "properties": {"cmd": {"type": "string", "description": "the shell command"}},
                       "required": ["cmd"]}}},
    {"type": "function", "function": {
        "name": "fs_read",
        "description": "Read a file in the repo.",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"}},
                       "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "fs_edit",
        "description": "Replace a VERBATIM snippet with a fix in a file. 'old' must be copied exactly (including indentation) and be unique in the file.",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"},
                                      "old": {"type": "string"},
                                      "new": {"type": "string"}},
                       "required": ["path", "old", "new"]}}},
    {"type": "function", "function": {
        "name": "fs_list",
        "description": "List a directory in the repo.",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"}},
                       "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "finish",
        "description": "Call ONLY after you re-ran your reproduction and confirmed the bug is fixed. Summarize the fix.",
        "parameters": {"type": "object",
                       "properties": {"summary": {"type": "string"}},
                       "required": ["summary"]}}},
]
TOOL2SYS = {"shell_exec": "shell.exec", "fs_read": "fs.read", "fs_edit": "fs.edit", "fs_list": "fs.list"}
SYS2TOOL = {v: k for k, v in TOOL2SYS.items()}


def _short(result, n=1800):
    s = json.dumps(result, default=str)
    if len(s) <= n:
        return s
    return s[:300] + " ...<snip>... " + s[-1200:]


class CodingCPU(OllamaCPU):
    """Overrides step() to drive the model through native tool-calling instead of
    the hand-escaped JSON-ISA. The kernel still receives ordinary Instructions."""

    def __init__(self, repo, problem, f2p, **kw):
        super().__init__(model=MODEL, host=HOST, num_predict=2048, num_ctx=16384, **kw)
        self.repo, self.problem, self.f2p = repo, problem, f2p

    def _system(self):
        return (
            "You are an autonomous software engineer fixing a bug in a Python repository.\n"
            f"The repository is checked out at: {self.repo}\n"
            "Shell commands run from that directory; use paths relative to it.\n\n"
            "Follow this loop and DO NOT skip verification:\n"
            "1. REPRODUCE: use shell_exec to run a tiny `python3 -c \"...\"` that triggers the bug, so you SEE the wrong output.\n"
            "2. LOCATE: grep and fs_read to find the exact buggy lines.\n"
            "3. FIX: fs_edit to replace a small VERBATIM snippet with the correction.\n"
            "4. VERIFY: re-run your reproduction with shell_exec and confirm the behavior is now correct.\n"
            "5. FINISH: only after verification, call finish with a short summary.\n\n"
            "Investigate just enough to find the cause, then EDIT -- do not keep exploring once you have located the bug. "
            "Do NOT modify test files. Make the smallest change that fixes the issue. "
            "Every turn MUST call exactly one tool; never reply with prose alone."
        )

    def _messages(self, pcb):
        msgs = [{"role": "system", "content": self._system()},
                {"role": "user", "content": f"ISSUE:\n{self.problem[:4000]}\n\nBegin by reproducing the bug."}]
        for s in pcb.context:
            op = s.get("op")
            if op == "CALL":
                name = s["args"].get("name", "")
                targs = s["args"].get("args", {}) or {}
                tool = SYS2TOOL.get(name, name.replace(".", "_"))
                cid = f"c{s['pc']}"
                msgs.append({"role": "assistant", "content": "",
                             "tool_calls": [{"id": cid, "type": "function",
                                             "function": {"name": tool, "arguments": targs}}]})
                msgs.append({"role": "tool", "tool_call_id": cid, "content": _short(s["result"])})
            elif op == "PLAN":
                # the model replied with prose and no tool call; feed the reasoning back
                # plus an explicit nudge so the next turn differs (breaks temp-0 loops)
                txt = s["args"].get("text", "") if isinstance(s.get("args"), dict) else ""
                msgs.append({"role": "assistant", "content": txt[:600]})
                msgs.append({"role": "user", "content":
                             "You replied with reasoning but did not call a tool. Call exactly ONE tool now "
                             "(shell_exec, fs_read, fs_edit, or finish) to act on that reasoning. Do not reply with prose."})
            elif op == "RETURN":
                # a trapped early finish (edit contract unmet): surface the note so the model corrects
                note = (s.get("result") or {}).get("note") if isinstance(s.get("result"), dict) else None
                if note:
                    cid = f"c{s['pc']}"
                    summ = s["args"].get("result", "") if isinstance(s.get("args"), dict) else ""
                    msgs.append({"role": "assistant", "content": "",
                                 "tool_calls": [{"id": cid, "type": "function",
                                                 "function": {"name": "finish", "arguments": {"summary": summ}}}]})
                    msgs.append({"role": "tool", "tool_call_id": cid, "content": note})
        # forcing function: if it has explored a while but never edited, push it to edit now
        did_edit = any(s.get("op") == "CALL" and (s.get("args") or {}).get("name") == "fs.edit"
                       for s in pcb.context)
        n_calls = sum(1 for s in pcb.context if s.get("op") == "CALL")
        if not did_edit and n_calls >= FORCE_AFTER:
            msgs.append({"role": "user", "content":
                         f"You have run {n_calls} tool calls and have already reproduced and located the bug, "
                         "but you have not edited any file yet. Stop investigating now. Call fs_edit with the "
                         "smallest change that fixes the bug, then re-run your reproduction to verify it."})
        return msgs

    def step(self, pcb):
        self.last_meta = {}
        try:
            msg, meta = self._chat(self._messages(pcb))
        except Exception as e:
            return Instruction(Op.RETURN, {"result": "CPU device error", "error": str(e)})
        self.last_meta = meta
        tcs = msg.get("tool_calls") or []
        if not tcs:
            txt = (msg.get("content") or msg.get("thinking") or "").strip()
            return Instruction(Op.PLAN, {"text": (txt[:200] or "continue")})
        fn = tcs[0].get("function", {})
        tool = fn.get("name", "")
        args = fn.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {"_raw": args}
        if tool == "finish":
            return Instruction(Op.RETURN, {"result": args.get("summary", "done")})
        sysname = TOOL2SYS.get(tool)
        if not sysname:
            return Instruction(Op.PLAN, {"text": f"unknown tool {tool}"})
        return Instruction(Op.CALL, {"name": sysname, "args": args})

    def _chat(self, messages):
        body = json.dumps({
            "model": self.model, "stream": False, "keep_alive": self.keep_alive,
            "messages": messages, "tools": TOOLS,
            "options": {"temperature": 0, "seed": self.seed,
                        "num_ctx": self.num_ctx, "num_predict": self.num_predict},
        }).encode()
        req = urllib.request.Request(self.host + "/api/chat", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            resp = json.loads(r.read())
        m = resp.get("message", {}) or {}
        meta = {"prompt_tokens": resp.get("prompt_eval_count"),
                "eval_tokens": resp.get("eval_count"),
                "eval_ms": (resp.get("eval_duration") or 0) / 1e6,
                "load_ms": (resp.get("load_duration") or 0) / 1e6,
                "retries": 0}
        return m, meta


def sh(cmd, cwd=None, timeout=300):
    return subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=timeout)


def setup(inst):
    repo = os.path.join(WORK, inst["instance_id"])
    shutil.rmtree(repo, ignore_errors=True)
    os.makedirs(repo)
    sh("git init -q", cwd=repo)
    sh("git remote add origin https://github.com/sympy/sympy.git", cwd=repo)
    sh(f"git fetch -q --depth 1 origin {inst['base_commit']}", cwd=repo, timeout=300)
    sh("git checkout -q FETCH_HEAD", cwd=repo)
    sh("git config user.email a@b.c; git config user.name a", cwd=repo)
    sh("python3 -m venv .venv", cwd=repo)
    sh(".venv/bin/pip install -q mpmath pytest", cwd=repo, timeout=300)
    return repo


def run_agent(inst, repo):
    db = tempfile.mktemp(suffix=".db")
    store = Store(db)
    cpu = CodingCPU(repo, inst["problem_statement"], inst["FAIL_TO_PASS"], log=lambda *a: None)
    pol = {"allowed": [repo], "writable": [repo], "untrusted": []}
    k = Kernel(store, cpu, log=lambda *a: None, fs_policy=pol)
    k.boot()
    caps = {"fs.read", "fs.write", "fs.list", "shell.exec", "dev.calc"}
    pid = k.spawn("fix the bug in this repo", capabilities=caps, budget=BUDGET, contract={"require_edit": True})
    k.run()
    steps = k.procs[pid].pc
    rows = store.trace_read(pid)
    calls = [r["args"].get("name", "?") for r in rows if r["op"] == "CALL"]
    edits = [str(r["result"])[:90] for r in rows if r["op"] == "CALL" and r["args"].get("name") == "fs.edit"]
    store.close()
    if os.path.exists(db):
        os.unlink(db)
    return steps, calls, edits


def score(inst, repo):
    diff = sh(f"git -C {repo} diff", timeout=60).stdout
    open(os.path.join(repo, "_t.patch"), "w").write(inst["test_patch"])
    ap = sh("git apply _t.patch", cwd=repo)
    if ap.returncode != 0:
        return False, len(diff), "test patch did not apply (agent touched test file?)"
    # target only the test files the test patch touches -- avoids collecting the
    # whole tree (which pulls in Python-2 files under bin/ and takes minutes)
    files = [l[6:].strip() for l in inst["test_patch"].splitlines()
             if l.startswith("+++ b/") and l.strip().endswith(".py")]
    target = " ".join(files) if files else "."
    names = " or ".join(inst["FAIL_TO_PASS"])
    r = sh(f'.venv/bin/python -m pytest {target} -k "{names}" -p no:cacheprovider -q --no-header',
           cwd=repo, timeout=600)
    ok = (r.returncode == 0) and ("passed" in r.stdout) and ("failed" not in r.stdout) and ("error" not in r.stdout.lower())
    return ok, len(diff), r.stdout[-200:].replace("\n", " ")


def main():
    os.makedirs(WORK, exist_ok=True)
    insts = json.load(open(os.path.expanduser("~/swe/instances.json")))
    N = int(sys.argv[1]) if len(sys.argv) > 1 else len(insts)
    insts = insts[:N]
    results = []
    for i, inst in enumerate(insts, 1):
        t0 = time.time()
        iid = inst["instance_id"]
        print(f"[{i}/{len(insts)}] {iid}", flush=True)
        try:
            repo = setup(inst)
            print("   setup done, running agent...", flush=True)
            steps, calls, edits = run_agent(inst, repo)
            resolved, difflen, tail = score(inst, repo)
        except Exception as e:
            resolved, difflen, steps, calls, edits, tail = False, 0, 0, [], [], f"ERROR {type(e).__name__}: {e}"
        dt = time.time() - t0
        print(f"   -> resolved={resolved}  steps={steps}  calls={calls}  patch_bytes={difflen}  {dt:.0f}s | {tail}", flush=True)
        for e in edits:
            print(f"      fs.edit -> {e}", flush=True)
        results.append({"id": iid, "resolved": bool(resolved), "steps": steps,
                        "patch_bytes": difflen, "secs": round(dt)})
        shutil.rmtree(os.path.join(WORK, iid), ignore_errors=True)   # delete; keep the outcome
        json.dump(results, open(os.path.expanduser("~/swe/results.json"), "w"), indent=2)
    res = sum(r["resolved"] for r in results)
    print(f"\n=== LLMOS + ornith on SWE-bench Lite (by-hand): {res}/{len(results)} resolved ===", flush=True)
    for r in results:
        print(f"  {r['id']}: {'RESOLVED' if r['resolved'] else 'no'}  ({r['steps']} steps, {r['secs']}s)", flush=True)
    print("### AGENT BATCH DONE ###", flush=True)


if __name__ == "__main__":
    main()
