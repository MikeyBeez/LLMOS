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
import envcheck   # version checker: pick + uv-provision the right Python per repo

HOST = "http://127.0.0.1:11434"      # ornith is local on pop
MODEL = "ornith:35b"
WORK = os.path.expanduser("~/swe/work")
TRACES = os.path.expanduser("~/swe/traces")   # persisted execution traces (observability; never fed back)
BUDGET = 40
FORCE_AFTER = 8    # tool calls with no edit -> nudge the model to stop exploring and edit
EDIT_DEADLINE = 16   # tool calls with no edit -> HARD stop: restrict the toolset to fs_edit + finish
CTX_HIGH = 48000   # verbatim tail budget in tokens; with num_ctx=64K the model gets real room before any compaction
CTX_CHUNK = 6      # compact in blocks of this many steps, so the digest/cache prefix is stable between jumps

# --- native tool schema (Ollama /api/chat tools) ------------------------
TOOLS = [
    {"type": "function", "function": {
        "name": "shell_exec",
        "description": "Run a shell command. Your shell is ALREADY at the repository root -- never use 'cd' and never use absolute paths; use paths relative to the repo root (e.g. \"grep -n foo sympy/core/expr.py\" or \"python3 -c '...'\"). Returns exit_code, stdout, stderr.",
        "parameters": {"type": "object",
                       "properties": {"cmd": {"type": "string", "description": "the shell command"}},
                       "required": ["cmd"]}}},
    {"type": "function", "function": {
        "name": "fs_read",
        "description": "Read a file in the repo. Big files are truncated in your view, so after `grep -n` gives a line number, pass start and end to read just that window (raw text you can copy verbatim into fs_edit). The result reports the file's total line count.",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"},
                                      "start": {"type": "integer", "description": "first line to read (1-based); optional"},
                                      "end": {"type": "integer", "description": "last line to read; optional"}},
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
EDIT_ONLY_TOOLS = [t for t in TOOLS if t["function"]["name"] in ("fs_edit", "finish")]


def _short(result, n=1800):
    """Truncate a JSON-ified result to ~n chars, honoring n (not a fixed size). Keeps a
    head and a tail so a large read shows the top of the file plus its end; range reads
    (start/end) stay under n and are returned whole."""
    s = json.dumps(result, default=str)
    if len(s) <= n:
        return s
    keep_tail = min(1200, n // 3)
    keep_head = max(300, n - keep_tail - 20)
    return s[:keep_head] + " ...<snip>... " + s[-keep_tail:]


class CodingCPU(OllamaCPU):
    """Overrides step() to drive the model through native tool-calling instead of
    the hand-escaped JSON-ISA. The kernel still receives ordinary Instructions."""

    def __init__(self, repo, problem, **kw):
        super().__init__(model=MODEL, host=HOST, num_predict=2048, num_ctx=65536, **kw)
        # the agent receives ONLY the repo path and the issue text -- never the
        # grading tests, gold patch, or any per-instance hint. keep it that way.
        self.repo, self.problem = repo, problem
        self.meta_log = []   # per-step token counts, to measure context fill/saturation

    def _system(self):
        return (
            "You are an autonomous software engineer fixing a bug in a Python repository.\n"
            "You are ALREADY positioned at the repository root. Never use 'cd' and never use absolute paths -- "
            "shell_exec runs from the repo root, and fs_read/fs_edit paths are relative to it (e.g. sympy/core/expr.py).\n\n"
            "Follow this loop and DO NOT skip verification:\n"
            "1. REPRODUCE: use shell_exec to run a tiny `python3 -c \"...\"` that triggers the bug, so you SEE the wrong output.\n"
            "2. LOCATE: `grep -n` for the symbol to get a line number, then fs_read with start/end to read that exact window. Do NOT re-read the same whole file repeatedly.\n"
            "3. FIX: fs_edit to replace a small VERBATIM snippet with the correction.\n"
            "4. VERIFY: re-run your reproduction with shell_exec and confirm the behavior is now correct.\n"
            "5. FINISH: only after verification, call finish with a short summary.\n\n"
            "Investigate just enough to find the cause, then EDIT -- do not keep exploring once you have located the bug. "
            "Do NOT modify test files. Make the smallest change that fixes the issue. "
            "Every turn MUST call exactly one tool; never reply with prose alone."
        )

    def _pair_for(self, s):
        """Render one context step as its verbatim message(s)."""
        op = s.get("op")
        if op == "CALL":
            name = s["args"].get("name", "")
            targs = s["args"].get("args", {}) or {}
            tool = SYS2TOOL.get(name, name.replace(".", "_"))
            cid = f"c{s['pc']}"
            # file reads get a larger window than shell/grep: the model must SEE the code it edits.
            budget = 6000 if name == "fs.read" else 1800
            return [{"role": "assistant", "content": "",
                     "tool_calls": [{"id": cid, "type": "function",
                                     "function": {"name": tool, "arguments": targs}}]},
                    {"role": "tool", "tool_call_id": cid, "content": _short(s["result"], budget)}]
        if op == "PLAN":
            txt = s["args"].get("text", "") if isinstance(s.get("args"), dict) else ""
            return [{"role": "assistant", "content": txt[:600]},
                    {"role": "user", "content":
                     "You replied with reasoning but did not call a tool. Call exactly ONE tool now "
                     "(shell_exec, fs_read, fs_edit, or finish) to act on that reasoning. Do not reply with prose."}]
        if op == "RETURN":
            note = (s.get("result") or {}).get("note") if isinstance(s.get("result"), dict) else None
            if note:
                cid = f"c{s['pc']}"
                summ = s["args"].get("result", "") if isinstance(s.get("args"), dict) else ""
                return [{"role": "assistant", "content": "",
                         "tool_calls": [{"id": cid, "type": "function",
                                         "function": {"name": "finish", "arguments": {"summary": summ}}}]},
                        {"role": "tool", "tool_call_id": cid, "content": note}]
        return []

    @staticmethod
    def _est(msgs):
        chars = sum(len(m.get("content") or "") + sum(len(json.dumps(t)) for t in (m.get("tool_calls") or []))
                    for m in msgs)
        return chars // 4   # ~4 chars/token

    @staticmethod
    def _digest(old_steps):
        """A faithful, compact record of the compacted-away steps: one line each, keeping WHAT was
        done and the key outcome, dropping bulky payloads (a file's contents are re-readable)."""
        lines = ["PROGRESS SO FAR (older steps compacted; re-read a file if you need its full contents):"]
        for s in old_steps:
            op = s.get("op"); a = s.get("args") or {}; res = s.get("result"); aa = a.get("args") or {}
            nm = a.get("name")
            if op == "CALL" and nm == "shell.exec":
                out = ""
                if isinstance(res, dict):
                    out = (res.get("stdout") or res.get("stderr") or "").strip().replace("\n", " ")
                lines.append("- ran: %s -> %s" % (str(aa.get("cmd", ""))[:80], out[:90]))
            elif op == "CALL" and nm == "fs.read":
                n = res.get("lines") if isinstance(res, dict) else "?"
                lines.append("- read %s (%s lines)" % (aa.get("path", ""), n))
            elif op == "CALL" and nm == "fs.edit":
                if isinstance(res, dict) and ("replaced" in res or "edited" in res) and not res.get("error"):
                    lines.append("- EDITED %s" % aa.get("path", ""))
                else:
                    msg = res.get("error") if isinstance(res, dict) else str(res)
                    lines.append("- edit %s FAILED: %s" % (aa.get("path", ""), str(msg)[:60]))
            elif op == "CALL" and nm == "fs.list":
                lines.append("- listed %s" % aa.get("path", ""))
            elif op == "RETURN":
                lines.append("- tried to finish but was told an edit is still required")
        return "\n".join(lines)

    def _cut(self, steps):
        """How many of the oldest steps to compact into the digest. Quantized to CTX_CHUNK
        so the boundary only jumps when the verbatim tail exceeds CTX_HIGH -- the digest and
        the cache prefix stay put between jumps (append-only growth), and we pay one reprefill
        at a jump rather than churning every step. Always keeps at least the last step verbatim."""
        toks = [self._est(self._pair_for(s)) for s in steps]
        n = len(steps)
        k = 0
        while k < n and sum(toks[k:]) > CTX_HIGH:
            k += CTX_CHUNK
        return min(k, max(0, n - 1))

    def _messages(self, pcb):
        # PINNED: identity + the task. never evicted.
        head = [{"role": "system", "content": self._system()},
                {"role": "user", "content": f"ISSUE:\n{self.problem[:4000]}\n\nBegin by reproducing the bug."}]
        ctx = list(pcb.context)

        # SUPERSEDE: a later read of a path makes earlier reads of it stale -> drop them
        # (this alone removes the re-read waste that filled the window).
        def _rkey(s):
            a = (s["args"].get("args") or {})
            return (a.get("path"), a.get("start"), a.get("end"))
        last_read = {}
        for i, s in enumerate(ctx):
            if s.get("op") == "CALL" and (s.get("args") or {}).get("name") == "fs.read":
                last_read[_rkey(s)] = i
        steps = []
        for i, s in enumerate(ctx):
            if (s.get("op") == "CALL" and (s.get("args") or {}).get("name") == "fs.read"
                    and last_read.get(_rkey(s)) != i):
                continue   # identical earlier read (same path+range) superseded; distinct windows kept
            steps.append(s)

        # WORKING SET: compact the oldest steps into a digest, quantized to CHUNK-sized blocks
        # so the digest (and therefore the cache prefix) stays stable between compactions --
        # append-only cache growth in between, one reprefill at a jump, not churn every step.
        cut = self._cut(steps)
        old, recent = steps[:cut], steps[cut:]

        body = []
        if old:
            body.append({"role": "user", "content": self._digest(old)})
        for s in recent:
            body.extend(self._pair_for(s))
        msgs = head + body

        # forcing function judged on the FULL history, not the window
        did_edit = any(s.get("op") == "CALL" and (s.get("args") or {}).get("name") == "fs.edit" for s in ctx)
        n_calls = sum(1 for s in ctx if s.get("op") == "CALL")
        if not did_edit and n_calls >= FORCE_AFTER:
            msgs.append({"role": "user", "content":
                         f"You have run {n_calls} tool calls and have already reproduced and located the bug, "
                         "but you have not edited any file yet. Stop investigating now. Call fs_edit with the "
                         "smallest change that fixes the bug, then re-run your reproduction to verify it."})
        return msgs

    def step(self, pcb):
        self.last_meta = {}
        # HARD forcing: after the exploration deadline with no edit, restrict the toolset to
        # fs_edit + finish so the model cannot keep exploring -- it must edit (or finish). The
        # restriction lifts once it edits, so it can then re-run its reproduction to verify.
        did_edit = any(s.get("op") == "CALL" and (s.get("args") or {}).get("name") == "fs.edit" for s in pcb.context)
        n_calls = sum(1 for s in pcb.context if s.get("op") == "CALL")
        tools = EDIT_ONLY_TOOLS if (not did_edit and n_calls >= EDIT_DEADLINE) else TOOLS
        try:
            msg, meta = self._chat(self._messages(pcb), tools)
        except Exception as e:
            return Instruction(Op.RETURN, {"result": "CPU device error", "error": str(e)})
        self.last_meta = meta
        self.meta_log.append({"pc": getattr(pcb, "pc", None),
                              "prompt_tokens": meta.get("prompt_tokens"),
                              "eval_tokens": meta.get("eval_tokens")})
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
            # verify gate: don't allow finishing unless a reproduction was re-run
            # AFTER the last edit. general operating discipline; no problem data.
            edits = [i for i, s in enumerate(pcb.context)
                     if s.get("op") == "CALL" and (s.get("args") or {}).get("name") == "fs.edit"]
            shells = [i for i, s in enumerate(pcb.context)
                      if s.get("op") == "CALL" and (s.get("args") or {}).get("name") == "shell.exec"]
            if edits and (not shells or max(shells) < max(edits)):
                return Instruction(Op.PLAN, {"text":
                    "Do not finish yet: you have not re-run your reproduction since your last edit. "
                    "Run your reproduction with shell_exec to confirm the fix changes the behavior."})
            return Instruction(Op.RETURN, {"result": args.get("summary", "done")})
        sysname = TOOL2SYS.get(tool)
        if not sysname:
            return Instruction(Op.PLAN, {"text": f"unknown tool {tool}"})
        return Instruction(Op.CALL, {"name": sysname, "args": args})

    def _chat(self, messages, tools=TOOLS):
        body = json.dumps({
            "model": self.model, "stream": False, "keep_alive": self.keep_alive,
            "messages": messages, "tools": tools,
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
    sh(f"git remote add origin https://github.com/{inst['repo']}.git", cwd=repo)
    sh(f"git fetch -q --depth 1 origin {inst['base_commit']}", cwd=repo, timeout=300)
    sh("git checkout -q FETCH_HEAD", cwd=repo)
    sh("git config user.email a@b.c; git config user.name a", cwd=repo)
    # version checker: read the repo's declared Python, provision it with uv, install deps.
    # setuptools<81 restores distutils for the middle-aged repos on newer interpreters.
    venv_py, ver = envcheck.build_venv(repo, 'mpmath pytest "setuptools<81"')
    print("   env: Python %s (uv)" % ver, flush=True)
    return repo


def run_agent(inst, repo):
    db = tempfile.mktemp(suffix=".db")
    store = Store(db)
    cpu = CodingCPU(repo, inst["problem_statement"], log=lambda *a: None)
    # the agent's shell runs inside the repo's uv venv, so `python`/`python3`/`pytest`
    # use the version the checker chose -- old repos import correctly during reproduction.
    venv_bin = os.path.join(repo, ".venv", "bin")
    pol = {"allowed": [repo], "writable": [repo], "untrusted": [],
           "shell_env": {"PATH": venv_bin + ":" + os.environ.get("PATH", ""),
                         "VIRTUAL_ENV": os.path.join(repo, ".venv")}}
    k = Kernel(store, cpu, log=lambda *a: None, fs_policy=pol)
    k.boot()
    caps = {"fs.read", "fs.write", "fs.list", "shell.exec", "dev.calc"}
    pid = k.spawn("fix the bug in this repo", capabilities=caps, budget=BUDGET, contract={"require_edit": True})
    k.run()
    steps = k.procs[pid].pc
    rows = store.trace_read(pid)
    calls = [r["args"].get("name", "?") for r in rows if r["op"] == "CALL"]
    edits = [str(r["result"])[:90] for r in rows if r["op"] == "CALL" and r["args"].get("name") == "fs.edit"]
    # persist the FULL execution trace: every instruction the CPU emitted and every
    # syscall result. this is the record of HOW the OS drove the model, for later
    # analysis. read-only artifact; it is never fed back into any agent.
    os.makedirs(TRACES, exist_ok=True)
    json.dump({"instance_id": inst["instance_id"], "model": MODEL, "budget": BUDGET,
               "num_ctx": cpu.num_ctx, "per_step_tokens": cpu.meta_log,
               "steps": steps, "calls": calls, "trace": rows},
              open(os.path.join(TRACES, inst["instance_id"] + ".trace.json"), "w"),
              indent=1, default=str)
    store.close()
    if os.path.exists(db):
        os.unlink(db)
    return steps, calls, edits


def score(inst, repo):
    diff = sh(f"git -C {repo} diff", timeout=60).stdout
    os.makedirs(TRACES, exist_ok=True)
    open(os.path.join(TRACES, inst["instance_id"] + ".patch"), "w").write(diff)   # the model's patch
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
    cmd = f'.venv/bin/python -m pytest {target} -k "{names}" -p no:cacheprovider -q --no-header'
    cur = envcheck.pick_python(repo)
    for _ in range(2):
        r = sh(cmd, cwd=repo, timeout=600)
        ran = ("passed" in r.stdout) or ("failed" in r.stdout)
        if ran:
            break
        # no test ran => collection/import failure. if it's a version signature, drop a
        # Python minor, rebuild the venv, and retry once.
        lower = envcheck.downgrade_for((r.stderr or "") + (r.stdout or ""), cur)
        if not lower:
            break
        envcheck.build_venv(repo, 'mpmath pytest "setuptools<81"', py=lower)
        cur = lower
    # pytest returns 0 iff every selected test passed; "passed" guards a zero-test run.
    ok = (r.returncode == 0) and ("passed" in r.stdout)
    tail = ("[py%s] " % cur) + (r.stdout[-200:] or r.stderr[-200:]).replace("\n", " ")
    return ok, len(diff), tail


def main():
    os.makedirs(WORK, exist_ok=True)
    os.makedirs(TRACES, exist_ok=True)
    insts = json.load(open(os.path.expanduser("~/swe/instances.json")))
    N = int(sys.argv[1]) if len(sys.argv) > 1 else len(insts)
    insts = insts[:N]
    rp = os.path.expanduser("~/swe/results.json")
    results = json.load(open(rp)) if os.path.exists(rp) else []   # resume-safe: keep prior outcomes
    done = {r["id"] for r in results}
    for i, inst in enumerate(insts, 1):
        iid = inst["instance_id"]
        if iid in done:
            print(f"[{i}/{len(insts)}] {iid} -- already scored, skipping", flush=True)
            continue
        t0 = time.time()
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
