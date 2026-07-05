#!/usr/bin/env python3
"""LLMOS on SWE-bench (by-hand, no Docker). For each instance: set up the repo at
its base commit in a venv, let ornith drive an LLMOS process using the fs/shell
devices to produce a patch, then score it (apply the test patch, run FAIL_TO_PASS),
then DELETE the repo and keep only the outcome. Streams one instance at a time.

    PYTHONPATH=~/Code/LLMOS python3 swe_agent.py
"""
import json, os, shutil, subprocess, sys, tempfile, time

sys.path.insert(0, os.path.expanduser("~/Code/LLMOS"))
from llmos.store import Store
from llmos.kernel import Kernel
from llmos.cpu import OllamaCPU

HOST = "http://127.0.0.1:11434"      # ornith is local on pop
MODEL = "ornith:35b"
WORK = os.path.expanduser("~/swe/work")
BUDGET = 14


class CodingCPU(OllamaCPU):
    def __init__(self, repo, problem, f2p, **kw):
        super().__init__(model=MODEL, host=HOST, num_predict=1024, num_ctx=32768, **kw)
        self.repo, self.problem, self.f2p = repo, problem, f2p

    def _build_prompt(self, pcb, correction=None):
        hist = "\n".join(
            f"  {s['pc']}: {s['op']} {json.dumps(s['args'])[:100]} -> {json.dumps(s['result'])[:180]}"
            for s in pcb.context) or "  (none yet)"
        head = ""
        if correction:
            head = f"Your previous reply could not be decoded: {correction}. End with ONE JSON instruction.\n\n"
        return head + (
            "You are an automated bug-fixing agent in a Python repository.\n"
            f"REPO (your working directory): {self.repo}\n"
            f"ISSUE:\n{self.problem[:1400]}\n\n"
            f"You must make this failing test pass: {self.f2p}\n\n"
            "Emit exactly ONE instruction as a single JSON object each step:\n"
            '  {"op":"CALL","args":{"name":"shell.exec","args":{"cmd":"grep -rn TERM sympy/ | head"}}}\n'
            '  {"op":"CALL","args":{"name":"fs.read","args":{"path":"sympy/x/y.py"}}}\n'
            '  {"op":"CALL","args":{"name":"fs.edit","args":{"path":"sympy/x/y.py","old":"<exact snippet copied verbatim>","new":"<replacement>"}}}\n'
            '  {"op":"RETURN","args":{"result":"done"}}\n'
            "Use shell.exec (grep/find) and fs.read to locate the exact buggy lines, then fs.edit to replace a SMALL unique\n"
            "snippet (copy the old text VERBATIM incl indentation) with the fix. Do NOT rewrite whole files. Then RETURN.\n\n"
            f"STEPS SO FAR:\n{hist}\n\nNext instruction (one JSON object):"
        )


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
    pid = k.spawn("fix the bug in this repo", capabilities=caps, budget=BUDGET, contract={})
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
    names = " or ".join(inst["FAIL_TO_PASS"])
    r = sh(f'.venv/bin/python -m pytest -k "{names}" -q', cwd=repo, timeout=600)
    return (r.returncode == 0), len(diff), r.stdout[-200:].replace("\n", " ")


def main():
    os.makedirs(WORK, exist_ok=True)
    insts = json.load(open(os.path.expanduser("~/swe/instances.json")))
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
