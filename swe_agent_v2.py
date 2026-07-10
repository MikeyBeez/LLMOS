#!/usr/bin/env python3
"""SWE-bench Lite agent v2: two-phase with env-verification gate.

Phase 1 — repo bootstrap:
  Model uses repo_bootstrap_tools until BOTH run_sanity and run_smoke_test
  have passed, then calls declare_env_ready. If either verification fails,
  the model must diagnose and try again. Bootstrap has its own budget so
  a broken repo doesn't eat the fix budget.

Phase 2 — bug fix:
  Only starts if phase 1 declared ready. Model uses swe_fix_tools
  (reproduce -> locate -> read_range -> patch -> run_failing_test) and
  calls submit only after run_failing_test on the FAIL_TO_PASS set passes.

Scoring:
  Apply the model's git diff + the SWE-bench test_patch, run FAIL_TO_PASS.

    PYTHONPATH=~/Code/LLMOS python3 swe_agent_v2.py [N]

Reads ~/swe/instances.json (from swe_lite_select.py), writes
~/swe/results_v2.json and a trace per instance.
"""
import json, os, shutil, subprocess, sys, tempfile, time

sys.path.insert(0, os.path.expanduser("~/Code/LLMOS"))
from tool_call_cpu import ToolCallCPU
from repo_bootstrap_tools import (BOOTSTRAP_TOOLS, BOOTSTRAP_TOOL2SYS,
                                   BOOTSTRAP_SYSTEM_PROMPT,
                                   make_bootstrap_handlers, env_ready)
from swe_fix_tools import (FIX_TOOLS, FIX_TOOL2SYS, FIX_SYSTEM_PROMPT,
                            make_fix_handlers)
import envcheck
from trace_consumers import remedies_for, format_remedy_context, harvest_trace

HOST = "http://127.0.0.1:11434"
MODEL = "ornith:35b"
NUMCTX = 131072
BOOTSTRAP_BUDGET = 50     # bumped for recursive install (each install_package is 1 turn)
FIX_BUDGET      = 40
WORK = os.path.expanduser("~/swe/work")
TRACES = os.path.expanduser("~/swe/traces_v2")


def sh(cmd, cwd=None, timeout=300):
    return subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True,
                          text=True, timeout=timeout)


def clone(inst):
    repo = os.path.join(WORK, inst["instance_id"])
    shutil.rmtree(repo, ignore_errors=True)
    os.makedirs(repo)
    sh("git init -q", cwd=repo)
    sh(f"git remote add origin https://github.com/{inst['repo']}.git", cwd=repo)
    # SWE-bench evaluations use a full clone so setuptools_scm can read tags;
    # our earlier shallow --depth 1 broke astropy's version detection.
    sh(f"git fetch -q origin {inst['base_commit']} --tags", cwd=repo, timeout=600)
    sh("git checkout -q FETCH_HEAD", cwd=repo)
    sh("git config user.email a@b.c; git config user.name a", cwd=repo)
    return repo


def phase_run(cpu, tools, tool2sys, handlers, system_prompt, user_goal,
              budget, gate=None, log=print):
    """Drive one phase: chat, dispatch tool calls, repeat until the model
    calls a RETURN-typed tool (env_ready/submit) or budget is exhausted.

    Returns (terminated_reason, transcript, meta_log)
      terminated_reason: 'declared', 'gate_blocked', 'budget', 'no_call'
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_goal},
    ]
    meta_log = []
    for turn in range(budget):
        try:
            msg, meta = cpu._chat(messages)
        except Exception as e:
            return "cpu_error", messages, meta_log + [{"error": str(e)}]
        meta_log.append({"turn": turn,
                          "prompt_tokens": meta.get("prompt_tokens"),
                          "eval_tokens":   meta.get("eval_tokens")})
        tcs = msg.get("tool_calls") or []
        if not tcs:
            content = (msg.get("content") or msg.get("thinking") or "")[:400]
            messages.append({"role": "assistant", "content": content or "..."})
            messages.append({"role": "user",
                              "content": "Call one of the provided tools now."})
            continue
        tc = tcs[0]
        fn = tc.get("function", {})
        tool = fn.get("name", "")
        args = fn.get("arguments") or {}
        if isinstance(args, str):
            try: args = json.loads(args)
            except Exception: args = {"_raw": args}
        target = tool2sys.get(tool, "")
        log(f"  [{turn:>2}] {tool}({str(args)[:80]}) -> ", end="", flush=True)
        # Terminal tool: check gate then break out.
        if target == "RETURN":
            if gate is not None and not gate():
                # Model tried to declare done but the gate says no. Feed back
                # the reason and continue.
                log("GATE-REJECTED")
                messages.append({"role": "assistant", "content": "",
                                  "tool_calls": [{"id": f"t{turn}", "type": "function",
                                                   "function": {"name": tool,
                                                                 "arguments": args}}]})
                messages.append({"role": "tool", "tool_call_id": f"t{turn}",
                                  "content": json.dumps({
                                      "error": "verification gate not passed; "
                                               "run_sanity and run_smoke_test must "
                                               "both return ok=true first"})})
                continue
            log(f"DECLARED {tool}")
            return "declared", messages + [
                {"role": "assistant", "content": "", "tool_calls": [tc]},
            ], meta_log
        # Normal tool: dispatch.
        h = handlers.get(target)
        if h is None:
            result = {"error": f"unknown tool {tool!r}"}
        else:
            try:
                result = h(None, args)
            except Exception as e:
                result = {"error": f"handler crashed: {type(e).__name__}: {e}"}
        log(str(result)[:120])
        messages.append({"role": "assistant", "content": "",
                         "tool_calls": [{"id": f"t{turn}", "type": "function",
                                          "function": {"name": tool, "arguments": args}}]})
        messages.append({"role": "tool", "tool_call_id": f"t{turn}",
                         "content": json.dumps(result, default=str)[:1800]})
    return "budget", messages, meta_log


def score(inst, repo, env_vars, env_kind="uv"):
    """Apply the model's diff + the test patch, run FAIL_TO_PASS."""
    diff = sh(f"git -C {repo} diff", timeout=60).stdout
    open(os.path.join(TRACES, inst["instance_id"] + ".patch"), "w").write(diff)
    open(os.path.join(repo, "_t.patch"), "w").write(inst["test_patch"])
    ap = sh("git apply _t.patch", cwd=repo)
    if ap.returncode != 0:
        return False, len(diff), "test patch did not apply (agent touched a test file?)"
    files = [l[6:].strip() for l in inst["test_patch"].splitlines()
             if l.startswith("+++ b/") and l.strip().endswith(".py")]
    target = " ".join(files) if files else "."
    names = " or ".join(inst["FAIL_TO_PASS"])
    env = os.environ.copy(); env.update(env_vars or {})
    env_dir = ".condaenv" if env_kind == "conda" else ".venv"
    venv_bin = os.path.join(repo, env_dir, "bin")
    env["PATH"] = venv_bin + ":" + env.get("PATH", "")
    if env_kind == "conda":
        env["CONDA_PREFIX"] = os.path.join(repo, env_dir)
    else:
        env["VIRTUAL_ENV"] = os.path.join(repo, env_dir)
    r = subprocess.run(
        f'{env_dir}/bin/python -m pytest {target} -k "{names}" -p no:cacheprovider -q --no-header',
        shell=True, cwd=repo, capture_output=True, text=True, timeout=600, env=env)
    ok = r.returncode == 0 and "passed" in r.stdout
    tail = (r.stdout[-200:] or r.stderr[-200:]).replace("\n", " ")
    return ok, len(diff), tail


def run_one(inst):
    print(f"\n=== {inst['instance_id']} ({inst['repo']}) ===", flush=True)
    t0 = time.time()
    repo = clone(inst)
    # -------- Phase 1: bootstrap --------
    b_handlers, b_state = make_bootstrap_handlers(repo)
    cpu = ToolCallCPU(tools=BOOTSTRAP_TOOLS, tool2sys=BOOTSTRAP_TOOL2SYS,
                     system_prompt=BOOTSTRAP_SYSTEM_PROMPT, model=MODEL, host=HOST,
                     temperature=1.0, num_predict=2048, num_ctx=NUMCTX,
                     keep_alive="24h")
    goal = (f"Set up the repository at ./ for testing. It is: {inst['repo']}. "
            f"The problem it addresses (for context, do not fix yet):\n\n"
            f"{inst['problem_statement'][:2000]}")
    rems = remedies_for(inst["repo"])
    if rems:
        goal += "\n\n" + format_remedy_context(rems)
        print(f" -- injected {len(rems)} known remedies for {inst['repo']}", flush=True)
    print(" -- phase 1: bootstrap --", flush=True)
    b_reason, b_msgs, b_meta = phase_run(cpu, BOOTSTRAP_TOOLS, BOOTSTRAP_TOOL2SYS,
                                          b_handlers, BOOTSTRAP_SYSTEM_PROMPT,
                                          goal, BOOTSTRAP_BUDGET,
                                          gate=lambda: env_ready(b_state))
    env_ok = env_ready(b_state)
    if not env_ok:
        # Save trace & bail out on this instance without spending fix budget.
        dt = time.time() - t0
        outcome = {"id": inst["instance_id"], "resolved": False,
                    "phase1_reason": b_reason, "env_ok": False,
                    "patch_bytes": 0, "secs": round(dt),
                    "note": "env_setup_failed"}
        print(f" -> env NOT ready ({b_reason})  {dt:.0f}s", flush=True)
        _save_trace(inst, {"phase1": b_msgs, "phase1_meta": b_meta,
                            "state": b_state, "outcome": outcome})
        return outcome
    print(f" -- phase 1 OK: {b_state.get('active_env_kind')}/{b_state.get('python_version')}, "
          f"{len(b_state.get('installed', []))} installs", flush=True)
    # -------- Phase 2: fix --------
    f_handlers, f_state = make_fix_handlers(
        repo, fail_to_pass=inst["FAIL_TO_PASS"], env_vars=b_state["env_vars"],
        env_kind=b_state.get("active_env_kind", "uv"))
    # New CPU instance for phase 2 — separate context, fresh system prompt.
    cpu2 = ToolCallCPU(tools=FIX_TOOLS, tool2sys=FIX_TOOL2SYS,
                       system_prompt=FIX_SYSTEM_PROMPT, model=MODEL, host=HOST,
                       temperature=1.0, num_predict=2048, num_ctx=NUMCTX,
                       keep_alive="24h")
    print(" -- phase 2: fix --", flush=True)
    fix_goal = (f"Problem:\n{inst['problem_statement'][:3000]}\n\n"
                f"You must make these tests pass: {inst['FAIL_TO_PASS']}.")
    f_reason, f_msgs, f_meta = phase_run(cpu2, FIX_TOOLS, FIX_TOOL2SYS,
                                          f_handlers, FIX_SYSTEM_PROMPT,
                                          fix_goal, FIX_BUDGET,
                                          gate=lambda: f_state["fix_verified"])
    # Score with the exact SWE-bench recipe.
    resolved, patch_bytes, tail = score(inst, repo, b_state["env_vars"],
                                        env_kind=b_state.get("active_env_kind", "uv"))
    dt = time.time() - t0
    outcome = {"id": inst["instance_id"], "resolved": bool(resolved),
                "phase1_reason": b_reason, "phase2_reason": f_reason,
                "env_ok": True,
                "env_kind": b_state.get("active_env_kind"),
                "python":   b_state.get("python_version"),
                "installs": b_state.get("installed", []),
                "env_vars": b_state["env_vars"],
                "patch_bytes": patch_bytes, "secs": round(dt),
                "fix_verified_by_model": f_state["fix_verified"],
                "score_tail": tail[:400]}
    print(f" -> resolved={resolved}  patch_bytes={patch_bytes}  {dt:.0f}s | {tail[:120]}",
          flush=True)
    _save_trace(inst, {"phase1": b_msgs, "phase1_meta": b_meta, "state": b_state,
                        "phase2": f_msgs, "phase2_meta": f_meta,
                        "fix_state": f_state, "outcome": outcome})
    shutil.rmtree(repo, ignore_errors=True)
    return outcome


def _save_trace(inst, blob):
    os.makedirs(TRACES, exist_ok=True)
    p = os.path.join(TRACES, inst["instance_id"] + ".trace.json")
    with open(p, "w") as f:
        json.dump(blob, f, indent=1, default=str)
    # Run trace consumers (events, remedy store, training export). Never
    # let a consumer failure damage the run or the already-saved trace.
    try:
        summary = harvest_trace(inst, blob)
        with open(p, "w") as f:
            json.dump(blob, f, indent=1, default=str)
        print(f" -- trace harvest: {summary}", flush=True)
    except Exception as e:
        print(f" -- trace harvest failed (trace still saved): {type(e).__name__}: {e}", flush=True)


def main():
    os.makedirs(WORK, exist_ok=True); os.makedirs(TRACES, exist_ok=True)
    insts = json.load(open(os.path.expanduser("~/swe/instances.json")))
    N = int(sys.argv[1]) if len(sys.argv) > 1 else len(insts)
    insts = insts[:N]
    results = []
    for i, inst in enumerate(insts, 1):
        try:
            r = run_one(inst)
        except Exception as e:
            r = {"id": inst["instance_id"], "resolved": False,
                  "error": f"{type(e).__name__}: {e}"}
        results.append(r)
        json.dump(results, open(os.path.expanduser("~/swe/results_v2.json"), "w"),
                  indent=2)
    resolved = sum(int(r.get("resolved")) for r in results)
    env_ok = sum(int(r.get("env_ok", False)) for r in results)
    print(f"\n=== LLMOS v2 on SWE-bench Lite: {resolved}/{len(results)} resolved, "
          f"{env_ok}/{len(results)} env_ok ===", flush=True)
    for r in results:
        tag = "RESOLVED" if r.get("resolved") else ("env_setup_failed" if not r.get("env_ok")
                                                     else "no")
        print(f"  {r['id']}: {tag}  ({r.get('secs','?')}s)", flush=True)


if __name__ == "__main__":
    main()
