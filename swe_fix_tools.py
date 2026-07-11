"""SWE-bench fix-loop toolkit — purpose-shaped tools for the bug-fix phase.

Runs AFTER repo_bootstrap_tools has verified the environment. Each tool
mirrors one step of the ideal loop (reproduce -> locate -> read -> patch
-> verify -> submit) and hides the shell/fs primitives so the model isn't
tempted to burn steps on generic exploration.

VERIFICATION MODEL (rewritten 2026-07-10 — "fix this the right way"):
The agent operates in the STRICT SWE-bench setting: it sees the problem
statement only. FAIL_TO_PASS test ids are NOT given to the model and are
NOT runnable here anyway — most of those tests are added by the scoring
test_patch and do not exist in the working tree during the fix phase.
The old run_failing_test targeted them regardless, which either errored
(pylint: 'not found') or vacuously passed, letting the model declare
victory with an EMPTY DIFF (requests-3362, xarray-5131: patch_bytes=0,
fix_verified=True).

The gate is now red -> green on the agent's OWN reproduction:
  1. reproduce(script): a script that exits NONZERO because of the bug.
     The harness registers the last failing script as THE reproduction.
  2. patch: any edit invalidates prior verification.
  3. verify_fix(): reruns the registered reproduction; ok when exit==0.
  4. submit: accepted only when (a) a reproduction failed at least once
     (seen RED), (b) the same script now passes (GREEN), and (c) the
     git diff of non-test source files is non-empty.
"""
import os, re, shlex, shutil, subprocess

from repo_bootstrap_tools import llm_call, _extract_json


def make_fix_handlers(repo_dir, env_vars=None, env_kind="uv", repo=None):
    """Return handlers bound to this repo checkout. env_vars carries anything
    the bootstrap phase set (e.g. DJANGO_SETTINGS_MODULE). env_kind selects
    .venv (uv/pip) or .condaenv (conda)."""
    env_vars = dict(env_vars or {})
    env_dir = ".condaenv" if env_kind == "conda" else ".venv"
    state = {"submitted": False, "fix_verified": False,
             "repro_script": None,      # the registered failing script
             "seen_red": False,         # a reproduction has failed (bug shown)
             "repro_green": False}      # registered script now exits 0

    def _run(cmd, timeout=300):
        env = os.environ.copy()
        env.update(env_vars)
        venv_bin = os.path.join(repo_dir, env_dir, "bin")
        env["PATH"] = venv_bin + ":" + env.get("PATH", "")
        if env_kind == "conda":
            env["CONDA_PREFIX"] = os.path.join(repo_dir, env_dir)
        else:
            env["VIRTUAL_ENV"] = os.path.join(repo_dir, env_dir)
        return subprocess.run(cmd, shell=True, cwd=repo_dir, capture_output=True,
                              text=True, timeout=timeout, env=env)

    def _diff_nonempty():
        r = _run("git diff --stat", timeout=60)
        return bool((r.stdout or "").strip())

    def _gate():
        state["fix_verified"] = (state["seen_red"] and state["repro_green"]
                                 and _diff_nonempty())
        return state["fix_verified"]

    def h_reproduce(pcb, args):
        """Run a reproduction script. A script that exits NONZERO because of
        the bug becomes the registered reproduction (RED)."""
        script = str(args.get("python_script", ""))
        r = _run(f'{env_dir}/bin/python -c {shlex.quote(script)}', timeout=180)
        registered = False
        if r.returncode != 0:
            state["repro_script"] = script
            state["seen_red"] = True
            state["repro_green"] = False
            registered = True
        result = {"exit": r.returncode,
                  "stdout": (r.stdout or "")[-2000:],
                  "stderr": (r.stderr or "")[-2000:],
                  "registered_as_reproduction": registered}
        if registered:
            result["note"] = ("This failing script is now the registered "
                              "reproduction. After you patch, verify_fix will "
                              "rerun EXACTLY this script — it must exit 0.")
        elif not state["seen_red"]:
            result["note"] = ("Script exited 0 — the bug is not demonstrated. "
                              "Write a script that FAILS (nonzero exit, e.g. "
                              "an assert or uncaught exception) because of "
                              "the reported bug.")
        return result

    def h_locate(pcb, args):
        """grep for a pattern, then ask the LLM which hit is most likely the
        actual site to investigate."""
        pat = str(args.get("pattern", ""))
        glob_pat = args.get("file_glob") or ""
        cmd = f'grep -RIn --include="*.py" {shlex.quote(pat)} .'
        if glob_pat:
            cmd = f'grep -RIn --include={shlex.quote(glob_pat)} {shlex.quote(pat)} .'
        r = _run(cmd, timeout=60)
        lines = (r.stdout or "").splitlines()[:40]
        result = {"matches": lines, "match_count": len(lines),
                  "truncated": len(lines) == 40}
        if len(lines) > 1:
            hits_blob = "\n".join(lines[:30])
            ranking = llm_call(
                system=("You rank grep hits by likelihood of being the actual "
                        "bug site vs test file / comment / unrelated match. "
                        "Answer JSON."),
                prompt=(f"grep pattern: {pat!r}\n\nHits:\n{hits_blob}\n\n"
                        'Return JSON: {"top_hit":"path/to/file.py:LINE", '
                        '"reason":"why this one", '
                        '"discard":["path:LINE reasons to skip"]}'))
            parsed = _extract_json(ranking) or {}
            result["ranked"] = parsed
        return result

    def h_read_range(pcb, args):
        path = str(args.get("file", ""))
        start = max(1, int(args.get("start", 1)))
        end = int(args.get("end", start + 40))
        full = os.path.join(repo_dir, path)
        if not os.path.isfile(full):
            return {"error": f"file not found: {path}"}
        try:
            with open(full, encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
        except OSError as e:
            return {"error": str(e)}
        total = len(lines)
        end = min(end, total)
        window = "".join(lines[start-1:end])
        return {"path": path, "start": start, "end": end, "total_lines": total,
                "content": window[:6000]}

    def h_patch(pcb, args):
        """Surgical edit. Any successful patch invalidates prior verification —
        the reproduction must be rerun."""
        path = str(args.get("file", ""))
        old = str(args.get("old_snippet", ""))
        new = str(args.get("new_snippet", ""))
        if re.search(r"(^|/)tests?/|/test_|_test\.py$", path):
            return {"error": "refusing to edit a test file — fix the source, "
                             "not the tests"}
        full = os.path.join(repo_dir, path)
        if not os.path.isfile(full):
            return {"error": f"file not found: {path}"}
        try:
            with open(full, encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except OSError as e:
            return {"error": str(e)}
        if old not in text:
            return {"error": "old_snippet not found in file (must match exactly, "
                             "including indentation and trailing whitespace)"}
        if text.count(old) > 1:
            return {"error": f"old_snippet is ambiguous — matches {text.count(old)} "
                             f"places in the file. Include more surrounding context."}
        new_text = text.replace(old, new, 1)
        with open(full, "w", encoding="utf-8") as f:
            f.write(new_text)
        state["repro_green"] = False
        state["fix_verified"] = False
        return {"edited": path, "old_bytes": len(old), "new_bytes": len(new),
                "delta_bytes": len(new) - len(old),
                "note": "verification invalidated — run verify_fix"}

    def h_verify_fix(pcb, args):
        """Rerun the registered reproduction. GREEN when it exits 0."""
        shutil.rmtree(os.path.join(repo_dir, ".hypothesis"), ignore_errors=True)
        if not state["repro_script"]:
            return {"ok": False,
                    "error": ("no registered reproduction — use reproduce() "
                              "with a script that fails because of the bug "
                              "BEFORE patching")}
        r = _run(f'{env_dir}/bin/python -c '
                 f'{shlex.quote(state["repro_script"])}', timeout=300)
        green = r.returncode == 0
        state["repro_green"] = green
        gate_ok = _gate()
        result = {"ok": green, "exit": r.returncode,
                  "stdout": (r.stdout or "")[-2000:],
                  "stderr": (r.stderr or "")[-1500:],
                  "gate": {"seen_red": state["seen_red"],
                           "repro_green": state["repro_green"],
                           "diff_nonempty": _diff_nonempty(),
                           "fix_verified": gate_ok}}
        if not green:
            result["diagnosis"] = llm_call(
                system=("You explain a failing reproduction for a bug-fix "
                        "agent. Be specific about the traceback and what to "
                        "change next."),
                prompt=(f"Reproduction script:\n{state['repro_script'][:1500]}\n\n"
                        f"stdout:\n{result['stdout']}\nstderr:\n{result['stderr']}\n\n"
                        "In 2-4 sentences: what does the failure show, which "
                        "code is the likely fault, what should the fix look "
                        "like?"))
        return result

    def h_run_tests(pcb, args):
        """Run existing suite test(s) as a regression check (NOT the gate).
        Delegates to the single deterministic test_runner."""
        tid = str(args.get("test_id", ""))
        if not tid:
            return {"error": "test_id required"}
        import test_runner as _tr
        res = _tr.run_tests(repo_dir, env_kind, [tid], env_vars=env_vars,
                            repo=repo, timeout=600)
        return {"ok": res["ok"], "exit": res["exit"],
                "stdout": res["stdout"], "installed": res.get("installed", [])}

    def h_submit(pcb, args):
        """Terminal call. Gate: RED seen, same reproduction now GREEN, and a
        non-empty non-test diff."""
        if not _gate():
            return {"error": ("cannot submit: gate not satisfied — "
                              f"seen_red={state['seen_red']}, "
                              f"repro_green={state['repro_green']}, "
                              f"diff_nonempty={_diff_nonempty()}. "
                              "You need: reproduce() failing (RED), a patch, "
                              "and verify_fix() passing (GREEN).")}
        state["submitted"] = True
        return {"submitted": True, "summary": args.get("summary", "")}

    handlers = {
        "swe.reproduce":   h_reproduce,
        "swe.locate":      h_locate,
        "swe.read_range":  h_read_range,
        "swe.patch":       h_patch,
        "swe.verify_fix":  h_verify_fix,
        "swe.run_tests":   h_run_tests,
        "swe.submit":      h_submit,
    }
    return handlers, state


FIX_TOOLS = [
    {"type": "function", "function": {
        "name": "reproduce",
        "description": (
            "Run a small Python script inside the (verified) venv that demonstrates "
            "the bug by EXITING NONZERO (uncaught exception or failed assert). The "
            "last failing script becomes the registered reproduction that verify_fix "
            "reruns after your patch. Do this FIRST — a fix without a failing "
            "reproduction is guessing, and submit will be rejected without one."),
        "parameters": {"type": "object", "properties": {
            "python_script": {"type": "string",
                              "description": "script that raises/asserts on the buggy "
                                             "behavior, exits 0 once fixed"},
        }, "required": ["python_script"]}}},
    {"type": "function", "function": {
        "name": "locate",
        "description": (
            "grep across the repo for a symbol/message/pattern. Returns file:line "
            "matches (up to 40) plus an LLM ranking of the likeliest bug site."),
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string"},
            "file_glob": {"type": "string",
                          "description": "Optional glob to scope the search, e.g. '*.py'."},
        }, "required": ["pattern"]}}},
    {"type": "function", "function": {
        "name": "read_range",
        "description": (
            "Read lines [start, end] of a specific file. Follow locate — grep gives you "
            "the line number, read_range opens the exact window."),
        "parameters": {"type": "object", "properties": {
            "file":  {"type": "string"},
            "start": {"type": "integer"},
            "end":   {"type": "integer"},
        }, "required": ["file", "start", "end"]}}},
    {"type": "function", "function": {
        "name": "patch",
        "description": (
            "Replace old_snippet with new_snippet in a SOURCE file (test files are "
            "refused). old_snippet must match exactly and be unique. Any patch "
            "invalidates verification — rerun verify_fix afterwards."),
        "parameters": {"type": "object", "properties": {
            "file":         {"type": "string"},
            "old_snippet":  {"type": "string"},
            "new_snippet":  {"type": "string"},
        }, "required": ["file", "old_snippet", "new_snippet"]}}},
    {"type": "function", "function": {
        "name": "verify_fix",
        "description": (
            "Rerun the registered reproduction script. ok=true when it exits 0 "
            "(the bug no longer occurs). submit is only accepted after this "
            "passes on a script that previously FAILED."),
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "run_tests",
        "description": (
            "Run an existing test file or test id from the repo's suite as a "
            "REGRESSION check (did my patch break something nearby?). This is not "
            "the verification gate — verify_fix is."),
        "parameters": {"type": "object", "properties": {
            "test_id": {"type": "string",
                        "description": "e.g. 'path/to/test_file.py::test_name'"},
        }, "required": ["test_id"]}}},
    {"type": "function", "function": {
        "name": "submit",
        "description": (
            "Terminal call. ONLY accepted after: a reproduction failed (RED), you "
            "patched source, and verify_fix passed (GREEN) with a non-empty diff."),
        "parameters": {"type": "object", "properties": {
            "summary": {"type": "string",
                        "description": "1-3 sentence summary of the fix."},
        }, "required": ["summary"]}}},
]


FIX_TOOL2SYS = {
    "reproduce":   "swe.reproduce",
    "locate":      "swe.locate",
    "read_range":  "swe.read_range",
    "patch":       "swe.patch",
    "verify_fix":  "swe.verify_fix",
    "run_tests":   "swe.run_tests",
    "submit":      "RETURN",   # terminal
}


FIX_SYSTEM_PROMPT = (
    "The environment is verified and ready. Fix the bug using this loop:\n"
    "  1. reproduce — write a script that FAILS (nonzero exit: uncaught "
    "exception or assert) because of the reported bug. This registers your "
    "reproduction. If your script exits 0, it does not demonstrate the bug — "
    "rewrite it.\n"
    "  2. locate — grep for the symbol or error message. Get file:line.\n"
    "  3. read_range — open the exact window around the match.\n"
    "  4. patch — surgical replacement in SOURCE files, small and targeted. "
    "Test files are refused.\n"
    "  5. verify_fix — reruns YOUR registered reproduction; it must now exit 0.\n"
    "  6. run_tests — run a nearby existing test file to check you broke nothing.\n"
    "  7. If verify_fix still fails, return to step 3 with the new evidence.\n"
    "  8. submit — only accepted after RED (step 1) then GREEN (step 5) with a "
    "real diff.\n\n"
    "Make the smallest change that fixes the issue. Every turn MUST call "
    "exactly one tool."
)
