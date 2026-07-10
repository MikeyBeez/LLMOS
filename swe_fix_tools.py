"""SWE-bench fix-loop toolkit — purpose-shaped tools for the bug-fix phase.

Runs AFTER repo_bootstrap_tools has verified the environment. Each tool
mirrors one step of the ideal loop (reproduce -> locate -> read -> patch
-> verify -> submit) and hides the shell/fs primitives so the model isn't
tempted to burn steps on generic exploration.

The astropy pilot case burned 40 steps in `shell.exec` + `fs.read` without
ever emitting an edit; a `locate(pattern) -> read_range(file, start, end)
-> patch(file, old, new)` pipeline makes the ideal move obvious.
"""
import os, re, shlex, subprocess

from repo_bootstrap_tools import llm_call, _extract_json


def make_fix_handlers(repo_dir, fail_to_pass, env_vars=None, env_kind="uv"):
    """Return handlers bound to this repo checkout + the FAIL_TO_PASS test
    id(s) we're supposed to make pass. env_vars carries anything the
    bootstrap phase set (e.g. DJANGO_SETTINGS_MODULE). env_kind selects
    which env directory to use (.venv for uv/pip, .condaenv for conda)."""
    env_vars = dict(env_vars or {})
    env_dir = ".condaenv" if env_kind == "conda" else ".venv"
    state = {"submitted": False, "fix_verified": False,
             "fail_to_pass": list(fail_to_pass)}

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

    def h_reproduce(pcb, args):
        """Run a small Python script inside the venv to observe the bug."""
        script = str(args.get("python_script", ""))
        r = _run(f'{env_dir}/bin/python -c {shlex.quote(script)}', timeout=180)
        return {"exit": r.returncode,
                "stdout": (r.stdout or "")[-2000:],
                "stderr": (r.stderr or "")[-2000:]}

    def h_locate(pcb, args):
        """grep for a pattern, then ask the LLM which hit is most likely the
        actual site to investigate. Deterministic search + intelligent
        ranking = the caller gets 'go read line 227 of foo.py' instead of
        40 mystery hits."""
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
            # LLM ranking: the model has repo context from problem_statement,
            # so it can name the hit that looks like it MATTERS for the bug.
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
        """Read lines [start, end] of a specific file. Encodes the
        'locate first, then open a specific window' pattern."""
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
        """Surgical fs.edit: replace old_snippet with new_snippet. old_snippet
        must match exactly (whitespace, indentation) and be unique in the file."""
        path = str(args.get("file", ""))
        old = str(args.get("old_snippet", ""))
        new = str(args.get("new_snippet", ""))
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
        return {"edited": path, "old_bytes": len(old), "new_bytes": len(new),
                "delta_bytes": len(new) - len(old)}

    def h_run_failing_test(pcb, args):
        import shutil as _sh
        _sh.rmtree(os.path.join(repo_dir, '.hypothesis'), ignore_errors=True)
        """Run the specific FAIL_TO_PASS test(s) we're supposed to make pass.
        The model can also pass an override test_id (useful during
        iteration), but the harness will always require the real
        FAIL_TO_PASS set to pass at submit time."""
        override = args.get("test_id")
        ids = [override] if override else state["fail_to_pass"]
        cmd = (f'{env_dir}/bin/python -m pytest -q -p no:cacheprovider ' +
               " ".join(f'"{tid}"' for tid in ids))
        r = _run(cmd, timeout=600)
        ok = r.returncode == 0 and "passed" in (r.stdout or "")
        if ok and not override:
            state["fix_verified"] = True
        result = {"ok": ok, "exit": r.returncode,
                  "stdout": (r.stdout or "")[-2500:],
                  "stderr": (r.stderr or "")[-1000:],
                  "tested_ids": ids}
        if not ok:
            # Explain the failure so the model doesn't have to parse
            # thousands of chars of pytest output itself.
            result["diagnosis"] = llm_call(
                system=("You explain pytest failures for a bug-fix agent. "
                        "Be specific about what the assertion / traceback "
                        "shows and what to change next."),
                prompt=(f"Target test(s): {ids}\n\n"
                        f"pytest output:\n{result['stdout']}\n"
                        f"{result['stderr']}\n\n"
                        "In 2-4 sentences: what is the assertion or error, "
                        "which line of code is the likely fault, and what "
                        "should the fix look like?"))
        return result

    def h_submit(pcb, args):
        """Terminal call. Rejected if run_failing_test hasn't succeeded on the
        real FAIL_TO_PASS set since the last patch."""
        if not state["fix_verified"]:
            return {"error": "cannot submit: run_failing_test on the FAIL_TO_PASS "
                             "set has not returned ok since the last patch. Run it "
                             "first and confirm the target tests pass."}
        state["submitted"] = True
        return {"submitted": True, "summary": args.get("summary", "")}

    handlers = {
        "swe.reproduce":         h_reproduce,
        "swe.locate":            h_locate,
        "swe.read_range":        h_read_range,
        "swe.patch":             h_patch,
        "swe.run_failing_test":  h_run_failing_test,
        "swe.submit":            h_submit,
    }
    return handlers, state


FIX_TOOLS = [
    {"type": "function", "function": {
        "name": "reproduce",
        "description": (
            "Run a small Python script inside the (verified) venv to observe the bug's "
            "current behavior. Returns stdout/stderr/exit. Use this before editing so "
            "you SEE the wrong output — a fix without a reproduction is guessing."),
        "parameters": {"type": "object", "properties": {
            "python_script": {"type": "string",
                              "description": "e.g. 'from foo import bar; print(bar(1))'"},
        }, "required": ["python_script"]}}},
    {"type": "function", "function": {
        "name": "locate",
        "description": (
            "grep across the repo for a symbol/message/pattern. Returns file:line "
            "matches (up to 40). Use this to find where a symbol is defined before "
            "reading files — do NOT use for generic exploration."),
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
            "Replace old_snippet with new_snippet in file. old_snippet MUST match exactly "
            "(whitespace, indentation) and be unique in the file — include surrounding "
            "context to disambiguate. Small targeted edits."),
        "parameters": {"type": "object", "properties": {
            "file":         {"type": "string"},
            "old_snippet":  {"type": "string"},
            "new_snippet":  {"type": "string"},
        }, "required": ["file", "old_snippet", "new_snippet"]}}},
    {"type": "function", "function": {
        "name": "run_failing_test",
        "description": (
            "Run the FAIL_TO_PASS test(s) that must pass for this task. If test_id is "
            "given, run just that one (useful during iteration). Otherwise runs the "
            "whole target set. submit will only be accepted after this returns ok on "
            "the full target set."),
        "parameters": {"type": "object", "properties": {
            "test_id": {"type": "string",
                        "description": "Optional single test id to run during iteration."},
        }}}},
    {"type": "function", "function": {
        "name": "submit",
        "description": (
            "Terminal call. ONLY call after run_failing_test on the target set has "
            "returned ok since your last patch."),
        "parameters": {"type": "object", "properties": {
            "summary": {"type": "string",
                        "description": "1-3 sentence summary of the fix."},
        }, "required": ["summary"]}}},
]


FIX_TOOL2SYS = {
    "reproduce":         "swe.reproduce",
    "locate":            "swe.locate",
    "read_range":        "swe.read_range",
    "patch":             "swe.patch",
    "run_failing_test":  "swe.run_failing_test",
    "submit":            "RETURN",   # terminal
}


FIX_SYSTEM_PROMPT = (
    "The environment is verified and ready. Fix the bug using this loop:\n"
    "  1. reproduce — run a small Python script that shows the current wrong output.\n"
    "  2. locate — grep for the symbol or error message. Get file:line.\n"
    "  3. read_range — open the exact window around the match.\n"
    "  4. patch — surgical replacement, small and targeted.\n"
    "  5. run_failing_test — confirm the FAIL_TO_PASS test now passes.\n"
    "  6. If it still fails, go back to step 3 with the new evidence.\n"
    "  7. submit — only after run_failing_test returned ok.\n\n"
    "Do NOT modify test files. Make the smallest change that fixes the issue. "
    "Every turn MUST call exactly one tool."
)
