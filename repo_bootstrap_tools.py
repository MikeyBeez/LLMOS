"""Generic Python repo bootstrap toolkit + a verification gate.

Purpose: give the model tools that mirror what a human developer does when
handed an unfamiliar Python project — read the docs, inspect the config,
pick a Python version, provision a venv, and VERIFY the env works before
touching anything else. Reusable outside SWE-bench: any time we need to
set up an arbitrary Python project.

Design principle (from Mikey): 'we should verify the env before trying to
move on'. The bootstrap agent doesn't get to declare success. The gate
requires both an import-sanity check AND a smoke test that runs at least
one test on the unmodified code. If either fails, the agent must diagnose
(read errors, read docs, web-search) and retry.

The tool schema is OpenAI-compatible (routes through ornith's qwen3_xml
tool-call parser via ollama /api/chat). Each tool_call maps to an LLMOS
syscall; the kernel dispatches and returns structured results.
"""
import glob, json, os, re, subprocess, urllib.parse, urllib.request


UV = os.path.expanduser("~/.local/bin/uv")


# ---------- OpenAI-format tool schemas ---------------------------------
BOOTSTRAP_TOOLS = [
    {"type": "function", "function": {
        "name": "read_repo_docs",
        "description": (
            "Read README/INSTALL/CONTRIBUTING and docs/**/install* from the repo root. "
            "Returns concatenated content (truncated to a reasonable size) of every "
            "matching file. Use this FIRST to learn the project's declared install "
            "steps, supported Python versions, and required system deps."),
        "parameters": {"type": "object", "properties": {
            "max_chars_per_file": {"type": "integer", "default": 8000,
                                    "description": "Truncate each file to this size."},
        }}}},
    {"type": "function", "function": {
        "name": "inspect_repo_config",
        "description": (
            "Parse setup.py / setup.cfg / pyproject.toml / requirements*.txt / tox.ini "
            "/ .python-version. Returns { declared_python_versions, install_deps, "
            "test_extras_names, dev_extras_names, build_system, has_editable_install }. "
            "Use this to decide which Python version and extras to provision."),
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "web_search",
        "description": (
            "Search the web for setup information (e.g. 'astropy install from source', "
            "'matplotlib python 3.11 compatibility'). Returns up to N result titles + "
            "snippets. Use when config files don't tell you enough."),
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"},
            "n": {"type": "integer", "default": 5},
        }, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "provision_env",
        "description": (
            "Create .venv at the repo root using uv with the given Python version and "
            "install the repo (editable) plus any extras. Returns install log + exit "
            "code. If it fails, read the log and try again with a different Python or "
            "different extras before giving up."),
        "parameters": {"type": "object", "properties": {
            "python_version": {"type": "string",
                                "description": "e.g. '3.9', '3.11'"},
            "extras": {"type": "array", "items": {"type": "string"},
                        "description": "Extras to install with the repo, e.g. ['test','docs']. "
                                       "Empty means bare `pip install -e .`."},
            "extra_packages": {"type": "array", "items": {"type": "string"},
                                "description": "Additional loose packages beyond the repo's own extras "
                                                "(e.g. ['hypothesis','pytest-xdist','numpy<2'])."},
            "env_vars": {"type": "object",
                          "description": "Env vars to set for installation and later tests, e.g. "
                                         "{'DJANGO_SETTINGS_MODULE': 'tests.settings'}."},
        }, "required": ["python_version"]}}},
    {"type": "function", "function": {
        "name": "run_sanity",
        "description": (
            "Try to import the package inside the provisioned venv. Returns stdout/stderr "
            "of `python -c '<import_stmt>'`. Use this to verify the install worked "
            "BEFORE trusting the env."),
        "parameters": {"type": "object", "properties": {
            "import_stmt": {"type": "string",
                             "description": "The import to run, e.g. 'import astropy; print(astropy.__version__)'"},
        }, "required": ["import_stmt"]}}},
    {"type": "function", "function": {
        "name": "run_smoke_test",
        "description": (
            "Run `pytest --collect-only -q` then run ONE specific test id. This is the "
            "second half of env verification — if pytest can't even collect, or a test "
            "that should pass fails, the env is broken. Returns collection output and "
            "the one test's outcome."),
        "parameters": {"type": "object", "properties": {
            "test_id": {"type": "string",
                         "description": "e.g. 'tests/test_basic.py::test_import' — a single specific test."},
            "extra_pytest_args": {"type": "string", "default": "",
                                    "description": "e.g. '-p no:cacheprovider'"},
        }, "required": ["test_id"]}}},
    {"type": "function", "function": {
        "name": "declare_env_ready",
        "description": (
            "Only call this AFTER both run_sanity and run_smoke_test have succeeded. "
            "Marks the environment as verified and hands off to the fix phase. If "
            "either verification hasn't passed yet, the harness will reject this call."),
        "parameters": {"type": "object", "properties": {
            "summary": {"type": "string",
                         "description": "Short summary of what setup steps worked, e.g. "
                                        "'Python 3.11 + .[test] + DJANGO_SETTINGS_MODULE=tests.settings'."},
        }, "required": ["summary"]}}},
]


# Map each tool name -> LLMOS syscall the kernel should dispatch to.
BOOTSTRAP_TOOL2SYS = {
    "read_repo_docs":     "repo.read_docs",
    "inspect_repo_config": "repo.inspect_config",
    "web_search":         "web.search",
    "provision_env":      "repo.provision_env",
    "run_sanity":         "repo.run_sanity",
    "run_smoke_test":     "repo.run_smoke_test",
    "declare_env_ready":  "RETURN",   # terminal — routed to Op.RETURN
}


# ---------- Handler implementations ------------------------------------
# These run when the kernel dispatches the tool. They live outside the
# SyscallTable so they can be attached per-run (each agent instance has
# its own repo path, env vars, etc.).

def make_bootstrap_handlers(repo_dir, base_env_vars=None):
    """Return a dict {syscall_name: handler(pcb, args) -> result} bound to
    this specific repo checkout. The handlers write to the venv at .venv/
    and keep an internal record of what was tried."""
    state = {
        "sanity_ok":       False,
        "smoke_ok":        False,
        "env_vars":        dict(base_env_vars or {}),
        "last_python":     None,
        "last_extras":     [],
    }

    def _run_in_venv(cmd, timeout=600):
        venv_env = os.environ.copy()
        venv_env.update(state["env_vars"])
        venv_bin = os.path.join(repo_dir, ".venv", "bin")
        venv_env["PATH"] = venv_bin + ":" + venv_env.get("PATH", "")
        venv_env["VIRTUAL_ENV"] = os.path.join(repo_dir, ".venv")
        return subprocess.run(cmd, shell=True, cwd=repo_dir, capture_output=True,
                              text=True, timeout=timeout, env=venv_env)

    def h_read_docs(pcb, args):
        max_c = int(args.get("max_chars_per_file", 8000))
        patterns = ["README*", "INSTALL*", "CONTRIBUTING*", "docs/**/install*",
                    "docs/**/README*", "docs/**/quickstart*"]
        out = []
        for pat in patterns:
            for p in glob.glob(os.path.join(repo_dir, pat), recursive=True):
                try:
                    with open(p, encoding="utf-8", errors="ignore") as f:
                        txt = f.read()[:max_c]
                    out.append({"path": os.path.relpath(p, repo_dir), "content": txt})
                except OSError:
                    pass
        return {"files": out, "count": len(out)}

    def h_inspect_config(pcb, args):
        cfg = {"declared_python_versions": [], "install_deps": [],
               "test_extras_names": [], "dev_extras_names": [],
               "build_system": None, "has_editable_install": False}
        for name in ("setup.py", "setup.cfg", "pyproject.toml"):
            p = os.path.join(repo_dir, name)
            if not os.path.isfile(p):
                continue
            try:
                text = open(p, encoding="utf-8", errors="ignore").read()
            except OSError:
                continue
            # declared Python versions
            for m in re.finditer(r"Python\s*::\s*3\.(\d+)\b", text):
                v = f"3.{m.group(1)}"
                if v not in cfg["declared_python_versions"]:
                    cfg["declared_python_versions"].append(v)
            # extras (both [options.extras_require] in cfg, and pyproject
            # optional-dependencies)
            for m in re.finditer(r"^\s*(test|testing|tests|dev|doc)s?\s*=", text, re.MULTILINE):
                nm = m.group(1)
                key = "test_extras_names" if "test" in nm else "dev_extras_names"
                if nm not in cfg[key]:
                    cfg[key].append(nm)
            for m in re.finditer(r"^\s*\[(test|testing|tests|dev|docs)\]\s*$",
                                  text, re.MULTILINE):
                nm = m.group(1)
                key = "test_extras_names" if "test" in nm else "dev_extras_names"
                if nm not in cfg[key]:
                    cfg[key].append(nm)
            if "[build-system]" in text or "build-backend" in text:
                cfg["build_system"] = "pep517"
        # requirements files
        for f in glob.glob(os.path.join(repo_dir, "requirements*.txt")):
            try:
                cfg["install_deps"].append({
                    "path": os.path.relpath(f, repo_dir),
                    "content": open(f, encoding="utf-8", errors="ignore").read()[:2000],
                })
            except OSError:
                pass
        # .python-version file
        pv = os.path.join(repo_dir, ".python-version")
        if os.path.isfile(pv):
            try:
                v = open(pv).read().strip()
                if v and v not in cfg["declared_python_versions"]:
                    cfg["declared_python_versions"].insert(0, v)
            except OSError:
                pass
        return cfg

    def h_web_search(pcb, args):
        # Simple DuckDuckGo instant-answer JSON scraper. Not perfect but no
        # API key required. If the network is blocked or the endpoint is
        # rate-limited, we fail closed and tell the model.
        q = str(args.get("query", ""))
        n = int(args.get("n", 5))
        try:
            url = "https://duckduckgo.com/html/?q=" + urllib.parse.quote(q)
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 LLMOS-repo-bootstrap/1.0",
            })
            with urllib.request.urlopen(req, timeout=15) as r:
                html = r.read().decode("utf-8", errors="ignore")
            hits = []
            for m in re.finditer(
                r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>.*?'
                r'class="result__snippet"[^>]*>(.*?)</a>',
                html, re.DOTALL,
            ):
                title = re.sub(r"<[^>]+>", "", m.group(2)).strip()
                snippet = re.sub(r"<[^>]+>", "", m.group(3)).strip()
                url_hit = m.group(1)
                hits.append({"title": title[:200], "snippet": snippet[:400],
                             "url": url_hit[:300]})
                if len(hits) >= n:
                    break
            return {"query": q, "hits": hits}
        except Exception as e:
            return {"query": q, "hits": [], "error": str(e)}

    def h_provision_env(pcb, args):
        pyv = str(args.get("python_version", "3.11"))
        extras = list(args.get("extras", []) or [])
        extra_pkgs = list(args.get("extra_packages", []) or [])
        new_env = args.get("env_vars", {}) or {}
        if isinstance(new_env, dict):
            state["env_vars"].update({str(k): str(v) for k, v in new_env.items()})
        # create/replace the venv
        r1 = _run_in_venv(f"{UV} venv --python {pyv} .venv", timeout=180)
        # install the repo with extras (fall back to bare if extras missing)
        target = "." if not extras else f'".[{",".join(extras)}]"'
        r2 = _run_in_venv(
            f'{UV} pip install --python .venv/bin/python -e {target}',
            timeout=900)
        # extra loose packages
        r3 = None
        if extra_pkgs:
            r3 = _run_in_venv(
                f'{UV} pip install --python .venv/bin/python ' +
                " ".join(f'"{p}"' for p in extra_pkgs),
                timeout=300)
        state["last_python"] = pyv
        state["last_extras"] = extras
        # any new provision invalidates prior verification
        state["sanity_ok"] = False
        state["smoke_ok"] = False
        return {
            "python": pyv, "extras": extras,
            "venv_exit": r1.returncode,
            "install_exit": r2.returncode,
            "install_stderr_tail": (r2.stderr or "")[-1500:],
            "extra_pkgs_exit": r3.returncode if r3 else None,
            "env_vars": dict(state["env_vars"]),
        }

    def h_run_sanity(pcb, args):
        stmt = str(args.get("import_stmt", ""))
        r = _run_in_venv(
            f'.venv/bin/python -c "{stmt.replace(chr(34), chr(39))}"',
            timeout=120)
        ok = r.returncode == 0
        state["sanity_ok"] = ok
        return {"ok": ok, "exit": r.returncode,
                "stdout": (r.stdout or "")[-1500:],
                "stderr": (r.stderr or "")[-1500:]}

    def h_run_smoke(pcb, args):
        test_id = str(args.get("test_id", ""))
        extra_args = str(args.get("extra_pytest_args", ""))
        r_collect = _run_in_venv(
            f'.venv/bin/python -m pytest --collect-only -q {extra_args}',
            timeout=180)
        r_test = _run_in_venv(
            f'.venv/bin/python -m pytest -q {extra_args} "{test_id}"',
            timeout=300)
        ok = r_test.returncode == 0 and "passed" in (r_test.stdout or "")
        state["smoke_ok"] = ok
        return {
            "ok": ok,
            "collect_exit": r_collect.returncode,
            "collect_tail": (r_collect.stdout or "")[-1500:] + (r_collect.stderr or "")[-500:],
            "test_exit": r_test.returncode,
            "test_tail": (r_test.stdout or "")[-1500:] + (r_test.stderr or "")[-500:],
        }

    handlers = {
        "repo.read_docs":       h_read_docs,
        "repo.inspect_config":  h_inspect_config,
        "web.search":           h_web_search,
        "repo.provision_env":   h_provision_env,
        "repo.run_sanity":      h_run_sanity,
        "repo.run_smoke_test":  h_run_smoke,
    }
    return handlers, state


# ---------- Env-ready gate ---------------------------------------------
def env_ready(state):
    """The gate the fix phase checks before starting. Both verifications
    must have passed since the last provision_env; a fresh provision
    resets them to False, so the model can't declare ready without
    re-verifying after any env change."""
    return state["sanity_ok"] and state["smoke_ok"]


BOOTSTRAP_SYSTEM_PROMPT = (
    "You are setting up an unfamiliar Python repository. Work like a careful "
    "developer:\n"
    "  1. read_repo_docs — see what the project itself says about installation.\n"
    "  2. inspect_repo_config — check declared Python versions and test extras.\n"
    "  3. If the config is thin or ambiguous, web_search for setup help.\n"
    "  4. provision_env with a Python version and extras you can justify from what "
    "you've read.\n"
    "  5. run_sanity to verify the package imports.\n"
    "  6. run_smoke_test to verify pytest can collect and one specific test passes.\n"
    "  7. Only then call declare_env_ready.\n\n"
    "If run_sanity or run_smoke_test fails, read the error, form a hypothesis about "
    "what's missing (a system package, a lower Python version, a needed env var like "
    "DJANGO_SETTINGS_MODULE, a numpy version pin, etc.), and provision_env again with "
    "the fix. Do NOT call declare_env_ready until BOTH verifications have passed."
)
