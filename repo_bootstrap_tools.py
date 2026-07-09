"""Generic Python repo bootstrap toolkit + a verification gate.

Purpose: give the model tools that mirror what a human developer does when
handed an unfamiliar Python project — read the docs, inspect the config,
pick a Python version, provision a venv, and VERIFY the env works before
touching anything else.

Design principle (from Mikey):
  (1) 'Verify the env before moving on.' declare_env_ready is rejected
      until both run_sanity and run_smoke_test have returned ok=true.
  (2) 'Tools are a sort of hardened protocol; they should also be able to
      call the model.' Each tool has a FIXED interface (input schema,
      output shape) but is FREE inside — it can use regex, subprocess,
      AND the LLM to produce its answer. Tools that reason about messy
      inputs (docs, config, search results, error logs) are much more
      valuable when they can ask the LLM to synthesize an actionable
      recommendation, not just return raw bytes.
"""
import glob, json, os, re, subprocess, urllib.parse, urllib.request


UV = os.path.expanduser("~/.local/bin/uv")


# ---------- LLM helper: tools can call the model ------------------------
# A tool is a hardened protocol at the interface layer; internally it may
# do whatever helps produce a good answer, including asking the LLM.
def llm_call(prompt, system="You are a helpful assistant. Answer concisely.",
             model="ornith:35b", host="http://127.0.0.1:11434",
             temperature=0.3, max_tokens=800, timeout=180):
    """Synchronous chat with the same ornith model the top-level agent uses.
    Lower temperature than the agent's T=1.0: this is subordinate reasoning,
    not exploration. Returns just the text content.
    """
    body = json.dumps({
        "model": model, "stream": False, "keep_alive": "24h",
        "messages": [{"role": "system", "content": system},
                     {"role": "user",   "content": prompt}],
        "options": {"temperature": temperature, "top_p": 0.95, "top_k": 20,
                    "num_predict": max_tokens},
    }).encode()
    try:
        req = urllib.request.Request(host + "/api/chat", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            resp = json.loads(r.read())
        m = resp.get("message", {}) or {}
        return m.get("content", "") or m.get("thinking", "")
    except Exception as e:
        return f"[llm_call error: {e}]"


def _extract_json(text):
    """Loose JSON extraction — the LLM often wraps JSON in prose. Grab the
    first balanced {...} that parses."""
    if not text: return None
    depth, start = 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0: start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                cand = text[start:i+1]
                try: return json.loads(cand)
                except Exception: pass
    return None


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
        """Read the docs AND ask the model to extract the specific install
        recommendation. The tool's job is to produce an actionable answer,
        not to dump raw README bytes into the caller's context."""
        max_c = int(args.get("max_chars_per_file", 8000))
        patterns = ["README*", "INSTALL*", "CONTRIBUTING*", "docs/**/install*",
                    "docs/**/README*", "docs/**/quickstart*"]
        raw = []
        for pat in patterns:
            for p in glob.glob(os.path.join(repo_dir, pat), recursive=True):
                try:
                    with open(p, encoding="utf-8", errors="ignore") as f:
                        raw.append({"path": os.path.relpath(p, repo_dir),
                                    "content": f.read()[:max_c]})
                except OSError:
                    pass
        # Ask the LLM to extract the install-relevant parts. Cheaper than
        # returning ~24KB of README for the caller to re-read.
        blob = "\n\n---\n\n".join(f"### {d['path']}\n{d['content']}" for d in raw[:4])
        recommendation = ""
        if blob:
            recommendation = llm_call(
                system=("You extract Python-project install info. "
                        "Answer JSON only."),
                prompt=("From the following project docs, extract:\n"
                        "- supported Python versions (as a list)\n"
                        "- exact install command(s) recommended\n"
                        "- any system prerequisites (apt/brew packages)\n"
                        "- any environment variables required to run tests\n"
                        "- extras (e.g. `.[test]`, `.[dev]`) mentioned\n\n"
                        f"Docs:\n{blob[:16000]}\n\n"
                        'Return JSON: {"python_versions":[], "install_cmds":[], '
                        '"system_prereqs":[], "test_env_vars":{}, '
                        '"extras":[], "notes":""}'))
        parsed = _extract_json(recommendation) or {}
        return {"files": [{"path": d["path"], "size": len(d["content"])} for d in raw],
                "count": len(raw),
                "recommendation": parsed,
                "recommendation_raw": recommendation[:2000]}

    def h_inspect_config(pcb, args):
        """Read the config files AND ask the LLM to synthesize a concrete
        install recommendation. The regex parser gets the shape; the LLM
        makes the call."""
        cfg = {"declared_python_versions": [], "install_deps": [],
               "test_extras_names": [], "dev_extras_names": [],
               "build_system": None, "has_editable_install": False}
        config_texts = []
        for name in ("setup.py", "setup.cfg", "pyproject.toml"):
            p = os.path.join(repo_dir, name)
            if not os.path.isfile(p):
                continue
            try:
                text = open(p, encoding="utf-8", errors="ignore").read()
            except OSError:
                continue
            config_texts.append((name, text))
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
        # LLM synthesis: given the config texts, what should we do?
        blob = "\n\n---\n\n".join(f"### {n}\n{t[:6000]}" for n, t in config_texts)
        recommendation = ""
        if blob:
            recommendation = llm_call(
                system=("You configure Python builds. Choose a specific Python "
                        "version and install command. Answer JSON only."),
                prompt=("Given these config files, give a concrete install plan "
                        "that will produce a working test environment.\n\n"
                        f"{blob[:12000]}\n\n"
                        'Return JSON: {"python_version":"3.X", '
                        '"install_cmd":"uv pip install -e \\".[test]\\"", '
                        '"extras_name":"test", "extra_packages":["hypothesis"], '
                        '"env_vars":{"KEY":"VALUE"}, '
                        '"expected_import":"import astropy", '
                        '"smoke_test":"path::test_name", '
                        '"reasoning":""}'))
        parsed = _extract_json(recommendation) or {}
        cfg["recommendation"] = parsed
        cfg["recommendation_raw"] = recommendation[:2000]
        return cfg

    def h_web_search(pcb, args):
        # Two-tier search: try DuckDuckGo Instant Answer API (JSON, stable)
        # first for a definitive answer; if that's empty, hit the HTML SERP
        # and parse the newer container class names DDG uses (2024+).
        # Historically the old class="result__a" regex was returning 0 hits
        # because DDG changed its HTML — the model was flying blind.
        q = str(args.get("query", ""))
        n = int(args.get("n", 5))
        hits = []
        # Tier 1: Instant Answer API
        try:
            api_url = ("https://api.duckduckgo.com/?format=json&no_html=1&q="
                       + urllib.parse.quote(q))
            with urllib.request.urlopen(api_url, timeout=15) as r:
                d = json.loads(r.read())
            if d.get("AbstractText"):
                hits.append({"title": d.get("Heading", ""),
                             "snippet": d["AbstractText"][:400],
                             "url": d.get("AbstractURL", "")})
            for rel in (d.get("RelatedTopics") or [])[:n]:
                if isinstance(rel, dict) and rel.get("Text"):
                    hits.append({"title": rel.get("Text", "")[:200],
                                 "snippet": rel.get("Text", "")[:400],
                                 "url": rel.get("FirstURL", "")})
        except Exception:
            pass
        # Tier 2: HTML SERP with the modern class names
        try:
            url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote(q)
            req = urllib.request.Request(url, headers={
                "User-Agent": ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                                "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"),
            })
            with urllib.request.urlopen(req, timeout=15) as r:
                html = r.read().decode("utf-8", errors="ignore")
            # try several patterns (DDG changes class names periodically)
            for pat in [
                r'<a[^>]+class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>.*?'
                r'class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>',
                r'<a[^>]+href="([^"]+)"[^>]+class="[^"]*result__a[^"]*"[^>]*>(.*?)</a>.*?'
                r'result__snippet"[^>]*>(.*?)</',
                r'<h2[^>]*>\s*<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>\s*</h2>.*?'
                r'<div[^>]*>(.*?)</div>',
            ]:
                for m in re.finditer(pat, html, re.DOTALL):
                    title = re.sub(r"<[^>]+>", "", m.group(2)).strip()
                    snippet = re.sub(r"<[^>]+>", "", m.group(3)).strip()
                    url_hit = m.group(1)
                    if title and snippet:
                        hits.append({"title": title[:200],
                                     "snippet": snippet[:400],
                                     "url": url_hit[:300]})
                    if len(hits) >= n:
                        break
                if len(hits) >= n:
                    break
        except Exception:
            pass
        try:
            # Ask the LLM to synthesize the answer from the hits. This is
            # the whole point — search dumps snippets; the tool returns
            # actionable knowledge.
            synthesis = ""
            if hits:
                hits_blob = "\n".join(f"- {h['title']}: {h['snippet']}"
                                       for h in hits)
                synthesis = llm_call(
                    system=("You synthesize search results for a Python-setup "
                            "question. Answer concisely."),
                    prompt=(f"Question: {q}\n\nSearch results:\n{hits_blob}\n\n"
                            "Extract the specific actionable answer to the "
                            "question in 1-3 sentences."))
            return {"query": q, "hits": hits, "answer": synthesis}
        except Exception as e:
            return {"query": q, "hits": [], "error": str(e)}

    def h_provision_env(pcb, args):
        pyv = str(args.get("python_version", "3.11"))
        extras = list(args.get("extras", []) or [])
        extra_pkgs = list(args.get("extra_packages", []) or [])
        new_env = args.get("env_vars", {}) or {}
        if isinstance(new_env, dict):
            state["env_vars"].update({str(k): str(v) for k, v in new_env.items()})
        # Wipe any prior .venv so uv can (re)create it. Without this, uv
        # refuses to overwrite and the model's retries with different Python
        # versions silently reuse the broken env — bootstrap can't converge.
        import shutil as _sh
        _sh.rmtree(os.path.join(repo_dir, ".venv"), ignore_errors=True)
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
        # If install failed, ask the LLM to name what's likely wrong. Cheaper
        # for the outer agent than parsing pip's error stream itself.
        diagnosis = ""
        if r2.returncode != 0:
            err_tail = (r2.stderr or "")[-3000:]
            diagnosis = llm_call(
                system="You diagnose Python pip-install failures. Be specific.",
                prompt=("A `uv pip install -e .[extras]` command failed. Given "
                        "the error tail below, name the ONE most likely cause "
                        f"and the ONE next thing to try.\n\nError:\n{err_tail}\n\n"
                        "Answer in 2-3 sentences."))
        return {
            "python": pyv, "extras": extras,
            "venv_exit": r1.returncode,
            "install_exit": r2.returncode,
            "install_stderr_tail": (r2.stderr or "")[-1500:],
            "install_diagnosis": diagnosis,
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
        result = {"ok": ok, "exit": r.returncode,
                  "stdout": (r.stdout or "")[-1500:],
                  "stderr": (r.stderr or "")[-1500:]}
        if not ok:
            result["diagnosis"] = llm_call(
                system="You diagnose Python import failures. Be specific.",
                prompt=(f"An import (`{stmt}`) failed inside a freshly-provisioned "
                        f"venv. Given the error, name the ONE most likely cause "
                        f"(missing system lib, wrong Python, missing extra) and "
                        f"the ONE next thing to try.\n\n"
                        f"Error:\n{result['stderr']}\n\nAnswer in 2 sentences."))
        return result

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
        result = {
            "ok": ok,
            "collect_exit": r_collect.returncode,
            "collect_tail": (r_collect.stdout or "")[-1500:] + (r_collect.stderr or "")[-500:],
            "test_exit": r_test.returncode,
            "test_tail": (r_test.stdout or "")[-1500:] + (r_test.stderr or "")[-500:],
        }
        if not ok:
            result["diagnosis"] = llm_call(
                system="You diagnose pytest failures. Be specific.",
                prompt=(f"pytest smoke check failed on `{test_id}`. Given the "
                        f"output, name the ONE most likely cause (missing env "
                        f"var like DJANGO_SETTINGS_MODULE, missing test dep, "
                        f"wrong extras, test doesn't exist, etc.) and the ONE "
                        f"next thing to try.\n\n"
                        f"Collect output:\n{result['collect_tail']}\n\n"
                        f"Test output:\n{result['test_tail']}\n\n"
                        f"Answer in 2-3 sentences."))
        return result

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
