"""Generic Python repo bootstrap toolkit + a verification gate.

Design principles (from Mikey):
  (1) 'Verify the env before moving on.' declare_env_ready is rejected
      until both run_sanity and run_smoke_test have returned ok=true.
  (2) 'Tools are a sort of hardened protocol; they should also be able to
      call the model.' Each tool has a FIXED interface (input schema,
      output shape) but is FREE inside — it can use regex, subprocess,
      AND the LLM to produce an actionable answer.
  (3) 'You need a subprocess.' Installations are recursive — install X,
      discover Y is needed, install Y, resume X. See install_tools.py for
      the primitives (create_venv, install_package, install_repo_editable,
      push/pop_subgoal, current_goal). This module wires them in with the
      docs/config-inspection tools that precede them.
"""
import glob, json, os, re, urllib.parse, urllib.request

from install_tools import (INSTALL_TOOLS, INSTALL_TOOL2SYS,
                           make_install_handlers, _stack_snapshot, _run)


# ---------- LLM helper: tools can call the model ------------------------
def llm_call(prompt, system="You are a helpful assistant. Answer concisely.",
             model="ornith:35b", host="http://127.0.0.1:11434",
             temperature=0.3, max_tokens=1600, timeout=300,
             format_json=False, num_ctx=131072):
    """Synchronous chat with the same ornith model the top-level agent uses.
    max_tokens defaulted UP to 1600 because thinking-mode's preamble was
    eating the whole budget before JSON emission. Pass format_json=True to
    request structured output.
    """
    body = {
        "model": model, "stream": False, "keep_alive": "24h",
        "messages": [{"role": "system", "content": system},
                     {"role": "user",   "content": prompt}],
        "options": dict({"temperature": temperature, "top_p": 0.95, "top_k": 20,
                    "num_predict": max_tokens},
                    **({"num_ctx": num_ctx} if num_ctx else {})),
    }
    if format_json:
        body["format"] = "json"
    data = json.dumps(body).encode()
    try:
        req = urllib.request.Request(host + "/api/chat", data=data,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            resp = json.loads(r.read())
        m = resp.get("message", {}) or {}
        return m.get("content", "") or m.get("thinking", "")
    except Exception as e:
        return f"[llm_call error: {e}]"


def _extract_json(text):
    """Loose JSON extraction — the LLM often wraps JSON in prose."""
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


# ---------- Tool schemas -----------------------------------------------
# Recon tools come first (read the docs, check the config, search the web),
# then the install primitives from install_tools, then verification/gate.

RECON_TOOLS = [
    {"type": "function", "function": {
        "name": "read_repo_docs",
        "description": (
            "Read README/INSTALL/CONTRIBUTING and docs/**/install* from the repo "
            "root. The tool ALSO asks the LLM to extract structured install info "
            "and returns {install_cmds, python_versions, system_prereqs, extras, "
            "backend_hint, notes}. backend_hint is 'conda' when the docs recommend "
            "miniforge/conda/mamba, else 'uv' or 'pip'."),
        "parameters": {"type": "object", "properties": {
            "max_chars_per_file": {"type": "integer", "default": 8000},
        }}}},
    {"type": "function", "function": {
        "name": "inspect_repo_config",
        "description": (
            "Parse setup.py / setup.cfg / pyproject.toml / requirements*.txt / "
            "tox.ini / .python-version. Returns {declared_python_versions, "
            "install_deps, test_extras_names, dev_extras_names, build_system, "
            "recommendation}. Use to decide Python version and extras."),
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "web_search",
        "description": (
            "Search the web (DDG). Returns hits + an LLM-synthesized answer in "
            "1-3 sentences. Use when config files don't tell you enough (e.g. "
            "'astropy 5.3 setuptools dep_util fix')."),
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"},
            "n":     {"type": "integer", "default": 5},
        }, "required": ["query"]}}},
]

VERIFY_TOOLS = [
    {"type": "function", "function": {
        "name": "run_sanity",
        "description": (
            "Try to import the package inside the active env. Returns stdout/"
            "stderr and an LLM diagnosis on failure. Sets state.sanity_ok."),
        "parameters": {"type": "object", "properties": {
            "import_stmt": {"type": "string",
                             "description": "e.g. 'import astropy; print(astropy.__version__)'"},
        }, "required": ["import_stmt"]}}},
    {"type": "function", "function": {
        "name": "run_smoke_test",
        "description": (
            "Run pytest --collect-only + one specific test id. Returns outputs "
            "and an LLM diagnosis on failure. Sets state.smoke_ok."),
        "parameters": {"type": "object", "properties": {
            "test_id":            {"type": "string"},
            "extra_pytest_args":  {"type": "string", "default": ""},
        }, "required": ["test_id"]}}},
    {"type": "function", "function": {
        "name": "declare_env_ready",
        "description": (
            "TERMINAL. Only accepted after run_sanity AND run_smoke_test have "
            "returned ok=true SINCE the last install_repo_editable. Any new "
            "install invalidates prior verifications."),
        "parameters": {"type": "object", "properties": {
            "summary": {"type": "string"},
        }, "required": ["summary"]}}},
]

BOOTSTRAP_TOOLS = RECON_TOOLS + INSTALL_TOOLS + VERIFY_TOOLS

BOOTSTRAP_TOOL2SYS = {
    "read_repo_docs":       "repo.read_docs",
    "inspect_repo_config":  "repo.inspect_config",
    "web_search":           "web.search",
    "run_sanity":           "repo.run_sanity",
    "run_smoke_test":       "repo.run_smoke_test",
    "declare_env_ready":    "RETURN",
}
BOOTSTRAP_TOOL2SYS.update(INSTALL_TOOL2SYS)




# ---------- doc research: read the docs like a careful developer ----------
# The old read_repo_docs skimmed 4 globbed files at 2000 chars and guessed.
# This pipeline does what a developer does: find the installation guide,
# READ it, follow its links (install docs usually point at the testing
# guide), read those too, then produce a concrete procedure. The tool
# interface stays hardened; the intelligence is llm_call inside.

_DOC_NAME_RE = re.compile(
    r"(install|setup|quickstart|contribut|develop|test|building|compil)",
    re.IGNORECASE)


def _doc_index(repo_dir):
    """All candidate doc files: top-level README/INSTALL/etc plus docs/**
    whose path suggests install/test/dev content."""
    idx = []
    for pat in ("README*", "INSTALL*", "CONTRIBUTING*", "TESTING*"):
        idx.extend(p for p in glob.glob(os.path.join(repo_dir, pat))
                   if os.path.isfile(p))
    docs_root = os.path.join(repo_dir, "docs")
    for root, dirs, files in os.walk(docs_root):
        dirs[:] = [d for d in dirs if not d.startswith((".", "_"))]
        for fn in files:
            full = os.path.join(root, fn)
            if fn.endswith((".rst", ".md", ".txt")) and _DOC_NAME_RE.search(
                    os.path.relpath(full, docs_root)):
                idx.append(full)
    out, seen = [], set()
    for p in idx:
        rp = os.path.relpath(p, repo_dir)
        if rp in seen:
            continue
        seen.add(rp)
        try:
            out.append({"path": rp, "size": os.path.getsize(p)})
        except OSError:
            pass
    return out


_LINK_RES = [
    re.compile(r":doc:`[^`<]*<([^>`]+)>`"),
    re.compile(r":doc:`([^`<>]+)`"),
    re.compile(r"\.\. include::\s*(\S+)"),
    re.compile(r"\]\(([^)#\s]+\.(?:rst|md))\)"),
    re.compile(r"^\s{3,}([\w][\w/.-]{3,})\s*$", re.MULTILINE),  # toctree
]


def _resolve_links(content, index_paths):
    """References to other LOCAL doc files, resolved against the index by
    stem suffix-match (e.g. 'development/testguide' ->
    'docs/development/testguide.rst'). Noise dies here: a toctree word
    that matches no indexed file resolves to nothing."""
    cands = set()
    for rx in _LINK_RES:
        for m in rx.finditer(content):
            cands.add(m.group(1).strip())
    hits = []
    for c in cands:
        stem = re.sub(r"\.(rst|md|txt)$", "", c).strip("/")
        if len(stem) < 4:
            continue
        for ip in index_paths:
            if re.sub(r"\.(rst|md|txt)$", "", ip).endswith(stem) and ip not in hits:
                hits.append(ip)
    return hits


def research_docs(repo_dir, max_read=6, chars_per_file=7000):
    """Find the install guide, read it, follow links one hop, synthesize
    a procedure. Returns recommendation with `procedure` (ordered steps)
    and `smoke_test_hint` (the docs' own recommended small test run)."""
    index = _doc_index(repo_dir)
    index_paths = [d["path"] for d in index]
    listing = "\n".join(f"- {d['path']} ({d['size']}B)" for d in index[:80])
    sel_raw = llm_call(
        system=("You plan documentation reading before installing a Python "
                "project from source. JSON only, no thinking preamble."),
        prompt=("A developer must install this repository from source and run "
                "its tests. Which files should be READ, in priority order? "
                "The INSTALLATION guide first, then testing/contributing docs.\n\n"
                f"{listing}\n\n"
                'Return JSON: {"read": ["path", ...]} (max 4 paths, exactly '
                "as listed above)"),
        max_tokens=800, format_json=True)
    sel = (_extract_json(sel_raw) or {}).get("read", []) or []
    to_read = [p for p in sel if p in index_paths][:4] or index_paths[:3]
    if not any("test" in t.lower() for t in to_read):
        tdocs = sorted((d for d in index if "test" in d["path"].lower()),
                       key=lambda d: -d["size"])
        if tdocs:
            to_read.append(tdocs[0]["path"])
    read, queue = [], list(to_read)
    while queue and len(read) < max_read:
        rp = queue.pop(0)
        if any(r["path"] == rp for r in read):
            continue
        try:
            content = open(os.path.join(repo_dir, rp), encoding="utf-8",
                           errors="ignore").read()[:chars_per_file]
        except OSError:
            continue
        read.append({"path": rp, "content": content})
        for linked in _resolve_links(content, index_paths):
            if linked not in queue and not any(r["path"] == linked
                                               for r in read):
                queue.append(linked)
    blob = "\n\n".join(f"=== FILE: {r['path']} ===\n{r['content']}"
                        for r in read)
    syn_raw = llm_call(
        system=("You turn project documentation into an exact install-and-"
                "test procedure. JSON only, no thinking preamble."),
        prompt=("Documentation that was actually read (install guide plus the "
                "pages it links to):\n\n"
                f"{blob[:26000]}\n\n"
                "Return JSON with keys:\n"
                '  python_versions: supported Python versions\n'
                '  backend_hint: "conda" if docs recommend conda/miniforge/'
                'mamba, else "uv" or "pip"\n'
                '  system_prereqs: OS packages the docs require\n'
                '  build_deps: python packages (with any version pins) the '
                "docs say are needed before/while building\n"
                '  procedure: ordered list of concrete steps to install from '
                "source with test extras, per the docs\n"
                '  test_env_vars: env vars the docs say tests need\n'
                '  smoke_test_hint: the docs-recommended way to run a SMALL '
                "test subset (single file or subpackage), verbatim from the "
                "docs\n"
                '  gotchas: notable warnings from the docs'),
        max_tokens=3200, format_json=True)
    parsed = _extract_json(syn_raw) or {}
    return {"files_indexed": len(index),
            "files_read": [r["path"] for r in read],
            "recommendation": parsed,
            "recommendation_raw": syn_raw[:2000]}


# ---------- Handlers ----------------------------------------------------

def make_bootstrap_handlers(repo_dir, base_env_vars=None, fail_to_pass=None):
    """Compose install handlers + recon/verify handlers into one dispatch
    table. All share state so goal_stack, active_env_kind, sanity_ok,
    smoke_ok are visible to every tool.

    fail_to_pass: the instance's FAIL_TO_PASS test ids. These are the BUG —
    they are expected to fail until phase 2 fixes the code, so run_smoke_test
    refuses them as environment-health checks (v8 postmortem: the model spent
    20+ turns trying to smoke-test the very test the issue breaks)."""
    install_handlers, state = make_install_handlers(repo_dir, base_env_vars)
    state["fail_to_pass"] = list(fail_to_pass or [])

    # ---- read_repo_docs -----------------------------------------------
    def h_read_docs(pcb, args):
        out = research_docs(repo_dir)
        out["goal_stack"] = _stack_snapshot(state)
        return out

    # ---- inspect_repo_config ------------------------------------------
    def h_inspect_config(pcb, args):
        cfg = {"declared_python_versions": [], "install_deps": [],
               "test_extras_names": [], "dev_extras_names": [],
               "build_system": None}
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
            for m in re.finditer(r"Python\s*::\s*3\.(\d+)\b", text):
                v = f"3.{m.group(1)}"
                if v not in cfg["declared_python_versions"]:
                    cfg["declared_python_versions"].append(v)
            for m in re.finditer(r"^\s*(test|testing|tests|dev|doc)s?\s*=",
                                  text, re.MULTILINE):
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
            # python_requires directive
            m = re.search(r'python[_-]?requires\s*=\s*[\'"]?([^\'"\n]+)',
                          text)
            if m and m.group(1).strip() not in cfg.get("python_requires_raw", ""):
                cfg["python_requires_raw"] = m.group(1).strip()
        for f in glob.glob(os.path.join(repo_dir, "requirements*.txt")):
            try:
                cfg["install_deps"].append({
                    "path": os.path.relpath(f, repo_dir),
                    "content": open(f, encoding="utf-8",
                                    errors="ignore").read()[:2000],
                })
            except OSError:
                pass
        pv = os.path.join(repo_dir, ".python-version")
        if os.path.isfile(pv):
            try:
                v = open(pv).read().strip()
                if v and v not in cfg["declared_python_versions"]:
                    cfg["declared_python_versions"].insert(0, v)
            except OSError:
                pass
        # LLM synthesizes a concrete plan
        blob = "\n\n---\n\n".join(f"### {n}\n{t[:6000]}"
                                    for n, t in config_texts)
        recommendation_raw = ""
        if blob:
            recommendation_raw = llm_call(
                system=("Configure a Python build. Return JSON only, "
                        "no thinking preamble."),
                prompt=("Given these config files, propose a concrete install "
                        "plan.\n\n"
                        f"{blob[:12000]}\n\n"
                        "Return JSON with keys:\n"
                        '  python_version: e.g. "3.11"\n'
                        '  backend: "uv"|"conda"|"pip"\n'
                        '  extras_name: which extras to install, e.g. "test"\n'
                        '  needs_no_build_isolation: bool\n'
                        '  build_deps: list of packages to install BEFORE the repo '
                        '(e.g. ["setuptools<69","numpy<2","cython<3","extension_helpers"])\n'
                        '  env_vars: dict of runtime env vars\n'
                        '  expected_import: what to run_sanity\n'
                        '  smoke_test: single test id for run_smoke_test\n'
                        '  reasoning: short prose'),
                max_tokens=1600, format_json=True)
        cfg["recommendation"] = _extract_json(recommendation_raw) or {}
        cfg["recommendation_raw"] = recommendation_raw[:2000]
        cfg["goal_stack"] = _stack_snapshot(state)
        return cfg

    # ---- web_search ----------------------------------------------------
    def h_web_search(pcb, args):
        q = str(args.get("query", ""))
        n = int(args.get("n", 5))
        hits = []
        try:
            api = ("https://api.duckduckgo.com/?format=json&no_html=1&q="
                   + urllib.parse.quote(q))
            with urllib.request.urlopen(api, timeout=15) as r:
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
        try:
            url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote(q)
            req = urllib.request.Request(url, headers={
                "User-Agent": ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/"
                                "537.36 (KHTML, like Gecko) Chrome/120.0 "
                                "Safari/537.36")})
            with urllib.request.urlopen(req, timeout=15) as r:
                html = r.read().decode("utf-8", errors="ignore")
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
                    if len(hits) >= n: break
                if len(hits) >= n: break
        except Exception:
            pass
        synthesis = ""
        if hits:
            hits_blob = "\n".join(f"- {h['title']}: {h['snippet']}"
                                    for h in hits)
            synthesis = llm_call(
                system="Synthesize search results for a Python-setup question.",
                prompt=(f"Question: {q}\n\nSearch results:\n{hits_blob}\n\n"
                        "Extract the specific actionable answer in 1-3 sentences."),
                max_tokens=600)
        return {"query": q, "hits": hits, "answer": synthesis,
                "goal_stack": _stack_snapshot(state)}

    # ---- run_sanity ----------------------------------------------------
    def h_run_sanity(pcb, args):
        active = state["active_env_kind"]
        if not active:
            return {"error": "no venv yet", "ok": False,
                    "goal_stack": _stack_snapshot(state)}
        stmt = str(args.get("import_stmt", ""))
        bin_ = ".condaenv/bin" if active == "conda" else ".venv/bin"
        r = _run(f'{bin_}/python -c "{stmt.replace(chr(34), chr(39))}"',
                 repo_dir, env_vars=state["env_vars"], timeout=120,
                 active_env_kind=active)
        ok = r.returncode == 0
        state["sanity_ok"] = ok
        result = {"ok": ok, "exit": r.returncode,
                  "stdout": (r.stdout or "")[-1500:],
                  "stderr": (r.stderr or "")[-1500:],
                  "goal_stack": _stack_snapshot(state)}
        if not ok:
            result["diagnosis"] = llm_call(
                system="Diagnose a Python import failure. Be specific.",
                prompt=(f"An import (`{stmt}`) failed inside a freshly-"
                        f"provisioned venv.\n\nError:\n{result['stderr']}\n\n"
                        "Name the ONE most likely cause (missing system lib, "
                        "wrong Python, missing extra, wrong install backend) "
                        "and the ONE next thing to try. 2 sentences."),
                max_tokens=400)
        return result

    # ---- run_smoke_test ------------------------------------------------
    def h_run_smoke(pcb, args):
        active = state["active_env_kind"]
        if not active:
            return {"error": "no venv yet", "ok": False,
                    "goal_stack": _stack_snapshot(state)}
        test_id = str(args.get("test_id", ""))
        extra_args = str(args.get("extra_pytest_args", ""))
        # Guard: the instance's failing tests ARE the bug — they cannot
        # validate the environment. Match on test function name or file.
        for ftp in state.get("fail_to_pass", []):
            fname = ftp.rsplit("::", 1)[-1]
            ffile = ftp.split("::", 1)[0]
            if (fname and fname in test_id) or (ffile and ffile in test_id):
                return {"ok": False, "error": (
                    f"'{test_id}' matches a FAIL_TO_PASS test ({ftp}). That "
                    "test is the BUG this task is about — it is EXPECTED to "
                    "fail until the fix phase, so it cannot prove the "
                    "environment works. Pick a stable existing test from a "
                    "module UNRELATED to the problem statement (e.g. a basic "
                    "utils/ or io/ test), not a test you wrote yourself."),
                    "fail_to_pass": state["fail_to_pass"],
                    "goal_stack": _stack_snapshot(state)}
        bin_ = ".condaenv/bin" if active == "conda" else ".venv/bin"
        r_collect = _run(
            f'{bin_}/python -m pytest --collect-only -q {extra_args}',
            repo_dir, env_vars=state["env_vars"], timeout=180,
            active_env_kind=active)
        r_test = _run(
            f'{bin_}/python -m pytest -q {extra_args} "{test_id}"',
            repo_dir, env_vars=state["env_vars"], timeout=300,
            active_env_kind=active)
        ok = r_test.returncode == 0 and "passed" in (r_test.stdout or "")
        state["smoke_ok"] = ok
        result = {"ok": ok,
                  "collect_exit": r_collect.returncode,
                  "collect_tail": (r_collect.stdout or "")[-1500:]
                                   + (r_collect.stderr or "")[-500:],
                  "test_exit": r_test.returncode,
                  "test_tail": (r_test.stdout or "")[-1500:]
                                + (r_test.stderr or "")[-500:],
                  "goal_stack": _stack_snapshot(state)}
        if not ok:
            result["diagnosis"] = llm_call(
                system="Diagnose a pytest failure. Be specific.",
                prompt=(f"pytest smoke check failed on `{test_id}`.\n\n"
                        f"Collect output:\n{result['collect_tail']}\n\n"
                        f"Test output:\n{result['test_tail']}\n\n"
                        "Name the ONE most likely cause and the ONE next thing "
                        "to try. 2-3 sentences."),
                max_tokens=500)
        return result

    handlers = {
        "repo.read_docs":       h_read_docs,
        "repo.inspect_config":  h_inspect_config,
        "web.search":           h_web_search,
        "repo.run_sanity":      h_run_sanity,
        "repo.run_smoke_test":  h_run_smoke,
    }
    handlers.update(install_handlers)
    return handlers, state


# ---------- Env-ready gate ---------------------------------------------
def env_ready(state):
    return state["sanity_ok"] and state["smoke_ok"]


# ---------- System prompt ----------------------------------------------
BOOTSTRAP_SYSTEM_PROMPT = (
    "You are setting up an unfamiliar Python repository. Work like a careful "
    "developer.\n\n"
    "RECON (do this first):\n"
    "  1. read_repo_docs — this READS the installation guide and the pages "
    "it links to (testing guide etc.) and returns recommendation.procedure "
    "(ordered install steps from the docs), recommendation.build_deps, "
    "recommendation.backend_hint, and recommendation.smoke_test_hint (the "
    "docs' own way to run a small test). FOLLOW the procedure; use "
    "smoke_test_hint for run_smoke_test later.\n"
    "  2. inspect_repo_config — declared Python versions, extras, and build "
    "system. Check recommendation.backend and recommendation.build_deps.\n"
    "  3. web_search when config is thin or an error message is unfamiliar.\n\n"
    "INSTALL (recursive — this is the key part):\n"
    "  4. create_venv(python_version, backend). Pick backend='uv' by default "
    "(fast). Pick backend='conda' when the docs or inspect_config recommend it "
    "(scientific packages with compiled C extensions, e.g. astropy/scipy/"
    "matplotlib on Ubuntu often need conda-forge prebuilt binaries).\n"
    "  5. install_repo_editable(extras=['test']).\n"
    "  6. IF it fails with a build error (e.g. missing setuptools.dep_util, "
    "missing numpy header, cython version mismatch), DO NOT retry blindly. "
    "The install has a subprocess:\n"
    "     a. push_subgoal('install build deps for <package>')\n"
    "     b. install_package(name='setuptools', version_spec='<69', "
    "backend='uv') — or conda if you're in a conda env.\n"
    "     c. install_package for numpy<2, cython<3, extension_helpers, etc. "
    "as the error suggests.\n"
    "     d. pop_subgoal()\n"
    "     e. install_repo_editable(extras=['test'], no_build_isolation=True). "
    "The pre-installed build deps will now be visible to the repo's build.\n"
    "  7. IF the build fails with COMPILER errors (nested declaration, "
    "implicit function, C-standard complaints) rather than missing packages: "
    "set_env_var('CFLAGS', '-std=c99'), then install_repo_editable again. "
    "A missing-package error needs install_package; a compiler error needs "
    "set_env_var. Read the stderr to tell which.\n"
    "  8. If you get lost, call current_goal to see the stack.\n\n"
    "VERIFY (must both pass since the last install_repo_editable):\n"
    "  9. run_sanity — verify the package imports.\n"
    "  10. run_smoke_test — verify pytest collects and ONE existing test passes. "
    "CRITICAL: the tests you were asked to make pass are the BUG — they are "
    "expected to FAIL until the fix phase and prove nothing about the env. "
    "Choose a stable test from a module UNRELATED to the problem statement.\n"
    " 11. declare_env_ready. The harness rejects this until 9 and 10 both "
    "succeeded since the last install.\n\n"
    "Backend rules:\n"
    "  - uv: default. pure-Python packages, fast. Works in a .venv env.\n"
    "  - pip: works in either .venv or .condaenv. Use when uv is confused or "
    "you want plain --no-build-isolation semantics.\n"
    "  - conda: ONLY in a .condaenv env. Required for compiled scientific "
    "packages that lack pip wheels for this platform.\n\n"
    "Every turn MUST call exactly one tool."
)
