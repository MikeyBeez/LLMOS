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
import json, os, re, shutil, subprocess, sys, tempfile, time

sys.path.insert(0, os.path.expanduser("~/Code/LLMOS"))
from tool_call_cpu import ToolCallCPU
from repo_bootstrap_tools import (BOOTSTRAP_TOOLS, BOOTSTRAP_TOOL2SYS, auto_verify_env,
                                   BOOTSTRAP_SYSTEM_PROMPT,
                                   make_bootstrap_handlers, env_ready)
from swe_fix_tools import (FIX_TOOLS, FIX_TOOL2SYS, FIX_SYSTEM_PROMPT,
                            make_fix_handlers)
import envcheck
from trace_consumers import (remedies_for, format_remedy_context,
                             patterns_load, format_patterns_context,
                             harvest_trace, critic_review, error_signature,
                             playbook_for, format_playbook_context)
from repo_bootstrap_tools import _ddg_search

HOST = "http://127.0.0.1:8080"   # llama-server direct (ollama retired)
MODEL = "ornith:35b"
NUMCTX = 131072
NUM_PREDICT = int(os.environ.get("NUM_PREDICT", "2048"))
BOOTSTRAP_BUDGET = 50     # bumped for recursive install (each install_package is 1 turn)
FIX_BUDGET      = 80
WORK = os.path.expanduser("~/swe/work")
TRACES = os.path.expanduser("~/swe/traces_v2")
SCORE_LOGS = os.path.expanduser("~/swe/score_logs")  # full final-scorer output (telemetry only)


def sh(cmd, cwd=None, timeout=300):
    return subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True,
                          text=True, timeout=timeout)


MIRRORS = os.path.expanduser("~/swe/mirrors")


def clone(inst):
    """Working checkout via a local --mirror cache. The network is touched
    only when the mirror is missing or lacks base_commit; checkouts are
    created from the mirror (fast, local). Existing checkouts are RESET and
    REUSED, never re-downloaded — and never deleted on failure."""
    repo = os.path.join(WORK, inst["instance_id"])
    mirror = os.path.join(MIRRORS, inst["repo"].replace("/", "__") + ".git")
    os.makedirs(MIRRORS, exist_ok=True)
    # 1. Ensure the mirror holds base_commit (full history + tags: SWE-bench
    #    scoring and setuptools_scm both need tags; shallow clones broke this).
    if not os.path.isdir(mirror):
        sh(f"git clone -q --mirror https://github.com/{inst['repo']}.git {mirror}",
           timeout=7200)
    if sh(f"git -C {mirror} cat-file -e {inst['base_commit']}").returncode != 0:
        sh(f"git -C {mirror} fetch -q --tags origin", timeout=7200)
    # 2. (Re)use the working checkout — reset + clean, no network.
    if os.path.isdir(os.path.join(repo, ".git")):
        sh("git reset -q --hard && git clean -qfdx", cwd=repo, timeout=600)
        sh(f"git checkout -q {inst['base_commit']}", cwd=repo, timeout=300)
    else:
        shutil.rmtree(repo, ignore_errors=True)
        sh(f"git clone -q --shared {mirror} {repo}", timeout=600)
        sh(f"git checkout -q {inst['base_commit']}", cwd=repo, timeout=300)
    sh("git config user.email a@b.c; git config user.name a", cwd=repo)
    return repo


def _auto_verify_reject_detail(res):
    """Turn an auto_verify_env() result into a short, actionable hint for the
    model when the env-ready gate rejects a declare_env_ready. Returns None if
    the env actually verified (nothing to surface). ENV-DIAGNOSTIC ONLY --
    auto_verify_env excludes the instance's FAIL_TO_PASS tests, so this never
    leaks gold/test content."""
    if not isinstance(res, dict) or res.get("ok"):
        return None
    mod = res.get("missing_module")
    if mod:
        return ("harness auto-verify: environment is missing a TEST dependency "
                "`" + str(mod) + "` -- install it (pip/uv) into the active env, "
                "then declare again.")
    err = res.get("error")
    if err:
        return "harness auto-verify failed: " + str(err)
    return ("harness auto-verify could not confirm any green test -- check the "
            "install; then call run_smoke_test WITH NO ARGUMENTS to let the "
            "harness pick a known-stable test.")


def phase_run(cpu, tools, tool2sys, handlers, system_prompt, user_goal,
              budget, gate=None, log=print, checkpoint=None):
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
    searched_sigs = {}   # error signature -> turn first searched
    for turn in range(budget):
        msg = None
        for attempt in range(3):
            try:
                msg, meta = cpu._chat(messages)
                break
            except Exception as e:
                err = str(e)
                time.sleep(20 * (attempt + 1))
        if msg is None:
            return "cpu_error", messages, meta_log + [{"error": err}]
        meta_log.append({"turn": turn,
                          "prompt_tokens": meta.get("prompt_tokens"),
                          "eval_tokens":   meta.get("eval_tokens")})
        if checkpoint:
            try:
                tmp = checkpoint + ".tmp"
                json.dump({"phase1": messages, "phase1_meta": meta_log,
                           "partial": True},
                          open(tmp, "w"), default=str)
                os.replace(tmp, checkpoint)
            except Exception:
                pass
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
                _gate_payload = {"error": "verification gate not passed; "
                                          "run_sanity and run_smoke_test must "
                                          "both return ok=true first"}
                _detail = getattr(gate, "reject_detail", None)
                if _detail:
                    _gate_payload["harness_check"] = _detail
                messages.append({"role": "tool", "tool_call_id": f"t{turn}",
                                  "content": json.dumps(_gate_payload)})
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
        # Human reflex: see an error -> search the web for it -> THEN act.
        failed = isinstance(result, dict) and (
            result.get("ok") is False or "error" in result)
        if failed:
            sig = error_signature(str(result.get("error")
                                      or result.get("stderr") or ""))
            if sig and len(sig) > 12:
                if sig in searched_sigs:
                    result["error_web_search"] = (
                        f"(already searched at turn {searched_sigs[sig]} — "
                        "same error again means your last change did not "
                        "address it; re-read those results, try a DIFFERENT "
                        "action)")
                else:
                    searched_sigs[sig] = turn
                    try:
                        hits = _ddg_search(sig[:120], 3)
                    except Exception:
                        hits = []
                    if hits:
                        result["error_web_search"] = [
                            {"title": h["title"][:120],
                             "snippet": h["snippet"][:240]} for h in hits]
        messages.append({"role": "assistant", "content": "",
                         "tool_calls": [{"id": f"t{turn}", "type": "function",
                                          "function": {"name": tool, "arguments": args}}]})
        messages.append({"role": "tool", "tool_call_id": f"t{turn}",
                         "content": json.dumps(result, default=str)[:4800]})
        # Mid-run critic: every 8 turns a detached reviewer scans the recent
        # trace (and web-searches the latest error) for loops/drift/self-harm.
        if turn % 8 == 7:
            try:
                advice = critic_review(messages)
            except Exception:
                advice = ""
            if advice:
                log(f"  [critic] {advice[:100]}")
                messages.append({"role": "user",
                                 "content": f"[HARNESS CRITIC] {advice}"})
    return "budget", messages, meta_log



# import-name -> pip package name, for the missing-module reflex
_PKG_ALIASES = {
    "cv2": "opencv-python", "yaml": "pyyaml", "PIL": "pillow",
    "sklearn": "scikit-learn", "bs4": "beautifulsoup4", "OpenSSL": "pyopenssl",
    "dateutil": "python-dateutil", "attr": "attrs", "jinja2": "jinja2",
}
_MISSING_RE = re.compile(r"No module named ['\"]([\w.]+)['\"]")


def _web_lookup_pkg(mod):
    """When the pip name isn't obvious, do what a developer does: search
    'how do I install <module>' and read the pip package name off the
    results. Returns a package name or None."""
    try:
        from repo_bootstrap_tools import _ddg_search, llm_call, _extract_json
    except Exception:
        return None
    hits = _ddg_search(f"python how do I install module {mod} pip", 5)
    if not hits:
        return None
    blob = "\n".join(f"- {h['title']}: {h['snippet']}" for h in hits)
    raw = llm_call(
        system="You map a Python import name to its pip package. JSON only.",
        prompt=(f"A Python import 'import {mod}' fails with ModuleNotFoundError. "
                f"From these search results, what is the exact pip install "
                f"name?\n\n{blob}\n\n"
                'Return JSON: {"pip_name": "..."} (just the package, or null '
                "if the results do not say)"),
        max_tokens=400, format_json=True)
    pkg = (_extract_json(raw) or {}).get("pip_name")
    return pkg if pkg and pkg not in ("null", "None") else None


def _try_pip(pkg, repo, env, env_dir):
    r = subprocess.run(f'{env_dir}/bin/python -m pip install "{pkg}"',
                       shell=True, cwd=repo, capture_output=True, text=True,
                       timeout=300, env=env)
    if r.returncode != 0 and "No module named pip" in (r.stderr or ""):
        subprocess.run(f'{env_dir}/bin/python -m ensurepip --default-pip',
                       shell=True, cwd=repo, capture_output=True, text=True,
                       timeout=120, env=env)
        r = subprocess.run(f'{env_dir}/bin/python -m pip install "{pkg}"',
                           shell=True, cwd=repo, capture_output=True, text=True,
                           timeout=300, env=env)
    return r.returncode == 0


def _pip_install(mod, repo, env, env_dir):
    """Install a missing module. Try the alias/bare name first; if that
    fails, web-search 'how do I install <module>' for the real pip name."""
    pkg = _PKG_ALIASES.get(mod, mod.split(".")[0])
    if _try_pip(pkg, repo, env, env_dir):
        return True, pkg
    looked = _web_lookup_pkg(mod)
    if looked and looked != pkg and _try_pip(looked, repo, env, env_dir):
        return True, looked
    return False, pkg


def _run_with_missing_module_reflex(cmd, repo, env, env_dir, max_installs=4):
    """Run cmd; while it fails with 'No module named X', install X and retry.
    Guard: each module installed at most once, so a genuine build failure
    (module truly unavailable) surfaces instead of looping."""
    tried = set()
    for _ in range(max_installs + 1):
        r = subprocess.run(cmd, shell=True, cwd=repo, capture_output=True,
                           text=True, timeout=600, env=env)
        out = (r.stdout or "") + (r.stderr or "")
        m = _MISSING_RE.search(out)
        if not m:
            return r
        mod = m.group(1)
        if mod in tried:
            return r   # already installed it once — real failure, surface it
        tried.add(mod)
        ok, pkg = _pip_install(mod, repo, env, env_dir)
        if not ok:
            return r
    return r


# --- Env-faithfulness correction for warnings-as-errors repos -----------------
# Some repos run with `filterwarnings = error`. A too-new *pure-python* dep can
# emit a DeprecationWarning at import time that becomes fatal and turns test
# COLLECTION into "found no collectors" -- scoring a CORRECT patch as a miss
# (false negative; Docker-confirmed on matplotlib 23913/23964/23987/24149).
# Pin such deps back to an era-compatible version. General, repo-level; derived
# from package behaviour, never from any instance's answer.
WARN_AS_ERROR_DEP_PINS = {
    # matplotlib 3.x calls pyparsing's camelCase API (enablePackrat/setParseAction);
    # pyparsing >=3.1 raises PyparsingDeprecationWarning on those -> fatal under
    # matplotlib's filterwarnings=error. <3.1 keeps the API but stays silent.
    # Second cause, same NO_COLLECTORS symptom: some matplotlib dev builds compute
    # __version__ via setuptools_scm.get_version() AT IMPORT; setuptools-scm 8+ pulls
    # in vcs-versioning, whose "release-branch-semver" entry-point is a deprecation
    # shim that raises DeprecationWarning -> fatal collection error even once
    # pyparsing is pinned. Downgrade to the self-contained setuptools-scm 7.x AND
    # remove the orphaned vcs-versioning so the native (silent) scheme is used.
    # A spec written as "-pkg" means uninstall pkg.
    "matplotlib/matplotlib": ["pyparsing<3.1", "setuptools_scm<8", "-vcs_versioning"],
}


def pin_warn_as_error_deps(repo_dir, repo_name, env_kind="uv", env_vars=None):
    """Downgrade too-new pure-python deps that break test collection under a
    warnings-as-errors repo. Env-layer; general (repo-level); never touches the
    answer. Returns the applied pins (empty if repo unaffected)."""
    pins = WARN_AS_ERROR_DEP_PINS.get(repo_name)
    if not pins:
        return []
    env_dir = ".condaenv" if env_kind == "conda" else ".venv"
    py = os.path.join(repo_dir, env_dir, "bin", "python")
    if not os.path.exists(py):
        return []
    env = os.environ.copy(); env.update(env_vars or {})
    installs = [p for p in pins if not p.startswith("-")]
    removals = [p[1:] for p in pins if p.startswith("-")]
    ok = True
    if installs:
        quoted = " ".join('"%s"' % p for p in installs)
        r = subprocess.run('"%s" -m pip install %s' % (py, quoted),
                           shell=True, cwd=repo_dir, capture_output=True, text=True,
                           timeout=600, env=env)
        ok = ok and r.returncode == 0
    for pkg in removals:
        # uninstalling an absent package is not an error (pip prints "not installed")
        r = subprocess.run('"%s" -m pip uninstall -y "%s"' % (py, pkg),
                           shell=True, cwd=repo_dir, capture_output=True, text=True,
                           timeout=300, env=env)
        ok = ok and r.returncode == 0
    print(" -- warn-as-error dep pins (%s): %s" % (
        "ok" if ok else "FAIL", pins), flush=True)
    return pins


_LOCAL_HTTPBIN_MARKER = "LLMOS harness httpbin shim"

# Repos whose test-suite reaches an EXTERNAL http service via an env var and so
# fail OFFLINE with ConnectionError -> a correct patch is scored a false
# negative. psf/requests test_requests.py does
#   HTTPBIN = os.environ.get("HTTPBIN_URL", "http://httpbin.org/")
# and httpbin.org is unreachable here. pytest-httpbin bundles a local httpbin
# app; a repo-root conftest.py can start it and point HTTPBIN_URL at 127.0.0.1
# BEFORE the test module imports. General, repo-level; no answer/instance data.
LOCAL_HTTPBIN_REPOS = {"psf/requests"}

_LOCAL_HTTPBIN_CONFTEST = """# """ + _LOCAL_HTTPBIN_MARKER + """ (env layer, auto-generated).
# Some tests read HTTPBIN_URL and otherwise hit the public httpbin.org, which is
# unreachable offline (ConnectionError). Start a local httpbin (bundled with
# pytest-httpbin) and point HTTPBIN_URL at it, BEFORE test modules import. This
# file contains NO instance-specific knowledge and nothing derived from a patch.
import os
import atexit

if not os.environ.get("HTTPBIN_URL"):
    try:
        from httpbin import app as _httpbin_app
        from pytest_httpbin import serve as _serve
        _srv = _serve.Server(application=_httpbin_app)
        _srv._thread.daemon = True  # never block pytest exit
        _srv.start()
        atexit.register(_srv.stop)
        os.environ["HTTPBIN_URL"] = _srv.url
    except Exception:
        pass
"""


def ensure_local_httpbin(repo_dir, repo_name, env_kind="uv", env_vars=None):
    """For repos whose tests read HTTPBIN_URL and otherwise hit httpbin.org
    (offline ConnectionError -> false negative): ensure pytest-httpbin (bundles
    a local server) is installed and drop a repo-root conftest.py that starts it
    and sets HTTPBIN_URL. Env-layer; general (repo-level); never touches the
    answer. Returns True if wired."""
    if repo_name not in LOCAL_HTTPBIN_REPOS:
        return False
    env_dir = ".condaenv" if env_kind == "conda" else ".venv"
    py = os.path.join(repo_dir, env_dir, "bin", "python")
    if not os.path.exists(py):
        return False
    env = os.environ.copy(); env.update(env_vars or {})
    try:
        chk = subprocess.run('"%s" -c "import pytest_httpbin, httpbin"' % py,
                             shell=True, cwd=repo_dir, capture_output=True,
                             text=True, timeout=120, env=env)
        if chk.returncode != 0:
            subprocess.run('"%s" -m pip install --prefer-binary pytest-httpbin' % py,
                           shell=True, cwd=repo_dir, capture_output=True,
                           text=True, timeout=900, env=env)
    except Exception:
        pass
    cf = os.path.join(repo_dir, "conftest.py")
    try:
        if os.path.exists(cf):
            existing = open(cf, encoding="utf-8", errors="ignore").read()
            if _LOCAL_HTTPBIN_MARKER not in existing:
                print(" -- local httpbin: repo already has a conftest.py; not modifying", flush=True)
                return False
        open(cf, "w", encoding="utf-8").write(_LOCAL_HTTPBIN_CONFTEST)
        print(" -- local httpbin wired (conftest + HTTPBIN_URL) for %s" % repo_name, flush=True)
        return True
    except Exception as e:
        print(" -- local httpbin wiring failed:", e, flush=True)
        return False


def score(inst, repo, env_vars, env_kind="uv"):
    """Apply the model's diff + the test patch, run FAIL_TO_PASS."""
    # .hypothesis dirs left by phase-1 test runs turn a UserWarning into a
    # collection ERROR (astropy makes warnings fatal) and produced a false
    # resolved=False on astropy-14995 (manual rescore: FTP + 6 P2P all pass).
    shutil.rmtree(os.path.join(repo, ".hypothesis"), ignore_errors=True)
    # Warnings-as-errors repos: pin era-compatible pure-python deps so an
    # unrelated DeprecationWarning cannot turn collection into a false negative.
    pin_warn_as_error_deps(repo, inst["repo"], env_kind, env_vars)
    diff = sh(f"git -C {repo} diff", timeout=60).stdout
    open(os.path.join(TRACES, inst["instance_id"] + ".patch"), "w").write(diff)
    open(os.path.join(repo, "_t.patch"), "w").write(inst["test_patch"])
    ap = sh("git apply _t.patch", cwd=repo)
    if ap.returncode != 0:
        return False, len(diff), "test patch did not apply (agent touched a test file?)"
    # Local httpbin for repos whose tests read HTTPBIN_URL (else offline
    # ConnectionError -> false negative). After the test patch so a suite-
    # provided conftest is never clobbered.
    ensure_local_httpbin(repo, inst["repo"], env_kind, env_vars)
    # One deterministic test path for everything (env kind, django runner,
    # positional node ids, ensure-pytest, missing-module reflex).
    import test_runner as _tr
    res = _tr.run_tests(repo, env_kind, inst["FAIL_TO_PASS"],
                        env_vars=env_vars, repo=inst["repo"], timeout=600,
                        log_path=os.path.join(SCORE_LOGS, inst["instance_id"] + ".log"))
    return res["ok"], len(diff), res["tail"]


def install_spec_extras(repo_dir, env_kind, env_vars, iid):
    """Install the instance's spec-declared optional TEST deps (pandas,
    matplotlib, ...) that a plain repo install does NOT pull, so importorskip-
    gated tests actually run. Sourced from ~/swe/spec_extras.json (SWE-bench
    spec packages), version-matched. Env-layer; never touches the answer."""
    import json as _json
    try:
        extras = _json.load(open(os.path.expanduser("~/swe/spec_extras.json"))).get(iid, [])
    except Exception:
        extras = []
    extras = [e for e in extras
              if not e.lower().endswith((".txt", ".yml", ".yaml", ".cfg", ".toml"))]
    if not extras:
        return []
    env_dir = ".condaenv" if env_kind == "conda" else ".venv"
    py = os.path.join(repo_dir, env_dir, "bin", "python")
    if not os.path.exists(py):
        return []
    env = os.environ.copy(); env.update(env_vars or {})
    quoted = " ".join('"%s"' % e for e in extras)
    r = subprocess.run('"%s" -m pip install --prefer-binary %s' % (py, quoted),
                       shell=True, cwd=repo_dir, capture_output=True, text=True,
                       timeout=1800, env=env)
    print(" -- spec extras (%s): %s" % ("ok" if r.returncode == 0 else "FAIL", extras), flush=True)
    return extras


def _load_repo_knowledge(repo):
    """Load the per-package knowledge base (knowledge/<repo>.md) if present."""
    if os.environ.get("DISABLE_KB"):
        return ""
    fp = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "knowledge", repo.replace("/", "__") + ".md")
    try:
        txt = open(fp, encoding="utf-8").read()
    except OSError:
        return ""
    return "PACKAGE KNOWLEDGE BASE for %s (accumulated, general; consult before guessing):\n%s" % (repo, txt[:2600])


def _archive_success(inst):
    """Before a re-run overwrites this instance's trace, preserve the prior one
    (tagged with its outcome + a timestamp) so a resolved run is never lost."""
    import shutil, json as _json, time as _time
    iid = inst["instance_id"]
    base = os.path.join(TRACES, iid + ".trace.json")
    if not os.path.isfile(base):
        return
    tag = "unknown"
    try:
        tag = "resolved" if _json.load(open(base)).get("outcome", {}).get("resolved") else "miss"
    except Exception:
        pass
    adir = os.path.join(TRACES, "archive"); os.makedirs(adir, exist_ok=True)
    stem = "%s__%s__%s" % (iid, _time.strftime("%Y%m%d_%H%M%S"), tag)
    try:
        shutil.copy2(base, os.path.join(adir, stem + ".trace.json"))
        pp = os.path.join(TRACES, iid + ".patch")
        if os.path.isfile(pp):
            shutil.copy2(pp, os.path.join(adir, stem + ".patch"))
        print(" -- archived prior trace (%s)" % tag, flush=True)
    except Exception as e:
        print(" -- trace archive failed:", e, flush=True)


def run_one(inst):
    print(f"\n=== {inst['instance_id']} ({inst['repo']}) ===", flush=True)
    t0 = time.time()
    repo = clone(inst)
    # -------- Phase 1: bootstrap --------
    b_handlers, b_state = make_bootstrap_handlers(
        repo, fail_to_pass=inst["FAIL_TO_PASS"],
        pass_to_pass=inst.get("PASS_TO_PASS"), repo=inst["repo"])
    cpu = ToolCallCPU(tools=BOOTSTRAP_TOOLS, tool2sys=BOOTSTRAP_TOOL2SYS,
                     system_prompt=BOOTSTRAP_SYSTEM_PROMPT, model=MODEL, host=HOST,
                     temperature=1.0, num_predict=NUM_PREDICT, num_ctx=NUMCTX,
                     keep_alive="24h")
    goal = (f"Set up the repository at ./ for testing. It is: {inst['repo']}. "
            f"The problem it addresses (for context, do not fix yet):\n\n"
            f"{inst['problem_statement'][:2000]}")
    # Known-green tests from the instance metadata — the principled smoke
    # choice (oracle-ish assist; disable for leaderboard-pure runs).
    p2p = (inst.get("PASS_TO_PASS") or [])[:3]
    if p2p:
        goal += ("\n\nKnown-stable tests that should already pass in a "
                 f"healthy environment (good run_smoke_test choices): {p2p}")
    pb = playbook_for(inst["repo"])
    if pb:
        goal += "\n\n" + format_playbook_context(pb)
        print(f" -- injected build playbook for {inst['repo']} "
              f"(validated {pb['validated_runs']}x)", flush=True)
    rems = remedies_for(inst["repo"])
    if rems:
        goal += "\n\n" + format_remedy_context(rems)
        print(f" -- injected {len(rems)} known remedies for {inst['repo']}", flush=True)
    _kb = _load_repo_knowledge(inst["repo"])
    if _kb:
        goal += "\n\n" + _kb
        print(f" -- injected package knowledge base for {inst['repo']}", flush=True)
    print(" -- phase 1: bootstrap --", flush=True)
    ckpt = os.path.join(TRACES, inst["instance_id"] + ".partial.json")
    def _boot_gate():
        # On declare: if the package imports but smoke hasn't run, the harness
        # verifies the env itself (auto_verify_env) so the model never burns
        # turns guessing a smoke test. Capture its diagnostic so a REJECTED
        # declare tells the model WHY (missing test dep / uncollectable suite)
        # instead of only a generic "gate not passed".
        if b_state.get("sanity_ok") and not b_state.get("smoke_ok"):
            try:
                _res = auto_verify_env(b_state, repo)
                _boot_gate.reject_detail = _auto_verify_reject_detail(_res)
            except Exception:
                _boot_gate.reject_detail = None
        ok = env_ready(b_state)
        if ok:
            _boot_gate.reject_detail = None
        return ok
    b_reason, b_msgs, b_meta = phase_run(cpu, BOOTSTRAP_TOOLS, BOOTSTRAP_TOOL2SYS,
                                          b_handlers, BOOTSTRAP_SYSTEM_PROMPT,
                                          goal, BOOTSTRAP_BUDGET,
                                          gate=_boot_gate,
                                          checkpoint=ckpt)
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
    # Corrections: install spec-declared optional test deps (pandas/matplotlib),
    # version-matched, so importorskip-gated tests run instead of silently skipping.
    install_spec_extras(repo, b_state.get("active_env_kind", "uv"), b_state["env_vars"], inst["instance_id"])
    ensure_local_httpbin(repo, inst["repo"], b_state.get("active_env_kind", "uv"), b_state["env_vars"])
    # -------- Phase 2: fix --------
    # STRICT setting: problem statement only — no FAIL_TO_PASS ids (those
    # tests mostly do not exist until the scoring test_patch is applied,
    # and leaking them is oracle information anyway).
    f_handlers, f_state = make_fix_handlers(
        repo, env_vars=b_state["env_vars"],
        env_kind=b_state.get("active_env_kind", "uv"), repo=inst["repo"])
    # New CPU instance for phase 2 — separate context, fresh system prompt.
    cpu2 = ToolCallCPU(tools=FIX_TOOLS, tool2sys=FIX_TOOL2SYS,
                       system_prompt=FIX_SYSTEM_PROMPT, model=MODEL, host=HOST,
                       temperature=1.0, num_predict=NUM_PREDICT, num_ctx=NUMCTX,
                       keep_alive="24h")
    print(" -- phase 2: fix --", flush=True)
    fix_goal = (f"Problem:\n{inst['problem_statement'][:3000]}\n\n"
                "Reproduce this bug with a failing script, fix the source, "
                "then verify your reproduction passes.")
    pats = patterns_load()
    if pats:
        fix_goal += "\n\n" + format_patterns_context(pats)
        print(f" -- injected {len(pats)} engineering patterns", flush=True)
    if _kb:
        fix_goal += "\n\n" + _kb
    f_reason, f_msgs, f_meta = phase_run(cpu2, FIX_TOOLS, FIX_TOOL2SYS,
                                          f_handlers, FIX_SYSTEM_PROMPT,
                                          fix_goal, FIX_BUDGET,
                                          gate=lambda: f_state["fix_verified"],
                                          checkpoint=ckpt)
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
    if outcome.get("resolved"):
        _archive_success(inst)
    return outcome


def _save_trace(inst, blob):
    os.makedirs(TRACES, exist_ok=True)
    # Clean completion supersedes the crash checkpoint.
    try:
        os.remove(os.path.join(TRACES, inst["instance_id"] + ".partial.json"))
    except OSError:
        pass
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
    # Batch-end hygiene (Mikey): clone once per repo (the mirror), reuse
    # checkouts during the batch, delete them ALL at the end. Mirrors make
    # recreation cheap. KEEP_WORK=1 skips (post-batch debugging/rescoring).
    if not os.environ.get("KEEP_WORK"):
        # Delete only RESOLVED instances' checkouts. Failures stay on disk —
        # we are probably going to do more work on those (Mikey).
        resolved_ids = {r["id"] for r in results if r.get("resolved")}
        freed, kept = 0, 0
        for inst in insts:
            d = os.path.join(WORK, inst["instance_id"])
            if not os.path.isdir(d):
                continue
            if inst["instance_id"] in resolved_ids:
                shutil.rmtree(d, ignore_errors=True)
                freed += 1
            else:
                kept += 1
        print(f"[cleanup] removed {freed} resolved checkouts, kept {kept} "
              f"failed ones for further work (mirrors retained)", flush=True)
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
