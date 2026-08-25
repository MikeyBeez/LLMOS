"""Recursive install primitives with backend routing.

Rationale (from Mikey):
  "You need a subprocess. You're starting the installation and it says you
   also need to install something else. Then install that and then get back
   to the installation."

Installations are recursive: installing X reveals Y is missing, installing
Y reveals Z, etc. A single atomic `provision_env` collapses that tree to one
shell command and the model has no way to represent "I've paused astropy to
install its build deps." This module exposes primitives that make the
sub-goal explicit — the tool tracks the goal stack; the model calls
push_subgoal, does the sub-install, calls pop_subgoal, resumes.

Backend routing:
  uv    — Mikey's default. Fast, PEP 517 build isolation by default.
  pip   — vanilla; works when uv is confused (rare) or for --no-build-
          isolation setups where you want plain pip's flag semantics.
  conda — micromamba static binary + conda-forge channel. For compiled
          scientific packages (numpy/scipy/cython/extension_helpers on
          Ubuntu, where prebuilt Linux wheels are the reason you're using
          conda in the first place).

The `active_env_kind` at state["active_env_kind"] pins which env is live:
  "uv"    -> repo/.venv/         (created by uv venv)
  "conda" -> repo/.condaenv/     (created by micromamba)

install_package(backend=X) is validated against active_env_kind — you can
mix pip/uv freely inside a uv .venv, but you can only conda-install into
a conda env. Trying to install conda pkgs into a uv .venv returns an error
telling the model to create_venv(backend="conda") first.
"""
import os, re, shutil, subprocess


UV = os.path.expanduser("~/.local/bin/uv")
MAMBA = os.path.expanduser("~/.local/bin/micromamba")


# ---- shared helpers ---------------------------------------------------

def _venv_bin(repo_dir, kind):
    """Path to the bin dir of the active env, for PATH manipulation."""
    if kind == "conda":
        return os.path.join(repo_dir, ".condaenv", "bin")
    return os.path.join(repo_dir, ".venv", "bin")


def _venv_root(repo_dir, kind):
    if kind == "conda":
        return os.path.join(repo_dir, ".condaenv")
    return os.path.join(repo_dir, ".venv")


def _run(cmd, cwd, env_vars=None, timeout=900, active_env_kind="uv"):
    """Run a shell command inside the active env's context.

    Prepends the active env's bin/ to PATH and sets VIRTUAL_ENV /
    CONDA_PREFIX so `python`, `pip`, and installed CLIs resolve to the
    env. Captures stdout+stderr, returns them + returncode."""
    env = os.environ.copy()
    if env_vars:
        env.update({str(k): str(v) for k, v in env_vars.items()})
    if active_env_kind:
        bin_dir = _venv_bin(cwd, active_env_kind)
        env["PATH"] = bin_dir + ":" + env.get("PATH", "")
        root = _venv_root(cwd, active_env_kind)
        if active_env_kind == "conda":
            env["CONDA_PREFIX"] = root
        else:
            env["VIRTUAL_ENV"] = root
    return subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True,
                          text=True, timeout=timeout, env=env)



def ensure_numpy(repo_dir, kind):
    """Guaranteed numpy in every env (Mikey, 2026-08-24), mirroring
    ensure_pytest: python -m pip from inside the env, the one way that
    always works. Pinned <2 -- numpy 2.x breaks old checkouts; pip's
    python_requires metadata picks the right 1.x for old pythons.
    Best-effort: False is recorded, never fatal (a spec pin or the
    model can still install an exact version over it)."""
    bin_dir = _venv_bin(repo_dir, kind)
    py = os.path.join(bin_dir, "python")
    if not os.path.isfile(py):
        return False
    if subprocess.run([py, "-c", "import numpy"],
                      capture_output=True).returncode == 0:
        return True
    subprocess.run([py, "-m", "ensurepip", "--upgrade"],
                   cwd=repo_dir, capture_output=True, timeout=180)
    subprocess.run([py, "-m", "pip", "install", "numpy<2", "-q"],
                   cwd=repo_dir, capture_output=True, timeout=300)
    return subprocess.run([py, "-c", "import numpy"],
                          capture_output=True).returncode == 0


def ensure_pytest(repo_dir, kind):
    """Canonical, always-works pytest install: ensurepip (bootstraps pip
    into uv venvs, idempotent on conda) then `python -m pip install pytest`
    from inside the env. Returns True if pytest imports afterwards."""
    bin_dir = _venv_bin(repo_dir, kind)
    py = os.path.join(bin_dir, "python")
    if not os.path.isfile(py):
        return False
    if subprocess.run([py, "-c", "import pytest"],
                      capture_output=True).returncode == 0:
        return True
    subprocess.run([py, "-m", "ensurepip", "--upgrade"],
                   cwd=repo_dir, capture_output=True, timeout=180)
    subprocess.run([py, "-m", "pip", "install", "pytest", "-q"],
                   cwd=repo_dir, capture_output=True, timeout=300)
    return subprocess.run([py, "-c", "import pytest"],
                          capture_output=True).returncode == 0


# ---- goal stack -------------------------------------------------------

def _stack_snapshot(state):
    """Human-readable one-liner of the current goal stack — attached to
    every tool result so the model always sees where it is in the tree."""
    stack = state.get("goal_stack", [])
    if not stack:
        return "no active subgoals"
    return " > ".join(g["reason"] for g in stack)


# ---- handlers ---------------------------------------------------------

def _net_shaped(text):
    """Does this failure look like the network went away, rather than the
    package being wrong? Only these get a retry."""
    t = (text or "").lower()
    return any(s in t for s in (
        "temporary failure in name resolution", "could not resolve host",
        "network is unreachable", "no route to host", "connection reset",
        "connection timed out", "read timed out", "connection aborted",
        "failed to establish a new connection", "max retries exceeded",
        "proxy", "ssl: wrong_version_number", "eof occurred in violation",
        "newconnectionerror", "timed out. (connect timeout"))


def _wait_for_net(max_wait=1200):
    """Block until PyPI answers again, or give up after max_wait seconds."""
    import socket as _s
    waited = 0
    while waited < max_wait:
        try:
            _s.create_connection(("pypi.org", 443), timeout=10).close()
            return True
        except OSError:
            time.sleep(30)
            waited += 30
    return False


def make_install_handlers(repo_dir, base_env_vars=None):
    """Handlers for the install primitives. Returns (handlers, state).

    state is shared with the smoke-test/sanity handlers so the env-ready
    gate can still check sanity_ok + smoke_ok."""
    state = {
        "active_env_kind":  None,          # "uv" | "conda" | None
        "python_version":   None,
        "env_vars":         dict(base_env_vars or {}),
        "goal_stack":       [],            # list of {reason, opened_turn}
        "installed":        [],            # log of successful installs
        "sanity_ok":        False,
        "smoke_ok":         False,
        "repo_installed":   False,         # True after install_repo_editable ok
    }

    # ---- create_venv --------------------------------------------------
    def h_create_venv(pcb, args):
        pyv = str(args.get("python_version", "3.11"))
        _pin = os.environ.get("PIN_PYTHON")
        if _pin:
            pyv = _pin  # SWE-bench spec-table canonical Python; pin overrides model guess
        backend = str(args.get("backend", "uv"))
        _pinb = os.environ.get("PIN_BACKEND")
        if _pinb:
            backend = _pinb  # force backend (conda for 3.6/3.7 which uv cannot build)
        if backend not in ("uv", "conda"):
            return {"error": f"backend must be 'uv' or 'conda', got {backend!r}",
                    "goal_stack": _stack_snapshot(state)}
        # Wipe any existing env of either kind so re-creation is clean.
        shutil.rmtree(os.path.join(repo_dir, ".venv"),     ignore_errors=True)
        shutil.rmtree(os.path.join(repo_dir, ".condaenv"), ignore_errors=True)
        if backend == "uv":
            # only-managed: uv downloads python-build-standalone, which SHIPS
            # dev headers (Python.h). System pythons on this box lack -dev
            # packages for anything but 3.12, which silently killed every
            # C-extension build in uv envs (v6 postmortem).
            r = _run(f"UV_PYTHON_PREFERENCE=only-managed {UV} venv --python {pyv} .venv",
                     repo_dir, timeout=300, active_env_kind=None)
        else:
            # micromamba: single command creates the env, installs python,
            # activates conda-forge as the primary channel.
            r = _run(f'{MAMBA} create -y -p ./.condaenv -c conda-forge '
                     f'"python={pyv}" pip setuptools wheel', repo_dir, timeout=600,
                     active_env_kind=None)
        ok = r.returncode == 0
        pytest_ready = False
        numpy_ready = False
        if ok:
            state["active_env_kind"] = backend
            state["python_version"]  = pyv
            state["installed"]       = []
            state["sanity_ok"]       = False
            state["smoke_ok"]        = False
            state["repo_installed"]  = False
            # Guarantee pytest in every env, the one way that always works.
            pytest_ready = ensure_pytest(repo_dir, backend)
            # Guarantee numpy too (Mikey, 2026-08-24) -- see ensure_numpy.
            numpy_ready = ensure_numpy(repo_dir, backend)
        return {"ok": ok, "backend": backend, "python_version": pyv,
                "pytest_ready": pytest_ready,
                "numpy_ready": numpy_ready,
                "exit": r.returncode,
                "stderr": (r.stderr or "")[-1500:],
                "goal_stack": _stack_snapshot(state)}

    # ---- install_package (atomic) -------------------------------------
    def h_install_package(pcb, args):
        name    = str(args.get("name", "")).strip()
        vspec   = str(args.get("version_spec", "") or "")
        backend = str(args.get("backend", "uv"))
        no_iso  = bool(args.get("no_build_isolation", False))
        channel = str(args.get("channel", "") or "conda-forge")
        reqfile = str(args.get("requirements_file", "") or "").strip()

        # REQUIREMENTS-FILE SUPPORT (2026-07-28). Measured: 48 of 107 install
        # failures across every recorded run were the model trying to install
        # from a requirements FILE -- tests/requirements/py3.txt (27 failures,
        # 0 successes), "-r tests/requirements/py3.txt" (11), and so on. The
        # instinct is CORRECT; the tool simply had no way to do it, so it
        # refused and the model kept retrying. 45% of install failures were a
        # missing capability, not a model error.
        if not reqfile and name:
            m_r = re.match(r"^-r\s+(\S+)$", name)       # "-r foo/bar.txt"
            if m_r:
                reqfile, name = m_r.group(1), ""
            elif name.endswith(".txt") and ("/" in name or "requirement" in name.lower()):
                reqfile, name = name, ""

        if reqfile:
            abs_req = os.path.realpath(os.path.join(repo_dir, reqfile))
            if not abs_req.startswith(os.path.realpath(repo_dir) + os.sep):
                return {"error": "requirements_file must be inside the repo; got %r"
                                 % reqfile, "goal_stack": _stack_snapshot(state)}
            if not os.path.isfile(abs_req):
                import dep_discovery as _dd
                return {"error": "no such requirements file: %r" % reqfile,
                        "available": _dd.discover(repo_dir)["requirements_files"],
                        "goal_stack": _stack_snapshot(state)}

        if not name and not reqfile:
            return {"error": "name is required",
                    "goal_stack": _stack_snapshot(state)}

        # A bare flag or a path in `name` is a malformed call. REDIRECT rather
        # than fail silently -- the model is asking for something real.
        if name and (name.startswith("-") or "/" in name or "\\" in name
                     or " " in name.strip()):
            import dep_discovery as _dd
            return {"error": ("%r is not a package name. To install from a "
                              "requirements file call install_package with "
                              "requirements_file=<path>." % name),
                    "available_requirements_files":
                        _dd.discover(repo_dir)["requirements_files"],
                    "goal_stack": _stack_snapshot(state)}
        active = state["active_env_kind"]
        if not active:
            return {"error": "no venv yet — call create_venv first",
                    "goal_stack": _stack_snapshot(state)}
        # Backend/env compatibility check
        if backend == "conda" and active != "conda":
            return {"error": (f"cannot conda-install into a {active} env. "
                              "Either re-create with create_venv(backend="
                              "'conda') or use backend='pip'/'uv' instead."),
                    "goal_stack": _stack_snapshot(state)}
        # Compose command
        if reqfile:
            if backend == "conda":
                return {"error": "requirements_file needs backend 'uv' or 'pip'",
                        "goal_stack": _stack_snapshot(state)}
            if backend == "pip":
                cmd = ('.%s/bin/pip install -r "%s"'
                       % ("condaenv" if active == "conda" else "venv", reqfile))
            else:
                cmd = '%s pip install --python .venv/bin/python -r "%s"' % (UV, reqfile)
            r = _run(cmd, repo_dir, env_vars=state["env_vars"], timeout=1800,
                     active_env_kind=active)
            ok = r.returncode == 0
            state.setdefault("installs", []).append(
                {"name": "-r " + reqfile, "version_spec": "", "backend": backend,
                 "no_build_isolation": False, "ok": ok})
            return {"ok": ok, "requirements_file": reqfile,
                    "stderr": "" if ok else (r.stderr or "")[-1200:],
                    "goal_stack": _stack_snapshot(state)}

        pkg = f'"{name}{vspec}"'
        if backend == "conda":
            cmd = (f'{MAMBA} install -y -p .condaenv -c {channel} '
                   f'"{name}{vspec.replace("==", "=")}"')
        elif backend == "pip":
            iso_flag = " --no-build-isolation" if no_iso else ""
            cmd = f'.{"condaenv" if active=="conda" else "venv"}/bin/pip install{iso_flag} {pkg}'
        else:  # uv
            iso_flag = " --no-build-isolation" if no_iso else ""
            py = f'--python .venv/bin/python'
            cmd = f'{UV} pip install {py}{iso_flag} {pkg}'
        r = _run(cmd, repo_dir, env_vars=state["env_vars"], timeout=900,
                 active_env_kind=active)
        # A network drop mid-bootstrap must not be recorded as a dependency
        # failure: it produces a false miss that looks exactly like a real one.
        # Retry ONLY network-shaped failures; a genuine version/build error
        # still fails immediately.
        if r.returncode != 0 and _net_shaped(r.stderr or ""):
            print("   -- install failed on a NETWORK error; waiting for the "
                  "network, then retrying once", flush=True)
            if _wait_for_net():
                r = _run(cmd, repo_dir, env_vars=state["env_vars"], timeout=900,
                         active_env_kind=active)
                state["net_retries"] = state.get("net_retries", 0) + 1
        ok = r.returncode == 0
        # CONDA VERIFY (2026-08-24, astropy-6938): micromamba can exit 0
        # while the package is invisible to the env's python. Verify the
        # DISTRIBUTION (not the module -- names differ) with the env's own
        # interpreter; a claimed success that fails verification is a
        # FAILURE, reported with what to do instead.
        conda_unverified = False
        if ok and backend == "conda":
            _vpy = os.path.join(repo_dir, ".condaenv", "bin", "python")
            if os.path.isfile(_vpy):
                _chk = subprocess.run(
                    [_vpy, "-c",
                     "import importlib.metadata as m; m.distribution(%r)"
                     % name],
                    capture_output=True, timeout=60)
                if _chk.returncode != 0:
                    ok = False
                    conda_unverified = True
        entry = {"name": name, "version_spec": vspec, "backend": backend,
                 "no_build_isolation": no_iso, "ok": ok}
        state["installed"].append(entry)
        if conda_unverified:
            return {"ok": False, "name": name, "backend": "conda",
                    "error": ("micromamba exited 0 but %r is NOT visible "
                              "to this env's python -- treat as NOT "
                              "installed. Retry with backend='pip' "
                              "(installs with the env's own pip)." % name),
                    "goal_stack": _stack_snapshot(state)}
        return {"ok": ok, "name": name, "version_spec": vspec,
                "backend": backend, "no_build_isolation": no_iso,
                "exit": r.returncode,
                "stderr": (r.stderr or "")[-1500:],
                "goal_stack": _stack_snapshot(state)}

    # ---- install_repo_editable (the outer goal) -----------------------
    def h_install_repo_editable(pcb, args):
        extras = list(args.get("extras", []) or [])
        no_iso = bool(args.get("no_build_isolation", False))
        active = state["active_env_kind"]
        if not active:
            return {"error": "no venv yet — call create_venv first",
                    "goal_stack": _stack_snapshot(state)}
        target = "." if not extras else f'".[{",".join(extras)}]"'
        # --no-build-isolation only works if the caller has pre-installed
        # numpy/cython/setuptools/etc. into the venv. That's precisely the
        # point of the goal stack — the model pushes "install build deps",
        # installs them, pops, then retries this with no_iso=True.
        if active == "conda":
            iso_flag = " --no-build-isolation" if no_iso else ""
            cmd = f'.condaenv/bin/pip install{iso_flag} -e {target}'
        else:
            iso_flag = " --no-build-isolation" if no_iso else ""
            cmd = f'{UV} pip install --python .venv/bin/python{iso_flag} -e {target}'
        r = _run(cmd, repo_dir, env_vars=state["env_vars"], timeout=1200,
                 active_env_kind=active)
        ok = r.returncode == 0
        state["repo_installed"] = ok
        if ok:
            # After the repo installs, invalidate any prior sanity/smoke —
            # they need to be re-checked against the new install.
            state["sanity_ok"] = False
            state["smoke_ok"]  = False
        return {"ok": ok, "extras": extras, "no_build_isolation": no_iso,
                "backend_env": active,
                "exit": r.returncode,
                "stderr": (r.stderr or "")[-4000:],
                "goal_stack": _stack_snapshot(state)}

    # ---- goal stack management ----------------------------------------
    def h_push_subgoal(pcb, args):
        reason = str(args.get("reason", "")).strip()
        if not reason:
            return {"error": "reason is required (one-line why you're pausing)"}
        state["goal_stack"].append({"reason": reason})
        return {"pushed": reason, "goal_stack": _stack_snapshot(state),
                "depth": len(state["goal_stack"])}

    def h_pop_subgoal(pcb, args):
        if not state["goal_stack"]:
            return {"error": "goal stack is empty — nothing to pop",
                    "goal_stack": _stack_snapshot(state)}
        popped = state["goal_stack"].pop()
        return {"popped": popped["reason"],
                "goal_stack": _stack_snapshot(state),
                "depth": len(state["goal_stack"])}

    def h_set_env_var(pcb, args):
        name  = str(args.get("name", "")).strip()
        value = str(args.get("value", ""))
        if not name:
            return {"error": "name is required",
                    "goal_stack": _stack_snapshot(state)}
        state["env_vars"][name] = value
        return {"ok": True, "set": {name: value},
                "env_vars": dict(state["env_vars"]),
                "note": ("applies to all subsequent installs, sanity checks, "
                         "smoke tests, and final test scoring"),
                "goal_stack": _stack_snapshot(state)}

    def h_current_goal(pcb, args):
        return {"active_env_kind": state["active_env_kind"],
                "python_version":  state["python_version"],
                "goal_stack":      _stack_snapshot(state),
                "depth":           len(state["goal_stack"]),
                "installed":       state["installed"][-10:],
                "repo_installed":  state["repo_installed"]}

    def h_list_dependencies(pcb, args):
        """How THIS repo declares its dependencies. Deterministic, no LLM, no
        network. Exists because assuming requirements.txt covers 10 of our 300
        checkouts while tests/requirements/ covers 50 -- and most repos declare
        test deps in packaging metadata instead."""
        import dep_discovery as _dd
        d = _dd.discover(repo_dir)
        return {"report": _dd.format_report(d), **d}

    handlers = {
        "install.list_dependencies":      h_list_dependencies,
        "install.create_venv":            h_create_venv,
        "install.install_package":        h_install_package,
        "install.install_repo_editable":  h_install_repo_editable,
        "install.push_subgoal":           h_push_subgoal,
        "install.pop_subgoal":            h_pop_subgoal,
        "install.current_goal":           h_current_goal,
        "install.set_env_var":            h_set_env_var,
    }
    return handlers, state


# ---- OpenAI tool schemas ---------------------------------------------
INSTALL_TOOLS = [
    {"type": "function", "function": {
        "name": "create_venv",
        "description": (
            "Create a fresh isolated environment at repo/.venv (backend='uv') or "
            "repo/.condaenv (backend='conda', uses micromamba + conda-forge). Wipes "
            "any prior env. Pick 'uv' by default; pick 'conda' when the repo has "
            "compiled dependencies (numpy/scipy/cython/extension_helpers on Ubuntu) "
            "that build-from-source with pip is known to fail on — the astropy install "
            "docs, for example, recommend miniforge/conda-forge for exactly this reason."),
        "parameters": {"type": "object", "properties": {
            "python_version": {"type": "string",
                                "description": "e.g. '3.11', '3.10', '3.9'"},
            "backend":        {"type": "string", "enum": ["uv", "conda"],
                                "description": "uv (default, fast) or conda (for compiled deps)"},
        }, "required": ["python_version"]}}},
    {"type": "function", "function": {
        "name": "list_dependencies",
        "description": (
            "List how THIS repo declares its dependencies: every requirements "
            "file (test/dev first), the extras_require / optional-dependencies "
            "names from setup.py, setup.cfg and pyproject.toml, and the tox "
            "environments. Call this BEFORE guessing package names -- repos "
            "rarely use a plain requirements.txt."),
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "install_package",
        "description": (
            "Install ONE package into the active env. Use this to install build "
            "prerequisites (setuptools<69, numpy<2, cython<3, extension_helpers) BEFORE "
            "install_repo_editable so PEP 517 build isolation can be disabled and the "
            "repo's own build finds them. backend='uv' is preferred for pure-Python "
            "packages; backend='conda' is required for scientific compiled packages "
            "when the active env is conda; backend='pip' as a fallback."),
        "parameters": {"type": "object", "properties": {
            "name":               {"type": "string",
                                    "description": "package name, e.g. 'setuptools'"},
            "requirements_file":  {"type": "string",
                                    "description": ("install from a requirements file in "
                                                    "the repo instead of one package, e.g. "
                                                    "'requirements_test.txt'. Get the list "
                                                    "from list_dependencies.")},
            "version_spec":       {"type": "string",
                                    "description": "e.g. '<69', '==1.24', '' for latest"},
            "backend":            {"type": "string", "enum": ["uv", "pip", "conda"],
                                    "description": "package manager to use"},
            "no_build_isolation": {"type": "boolean",
                                    "description": "pip/uv only. skip PEP 517 build isolation "
                                                   "so the build uses this venv's setuptools/etc "
                                                   "instead of pip's ephemeral build env."},
            "channel":            {"type": "string",
                                    "description": "conda channel, default conda-forge"},
        }, "required": ["name", "backend"]}}},
    {"type": "function", "function": {
        "name": "install_repo_editable",
        "description": (
            "Install the checked-out repo in editable mode (`pip install -e .[extras]`). "
            "This is the OUTER install goal — if it fails complaining about missing/"
            "incompatible build deps, DO NOT retry blindly: push_subgoal, install the "
            "specific build deps with install_package, pop_subgoal, then call this "
            "again with no_build_isolation=True."),
        "parameters": {"type": "object", "properties": {
            "extras":              {"type": "array", "items": {"type": "string"},
                                     "description": "e.g. ['test'] or ['test','docs']"},
            "no_build_isolation":  {"type": "boolean",
                                     "description": "set True after you've pre-installed the "
                                                    "build deps (setuptools, numpy, cython, etc.) "
                                                    "into this venv with install_package."},
        }, "required": []}}},
    {"type": "function", "function": {
        "name": "push_subgoal",
        "description": (
            "Explicitly note that you're pausing the current install to work on a "
            "prerequisite. The stack is visible in every tool result under 'goal_stack'. "
            "Example reason: 'install setuptools<69 as build dep for astropy'."),
        "parameters": {"type": "object", "properties": {
            "reason": {"type": "string"},
        }, "required": ["reason"]}}},
    {"type": "function", "function": {
        "name": "pop_subgoal",
        "description": (
            "Pop the top subgoal — signals the sub-install is done and you're returning "
            "to the outer goal. Call this after the last install_package in a sub-tree."),
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "set_env_var",
        "description": (
            "Set an environment variable that applies to ALL subsequent builds, "
            "installs, sanity checks and test runs. Use when a build fails on "
            "COMPILER errors rather than missing packages — e.g. a C extension "
            "failing with 'nested declaration', implicit-function, or other "
            "C-standard errors needs set_env_var('CFLAGS', '-std=c99') (or "
            "'-std=gnu99') and then install_repo_editable again. Also useful: "
            "LDFLAGS, CC, and package-specific vars."),
        "parameters": {"type": "object", "properties": {
            "name":  {"type": "string", "description": "e.g. 'CFLAGS'"},
            "value": {"type": "string", "description": "e.g. '-std=c99'"},
        }, "required": ["name", "value"]}}},
    {"type": "function", "function": {
        "name": "current_goal",
        "description": (
            "Show the current active env, python version, goal stack, and last 10 "
            "installs. Use when you've lost track of what's staged."),
        "parameters": {"type": "object", "properties": {}}}},
]


INSTALL_TOOL2SYS = {
    "create_venv":            "install.create_venv",
    "list_dependencies":      "install.list_dependencies",
    "install_package":        "install.install_package",
    "install_repo_editable":  "install.install_repo_editable",
    "push_subgoal":           "install.push_subgoal",
    "pop_subgoal":            "install.pop_subgoal",
    "current_goal":           "install.current_goal",
    "set_env_var":            "install.set_env_var",
}
