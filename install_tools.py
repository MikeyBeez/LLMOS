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
import os, shutil, subprocess


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


# ---- goal stack -------------------------------------------------------

def _stack_snapshot(state):
    """Human-readable one-liner of the current goal stack — attached to
    every tool result so the model always sees where it is in the tree."""
    stack = state.get("goal_stack", [])
    if not stack:
        return "no active subgoals"
    return " > ".join(g["reason"] for g in stack)


# ---- handlers ---------------------------------------------------------

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
        backend = str(args.get("backend", "uv"))
        if backend not in ("uv", "conda"):
            return {"error": f"backend must be 'uv' or 'conda', got {backend!r}",
                    "goal_stack": _stack_snapshot(state)}
        # Wipe any existing env of either kind so re-creation is clean.
        shutil.rmtree(os.path.join(repo_dir, ".venv"),     ignore_errors=True)
        shutil.rmtree(os.path.join(repo_dir, ".condaenv"), ignore_errors=True)
        if backend == "uv":
            r = _run(f"{UV} venv --python {pyv} .venv", repo_dir, timeout=180,
                     active_env_kind=None)
        else:
            # micromamba: single command creates the env, installs python,
            # activates conda-forge as the primary channel.
            r = _run(f'{MAMBA} create -y -p .condaenv -c conda-forge '
                     f'"python={pyv}" pip setuptools wheel', repo_dir, timeout=600,
                     active_env_kind=None)
        ok = r.returncode == 0
        if ok:
            state["active_env_kind"] = backend
            state["python_version"]  = pyv
            state["installed"]       = []
            state["sanity_ok"]       = False
            state["smoke_ok"]        = False
            state["repo_installed"]  = False
        return {"ok": ok, "backend": backend, "python_version": pyv,
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
        if not name:
            return {"error": "name is required",
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
        ok = r.returncode == 0
        entry = {"name": name, "version_spec": vspec, "backend": backend,
                 "no_build_isolation": no_iso, "ok": ok}
        state["installed"].append(entry)
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

    handlers = {
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
    "install_package":        "install.install_package",
    "install_repo_editable":  "install.install_repo_editable",
    "push_subgoal":           "install.push_subgoal",
    "pop_subgoal":            "install.pop_subgoal",
    "current_goal":           "install.current_goal",
    "set_env_var":            "install.set_env_var",
}
