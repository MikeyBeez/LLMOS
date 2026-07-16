"""spec_env.py -- reconstruct the OFFICIAL SWE-bench environment recipe at home.

WHY THIS EXISTS (measured 2026-07-15):
The official SWE-bench Docker images are built from per-(repo, version) recipes in
swebench.harness.constants.MAP_REPO_VERSION_TO_SPECS. Their `pre_install` step
PATCHES the repo's own setup.py to pin dependencies that have since drifted --
e.g. sphinx: Jinja2<3.0, markupsafe<=2.0.1, sphinxcontrib-* ceilings, alabaster.

Our home bootstrap installed against the ORIGINAL setup.py, whose requirements are
unbounded ("Jinja2>=2.3"), so pip resolved them to TODAY's versions. Measured on
sphinx-doc__sphinx-8474: our build got Jinja2 3.1.6 / markupsafe 3.0.3; the spec
requires Jinja2 3.0.3 / markupsafe 2.0.1. markupsafe 2.1 removed soft_unicode,
which old Jinja2 imports. Under a warnings-as-errors repo (sphinx, matplotlib,
astropy) the resulting DeprecationWarning is FATAL at collection: zero tests run,
and a CORRECT patch is scored a miss (7 such false negatives Docker-confirmed).

Applying the spec's pre_install BEFORE the env is built makes the home env match
the container's, so the model's own `pip install -e .[test]` resolves correctly
without the model having to rediscover the pins by hand. (Trace evidence: sphinx-7975
burned its ENTIRE bootstrap budget hand-pinning jinja2<3.0, markupsafe<2.1, babel --
exactly this list.)

LEAKAGE-SAFE: dependency versions are ENVIRONMENT metadata. Nothing here derives
from gold_patch, test_patch, or FAIL_TO_PASS, and nothing is shown to the model.

SAFETY: only local, non-root commands are executed (sed). Anything needing root or
the network (apt-get, conda, curl) is reported and SKIPPED, never run.
"""
import json, os, subprocess

_VERSIONS_PATH = os.path.expanduser("~/swe/instance_versions.json")
_versions = None
_specs = None


def _load_versions():
    global _versions
    if _versions is None:
        try:
            _versions = json.load(open(_VERSIONS_PATH))
        except Exception:
            _versions = {}
    return _versions


_SPECS_PATH = os.path.expanduser("~/swe/swebench_specs.json")


def _load_specs():
    """Read the official recipes as DATA.

    swebench is installed only in ~/swebench-venv, not the system python the
    agent runs under, so importing it at runtime silently yields {} and every
    lookup misses. The recipes are static data, so they are exported once
    (see /tmp/dump_specs.py -> ~/swe/swebench_specs.json) and just read here.
    Falls back to a live import if the export is absent.
    """
    global _specs
    if _specs is None:
        try:
            _specs = json.load(open(_SPECS_PATH))
        except Exception:
            try:
                from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS
                _specs = MAP_REPO_VERSION_TO_SPECS
            except Exception:
                _specs = {}
    return _specs


def version_for(iid):
    return _load_versions().get(iid)


def spec_for(iid, repo):
    v = version_for(iid)
    if not v:
        return None, None
    return _load_specs().get(repo, {}).get(str(v)), v


def _is_safe(cmd):
    """Only local, non-root, no-network edits. sed on repo files is the whole point."""
    c = cmd.strip()
    if c.startswith("sed "):
        return True
    return False


def apply_pre_install(repo_dir, iid, repo):
    """Run the official spec's pre_install dep-pin edits in repo_dir.

    Returns {"ok", "version", "applied": [...], "skipped": [...], "pip_pins": [...]}.
    Never raises: an env-layer assist must not be able to kill an instance.
    """
    out = {"ok": False, "version": None, "applied": [], "skipped": [], "pip_pins": []}
    try:
        spec, v = spec_for(iid, repo)
        out["version"] = v
        if not spec:
            return out
        out["pip_pins"] = [p for p in (spec.get("pip_packages") or []) if "==" in p or "<" in p]
        for cmd in (spec.get("pre_install") or []):
            if not _is_safe(cmd):
                out["skipped"].append(cmd)
                continue
            r = subprocess.run(cmd, shell=True, cwd=repo_dir, capture_output=True,
                               text=True, timeout=60)
            (out["applied"] if r.returncode == 0 else out["skipped"]).append(cmd)
        out["ok"] = bool(out["applied"])
    except Exception as e:
        out["error"] = "%s: %s" % (type(e).__name__, e)
    return out


def install_spec_pins(repo_dir, env_kind, env_vars, iid, repo):
    """Install the spec's exact pinned pip_packages into an already-built env.

    Generalizes the hardcoded WARN_AS_ERROR_DEP_PINS to spec-derived data
    (e.g. matplotlib 3.6: pyparsing==3.0.9, numpy==1.25.2, pillow==10.0.0).
    """
    spec, v = spec_for(iid, repo)
    if not spec:
        return []
    pins = [p for p in (spec.get("pip_packages") or []) if "==" in p or "<" in p]
    if not pins:
        return []
    env_dir = ".condaenv" if env_kind == "conda" else ".venv"
    py = os.path.join(repo_dir, env_dir, "bin", "python")
    if not os.path.exists(py):
        return []
    env = os.environ.copy(); env.update(env_vars or {})
    quoted = " ".join('"%s"' % p for p in pins)
    try:
        r = subprocess.run('"%s" -m pip install %s' % (py, quoted), shell=True,
                           cwd=repo_dir, capture_output=True, text=True,
                           timeout=900, env=env)
        return pins if r.returncode == 0 else []
    except Exception:
        return []


# --------------------------------------------------------------------------
# System (apt) dependencies from the spec's pre_install.
# --------------------------------------------------------------------------
# Some recipes (matplotlib) install SYSTEM libs the test-suite needs -- imagemagick,
# ffmpeg, texlive (LaTeX-rendered figures), dvipng. Their absence does not cause the
# collection-error false negatives (those are pip-level, e.g. pyparsing), but it does
# fail image/animation/LaTeX tests, so it is real env fidelity.
#
# TWO DELIBERATE DEVIATIONS FROM THE RECIPE, because the container is disposable and
# this machine is NOT:
#   1. `apt-get upgrade` is NEVER run. A system-wide upgrade on a real workstation can
#      take the NVIDIA/CUDA stack (and the llama-server the agent depends on) with it.
#      The recipe only wants the INSTALLS; the upgrade is gratuitous risk here.
#   2. Non-apt pre_install steps (matplotlib's qhull download) are skipped: they are
#      separate shell lines whose variables only persist inside one container RUN, and
#      they hardcode the container path /testbed. Running them here would litter the
#      host and accomplish nothing. matplotlib builds fine without them.
import re as _re

_APT_INSTALL_RE = _re.compile(r"apt-get\s+(?:-y\s+)?install\s+(?:-y\s+)?([^&|;]+)")
_APT_MARKER_DIR = os.path.expanduser("~/swe/.spec_apt_done")


def _apt_packages(spec):
    pkgs = []
    for cmd in (spec.get("pre_install") or []):
        if "apt-get" not in cmd:
            continue
        for m in _APT_INSTALL_RE.finditer(cmd):
            for tok in m.group(1).split():
                if tok.startswith("-") or "=" in tok:
                    continue
                pkgs.append(tok)
    return sorted(set(pkgs))


def _installed(pkg):
    try:
        r = subprocess.run(["dpkg", "-s", pkg], capture_output=True, text=True, timeout=20)
        return "Status: install ok installed" in r.stdout
    except Exception:
        return False


def apply_system_deps(iid, repo):
    """Install the spec's apt packages (install-only, idempotent, no upgrade).

    Cheap no-op once satisfied: only missing packages are installed. Never raises.
    """
    out = {"ok": False, "needed": [], "installed": [], "failed": []}
    try:
        spec, v = spec_for(iid, repo)
        if not spec:
            return out
        pkgs = _apt_packages(spec)
        if not pkgs:
            return out
        missing = [p for p in pkgs if not _installed(p)]
        out["needed"] = missing
        if not missing:
            out["ok"] = True
            return out
        if subprocess.run(["sudo", "-n", "true"], capture_output=True).returncode != 0:
            out["failed"] = missing
            return out
        subprocess.run(["sudo", "-n", "apt-get", "-y", "update"],
                       capture_output=True, text=True, timeout=600)
        env = os.environ.copy(); env["DEBIAN_FRONTEND"] = "noninteractive"
        r = subprocess.run(["sudo", "-n", "apt-get", "install", "-y"] + missing,
                           capture_output=True, text=True, timeout=3600, env=env)
        if r.returncode == 0:
            out["installed"] = missing; out["ok"] = True
        else:
            out["failed"] = missing
        os.makedirs(_APT_MARKER_DIR, exist_ok=True)
        open(os.path.join(_APT_MARKER_DIR, "%s_%s" % (repo.replace("/", "__"), v)), "w").write("done")
    except Exception as e:
        out["error"] = "%s: %s" % (type(e).__name__, e)
    return out
