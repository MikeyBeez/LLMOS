"""test_runner — the ONE deterministic way to run tests in a repo checkout.

Every test invocation in LLMOS (env smoke check, phase-2 regression check,
verify_fix, final scoring) goes through here, so the behavior is defined
and fixed in exactly one place. This module is pure/deterministic: no model
calls. Rationale (Mikey, 2026-07-10): the test-running logic had been
copied into four handlers and drifted (pytest ensured in score() but not
run_tests, -k selection bug in score() only) — consolidate it.

Guarantees on every run:
  * pytest present (ensurepip + pip install pytest — the always-works path)
  * .hypothesis purged (its warnings become collection errors)
  * node ids passed POSITIONALLY (never -k, which deselects path::node ids)
  * django/django uses tests/runtests.py (unittest), not pytest
  * missing external module -> install once and retry (name via alias map)
"""
import os, re, subprocess

_PKG_ALIASES = {
    "cv2": "opencv-python", "yaml": "pyyaml", "PIL": "pillow",
    "sklearn": "scikit-learn", "bs4": "beautifulsoup4", "OpenSSL": "pyopenssl",
    "dateutil": "python-dateutil", "attr": "attrs",
}
_MISSING_RE = re.compile(r"No module named ['\"]([\w.]+)['\"]")


def _bin(repo_dir, kind):
    return os.path.join(repo_dir, ".condaenv" if kind == "conda" else ".venv",
                        "bin")


def _env(repo_dir, kind, env_vars):
    env = os.environ.copy()
    env.update(env_vars or {})
    b = _bin(repo_dir, kind)
    env["PATH"] = b + ":" + env.get("PATH", "")
    root = os.path.dirname(b)
    env["CONDA_PREFIX" if kind == "conda" else "VIRTUAL_ENV"] = root
    return env


def ensure_pytest(repo_dir, kind, env=None):
    """Canonical always-works pytest install."""
    py = os.path.join(_bin(repo_dir, kind), "python")
    if not os.path.isfile(py):
        return False
    if subprocess.run([py, "-c", "import pytest"],
                      capture_output=True).returncode == 0:
        return True
    subprocess.run([py, "-m", "ensurepip", "--upgrade"], cwd=repo_dir,
                   capture_output=True, timeout=180, env=env)
    subprocess.run([py, "-m", "pip", "install", "pytest", "-q"], cwd=repo_dir,
                   capture_output=True, timeout=300, env=env)
    return subprocess.run([py, "-c", "import pytest"],
                          capture_output=True).returncode == 0


def _run(cmd, repo_dir, env, timeout=600):
    return subprocess.run(cmd, shell=True, cwd=repo_dir, capture_output=True,
                          text=True, timeout=timeout, env=env)


def collect_ids(repo_dir, kind, env_vars=None, path="", exclude=None):
    """Return the real, currently-present test node ids (pytest --collect-only).
    `exclude` node-substrings are dropped. Deterministic ground truth for what
    can actually run in this tree right now."""
    env = _env(repo_dir, kind, env_vars)
    ensure_pytest(repo_dir, kind, env)
    py = f"{_bin_rel(kind)}/python"
    r = _run(f'{py} -m pytest --collect-only -q -p no:cacheprovider {path}',
             repo_dir, env, timeout=240)
    ids = []
    for line in (r.stdout or "").splitlines():
        line = line.strip()
        if "::" not in line or line.startswith(("<", "=", "_", " ")):
            continue
        nid = line.split(" ")[0]
        if exclude and any(x in nid for x in exclude):
            continue
        ids.append(nid)
    return ids


def _bin_rel(kind):
    return ".condaenv/bin" if kind == "conda" else ".venv/bin"


def run_tests(repo_dir, kind, node_ids, env_vars=None, repo=None,
              timeout=600, max_installs=4):
    """Run the given test node ids and report pass/fail. THE single test
    execution path. Returns dict: ok, exit, passed, tail, installed."""
    env = _env(repo_dir, kind, env_vars)
    ensure_pytest(repo_dir, kind, env)
    subprocess.run("rm -rf .hypothesis", shell=True, cwd=repo_dir,
                   capture_output=True)
    py = f"{_bin_rel(kind)}/python"
    ids = node_ids if isinstance(node_ids, (list, tuple)) else [node_ids]

    if repo == "django/django" and os.path.isfile(
            os.path.join(repo_dir, "tests/runtests.py")):
        labels = " ".join(_django_label(t) for t in ids)
        cmd = f'{py} tests/runtests.py {labels} -v 0'
    else:
        nodes = " ".join(f'"{t}"' for t in ids)   # POSITIONAL, never -k
        cmd = f'{py} -m pytest {nodes} -p no:cacheprovider -q --no-header'

    installed = []
    tried = set()
    for _ in range(max_installs + 1):
        r = _run(cmd, repo_dir, env, timeout)
        out = (r.stdout or "") + (r.stderr or "")
        m = _MISSING_RE.search(out)
        if not m:
            break
        mod = m.group(1)
        if mod in tried:
            break
        tried.add(mod)
        pkg = _PKG_ALIASES.get(mod, mod.split(".")[0])
        ok_i = subprocess.run(
            f'{py} -m pip install "{pkg}"', shell=True, cwd=repo_dir,
            capture_output=True, text=True, timeout=300, env=env).returncode == 0
        if not ok_i:
            break
        installed.append(pkg)

    out = (r.stdout or "") + (r.stderr or "")
    passed = ("passed" in out) or (repo == "django/django" and "OK" in out
                                   and "FAILED" not in out and r.returncode == 0)
    ok = r.returncode == 0 and passed
    tail = out.strip().splitlines()[-1][:160] if out.strip() else "(no output)"
    return {"ok": ok, "exit": r.returncode, "passed": passed,
            "tail": tail, "stdout": (r.stdout or "")[-1500:],
            "installed": installed}


def _django_label(node_id):
    """django FAIL_TO_PASS -> runtests label. SWE-bench gives django ids in
    unittest verbose form 'method (dotted.path.Class.method)'; the runnable
    label is the dotted path inside the parens. Fallback: pytest path form."""
    m = re.search(r"\(([^)]+)\)", node_id)
    if m:
        return m.group(1).strip()
    part = node_id.split("::")
    mod = part[0].replace("tests/", "").replace("/", ".")
    mod = mod[:-3] if mod.endswith(".py") else mod
    return ".".join([mod] + part[1:])
