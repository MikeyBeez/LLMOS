"""Version checker + environment builder.

Most 'build failures' are version failures: an old repo written for Python 3.6 will
not import on 3.12 (distutils gone, collections.Mapping gone, ...), which silently turns
a correct patch into an unscorable 'error'. This module picks the right Python for a
repo, provisions it with uv (which downloads standalone CPython on demand -- no Docker,
no sudo), and steps down a version on known incompatibility signatures.

    py = pick_python(repo_dir)                 # inspect declared support -> "3.9"
    venv_py, ver = build_venv(repo_dir, "mpmath pytest")   # provision via uv, install
    lower = downgrade_for(stderr, ver)         # on a version-signature failure, retry
"""
import os, re, subprocess

UV = os.path.expanduser("~/.local/bin/uv")
# newest-first Pythons uv can fetch and we trust to build these repos
CANDIDATES = ["3.11", "3.10", "3.9", "3.8", "3.7", "3.6"]
DEFAULT = "3.9"          # old enough for distutils + collections.Mapping, new enough to run
CAP = "3.11"             # don't pick newer than this even if a repo "supports" it


def _run(cmd, cwd=None, t=400):
    return subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=t)


def _declared(repo_dir):
    """Read the repo's own declaration of supported Pythons from setup.py / setup.cfg /
    pyproject.toml: the 'Programming Language :: Python :: 3.X' classifiers, plus any
    python_requires. Returns (set_of_minors, combined_text)."""
    text = ""
    for name in ("setup.py", "setup.cfg", "pyproject.toml"):
        p = os.path.join(repo_dir, name)
        if os.path.isfile(p):
            try:
                text += open(p, encoding="utf-8", errors="ignore").read() + "\n"
            except OSError:
                pass
    minors = {int(m.group(1)) for m in re.finditer(r"Python\s*::\s*3\.(\d+)\b", text)}
    return minors, text


def pick_python(repo_dir, cap=CAP):
    """Choose the newest candidate Python within the repo's declared support and <= cap.
    A repo that lists up to 3.7 in its classifiers gets 3.7; one that declares nothing
    gets the safe DEFAULT."""
    minors, text = _declared(repo_dir)
    cap_minor = int(cap.split(".")[1])
    if minors:
        chosen = min(max(minors), cap_minor)
        # respect a python_requires lower bound if it is higher than what we picked
        mlo = re.search(r"python_requires\s*=\s*['\"][^'\"]*?>=\s*3\.(\d+)", text)
        if mlo:
            chosen = max(chosen, int(mlo.group(1)))
        # floor at 3.8: old repos advertise 3.5/3.6 (pre-3.7), but modern pytest/pip
        # wheels need >=3.8, and 3.8 still has distutils + collections.Mapping.
        return "3.%d" % min(max(chosen, 8), cap_minor)
    return DEFAULT


def build_venv(repo_dir, deps, py=None, t=400):
    """Provision repo/.venv with the chosen Python (via uv) and install deps. Returns
    (path_to_venv_python, chosen_version). deps is a pip-arg string."""
    py = py or pick_python(repo_dir)
    _run("%s venv --python %s .venv" % (UV, py), cwd=repo_dir, t=180)
    _run("%s pip install --python .venv/bin/python -q %s" % (UV, deps), cwd=repo_dir, t=t)
    return os.path.join(repo_dir, ".venv", "bin", "python"), py


# a version-incompatibility signature -> the highest Python minor that avoids it
_SIG_MAX = [
    (r"No module named ['\"]distutils['\"]", 11),
    (r"cannot import name ['\"]\w+['\"] from ['\"]collections['\"]", 9),
    (r"ImportError: cannot import name ['\"]\w+['\"] from ['\"]inspect['\"]", 10),
    (r"module ['\"]?time['\"]? has no attribute ['\"]clock['\"]", 7),
    (r"getargspec", 10),
    (r"'async' and 'await' are reserved", 6),
]


def downgrade_for(stderr, current):
    """Given a failed run's stderr and the current version, return a lower version to
    retry with (or None). Used as a safety net when detection guessed too new."""
    if not stderr:
        return None
    cur = int(str(current).split(".")[1])
    caps = [mx for pat, mx in _SIG_MAX if re.search(pat, stderr)]
    if caps:
        target = min(min(caps), cur - 1)
        if target >= 6:
            return "3.%d" % target
    # a bare collection/import error with no known signature: step down one minor
    if re.search(r"(ImportError|ModuleNotFoundError|error during collection)", stderr) and cur > 6:
        return "3.%d" % (cur - 1)
    return None


if __name__ == "__main__":
    import sys
    d = sys.argv[1]
    minors, _ = _declared(d)
    print("declared minors:", sorted(minors), "-> pick", pick_python(d))
