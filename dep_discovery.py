"""Mechanical discovery of how a repository declares its dependencies.

MEASURED MOTIVATION (2026-07-28). Across every run ever recorded, 48 of 107
install failures were the model trying to install from a requirements FILE:
    tests/requirements/py3.txt      27 failed, 0 succeeded
    -r tests/requirements/py3.txt   11 failed
    -r requirements/py3.txt          4 failed
    requirements_test.txt            2 failed
install_package only accepted a package NAME, so every attempt failed -- and the
model kept trying, because wanting `pip install -r <the repo's own test deps>`
is exactly right. 45% of our install failures were a capability we did not have.

AND CODING TO `requirements.txt` WOULD HAVE BEEN NEARLY USELESS. Counted across
our 300 checkouts:
    tests/requirements/                     50
    docs/requirements.txt                   42
    requirements/                           26
    requirements/dev/dev-requirements.txt   23
    doc/en/requirements.txt                 17
    requirements.txt                        10   <-- the obvious convention
Worse, most repos do not use a requirements file at all: setup.cfg appears in
188 checkouts, setup.py in 186, tox.ini in 119, pyproject.toml in 86. Test deps
usually live in packaging metadata (extras_require / optional-dependencies /
tox deps).

So: DISCOVER, do not assume. Deterministic, no LLM calls, no network.
"""
import configparser
import os
import re

try:
    import tomllib                      # py3.11+
except Exception:                        # pragma: no cover
    tomllib = None

_SKIP_DIRS = {".git", ".venv", ".condaenv", "node_modules", "__pycache__",
              "build", "dist", ".tox", ".eggs"}
_REQ_RE = re.compile(r"requirements?.*\.txt$|^requirements\.txt$", re.I)


def _walk(repo, max_depth=4):
    base = repo.rstrip("/").count("/")
    for root, dirs, files in os.walk(repo):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS and not d.startswith(".")]
        if root.count("/") - base >= max_depth:
            dirs[:] = []
        yield root, files


# Extensions a requirements file never has. Measured across all 300 SWE-bench
# Lite checkouts on 2026-08-27: relaxing the ".txt" rule to catch astropy's
# extensionless "pip-requirements" ALSO catches 15 copies of
# "update_requirements.sh", which is a shell script. Handing the model a
# shell script as a requirements file is worse than missing one, because it
# will try to install from it. So the rule loosens the EXTENSION and adds an
# explicit exclusion rather than matching on the word alone.
_NOT_REQ_EXT = (".sh", ".bash", ".py", ".pyc", ".cfg", ".ini", ".toml",
                ".yml", ".yaml", ".json", ".md", ".rst", ".lock", ".bat")


def requirements_files(repo, limit=25):
    """Every requirements-style file, relative to the repo root.

    NOT just *.txt. astropy's 2018-era checkouts name theirs "pip-requirements"
    with no extension at all, and its contents are exactly the two lines the
    bootstrap needed ("numpy>=1.10.0", "pytest>=3.1"). Because nothing matched
    it, install_deps came back EMPTY, astropy-6938 spent its whole 720s
    bootstrap budget failing to import numpy, and the run recorded a
    capability failure for what was a discovery failure. Filed 2026-08-24,
    fixed 2026-08-27. Affects 4 of the 300 checkouts (all astropy).

    Also accepts .in (pip-compile's source format) and .pip.
    """
    out = []
    for root, files in _walk(repo):
        for f in files:
            low = f.lower()
            if "requirement" not in low:
                continue
            if low.endswith(_NOT_REQ_EXT):
                continue
            out.append(os.path.relpath(os.path.join(root, f), repo))
    # test/dev deps first -- that is what the agent almost always wants
    def rank(p):
        low = p.lower()
        return (0 if ("test" in low or "dev" in low) else
                2 if ("doc" in low or "binder" in low) else 1, len(p))
    return sorted(set(out), key=rank)[:limit]


def pyproject_extras(repo):
    p = os.path.join(repo, "pyproject.toml")
    if not (tomllib and os.path.isfile(p)):
        return []
    try:
        with open(p, "rb") as fh:
            data = tomllib.load(fh)
    except Exception:
        return []
    return sorted((data.get("project") or {}).get("optional-dependencies", {}).keys())


def setup_cfg_extras(repo):
    p = os.path.join(repo, "setup.cfg")
    if not os.path.isfile(p):
        return []
    cp = configparser.ConfigParser(strict=False)
    try:
        cp.read(p, encoding="utf-8")
    except Exception:
        return []
    return sorted(cp["options.extras_require"].keys()) if cp.has_section(
        "options.extras_require") else []


def setup_py_extras(repo):
    """Regex, not import -- executing a repo's setup.py to read it would be
    both a security hole and frequently broken at these historical commits."""
    p = os.path.join(repo, "setup.py")
    if not os.path.isfile(p):
        return []
    try:
        src = open(p, encoding="utf-8", errors="ignore").read()
    except Exception:
        return []
    m = re.search(r"extras_require\s*=\s*\{(.{0,4000}?)\}", src, re.S)
    return sorted(set(re.findall(r"[\"']([\w.-]+)[\"']\s*:", m.group(1)))) if m else []


def tox_envs(repo):
    p = os.path.join(repo, "tox.ini")
    if not os.path.isfile(p):
        return []
    cp = configparser.ConfigParser(strict=False)
    try:
        cp.read(p, encoding="utf-8")
    except Exception:
        return []
    return sorted(s for s in cp.sections() if s.startswith("testenv"))


def discover(repo):
    return {
        "requirements_files": requirements_files(repo),
        "pyproject_extras":   pyproject_extras(repo),
        "setup_cfg_extras":   setup_cfg_extras(repo),
        "setup_py_extras":    setup_py_extras(repo),
        "tox_envs":           tox_envs(repo),
    }


def format_report(d):
    """Facts with paths, never advice -- the model decides what to install."""
    if not any(d.values()):
        return "no dependency declarations found in this repo"
    out = []
    if d["requirements_files"]:
        out.append("requirements files (install with "
                   "install_package(requirements_file=...)):")
        out += ["  " + f for f in d["requirements_files"]]
    for key, label in (("pyproject_extras", "pyproject.toml optional-dependencies"),
                       ("setup_cfg_extras", "setup.cfg extras_require"),
                       ("setup_py_extras",  "setup.py extras_require")):
        if d[key]:
            out.append("%s (install with install_repo_editable(extras=[...])):" % label)
            out.append("  " + ", ".join(d[key]))
    if d["tox_envs"]:
        out.append("tox environments: " + ", ".join(d["tox_envs"]))
    return "\n".join(out)
