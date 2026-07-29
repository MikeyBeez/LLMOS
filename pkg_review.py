#!/usr/bin/env python3
"""Review a third-party Python package before you trust it.

Mikey's practice, written down so it takes a minute instead of an afternoon:
fork it, clone it, prove the wheel you installed is the source you read, grep
for the shapes that can only be there on purpose, and audit the dependency
tree.

    python3 pkg_review.py graphifyy \
        --venv ~/.graphify_venv \
        --repo https://github.com/Graphify-Labs/graphify \
        --fork MikeyBeez/graphify

WHAT EACH STEP IS ACTUALLY FOR

  1. FORK + CLONE. Not a security step on its own -- it is what makes the rest
     possible. You cannot diff against source you do not have, and you cannot
     patch a dependency you do not control.

  2. WHEEL vs REPO. The single highest-value check, and the one almost nobody
     runs. The repo is what the world reviews; the wheel is what executes on
     your machine. They are uploaded by different processes and there is no
     rule that they match. A file present in the WHEEL but ABSENT from the
     REPO is the finding -- that is code nobody reviewed. Files present only
     in the repo (tests, docs, CI) are normal and expected.

  3. DANGER SHAPES. Not "is this package malicious" -- that question has no
     mechanical answer. This asks "does it do things that are hard to do by
     accident": exec of decoded bytes, network calls at import time,
     subprocess with a shell, pickle of remote data, install-time hooks.
     Every hit is a FACT WITH A LINE NUMBER for a human to read, never a
     verdict. Most hits in a legitimate package are legitimate.

  4. IMPORT-TIME side effects. Everything in 3 matters far more if it runs on
     `import`, because then merely depending on the package executes it. This
     step separates module-level code from code inside a function.

  5. DEP AUDIT. Known CVEs in the resolved tree, via pip-audit and the PyPI
     advisory database. Note that findings in the venv's own bundled `pip` are
     about your venv, not about the package under review.

WHAT THIS DOES NOT DO

  It does not sandbox, run, or dynamically trace the package. It does not
  detect a compromise that lives in a transitive dependency's own wheel --
  point it at that dependency to check that. And it will not catch a clever
  attacker; it catches the ordinary case, which is what the ordinary case
  deserves.
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile

# ---------------------------------------------------------------------------
# Shapes that are hard to write by accident. Each entry: (label, regex, why).
# These are SIGNALS, not verdicts. A legitimate package trips several.
DANGER_SHAPES = [
    ("exec-of-data",
     r"\b(?:exec|eval)\s*\(\s*(?:base64|codecs|zlib|bytes|bytearray|marshal|"
     r"binascii|_?decode)",
     "executing decoded/decompressed bytes -- the classic packed payload"),
    ("exec-any",
     r"\b(?:exec|eval)\s*\(",
     "dynamic execution; check what the argument is"),
    ("shell-true",
     r"subprocess\.(?:run|call|check_output|check_call|Popen)\([^)]*shell\s*=\s*True",
     "a shell parses the command string; any interpolated value is injectable"),
    ("os-system",
     r"\bos\.(?:system|popen)\s*\(",
     "same as shell-true, older spelling"),
    ("pickle-load",
     r"\b(?:pickle|cPickle|dill|joblib)\.loads?\s*\(",
     "unpickling untrusted bytes is arbitrary code execution"),
    ("net-fetch",
     r"\b(?:urllib\.request\.urlopen|requests\.(?:get|post|put)|httpx\.|"
     r"urlopen|socket\.(?:socket|create_connection))\s*\(",
     "outbound network; check the destination and when it fires"),
    ("download-pipe-shell",
     r"(?:curl|wget)\s+[^\n]*\|\s*(?:ba)?sh",
     "download-and-execute"),
    ("env-exfil",
     r"os\.environ(?:\.get\()?\s*\[?['\"](?:AWS_|GITHUB_TOKEN|SSH_|"
     r"OPENAI_API_KEY|ANTHROPIC_API_KEY|NPM_TOKEN|PYPI)",
     "reads credentials; fine if used locally, not fine if sent anywhere"),
    ("home-write",
     r"(?:os\.path\.expanduser|Path\.home\(\))[^\n]*(?:\.ssh|\.aws|"
     r"\.gnupg|authorized_keys|\.bashrc|\.zshrc|\.profile)",
     "touching credential or shell-startup files"),
    ("install-hook",
     r"class\s+\w+\((?:install|develop|egg_info|build_py)\)|cmdclass\s*=",
     "setup.py install-time hook -- runs during `pip install`"),
]

WHEEL_ONLY_EXEMPT = re.compile(
    r"(?:\.dist-info/|__pycache__/|\.pyc$|/_version\.py$|/version\.py$)")


def sh(argv, cwd=None, timeout=600):
    """Run argv (never a shell string) and return (rc, out)."""
    try:
        r = subprocess.run(argv, cwd=cwd, capture_output=True, text=True,
                           timeout=timeout)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except (OSError, subprocess.SubprocessError) as e:
        return 127, str(e)


def find_installed(venv, import_name):
    for base in ("lib", "lib64"):
        root = os.path.join(os.path.expanduser(venv), base)
        if not os.path.isdir(root):
            continue
        for py in sorted(os.listdir(root)):
            cand = os.path.join(root, py, "site-packages", import_name)
            if os.path.isdir(cand):
                return cand
    return None


def rel_files(root):
    out = set()
    for dirpath, _dirs, files in os.walk(root):
        if "__pycache__" in dirpath:
            continue
        for f in files:
            if f.endswith(".pyc"):
                continue
            out.add(os.path.relpath(os.path.join(dirpath, f), root))
    return out


def step_clone(repo_url, fork, workdir):
    print("\n=== 1. fork + clone " + "=" * 50)
    url = ("git@github.com:%s.git" % fork) if fork else repo_url
    dest = os.path.join(workdir, "src")
    if os.path.isdir(os.path.join(dest, ".git")):
        print("  already cloned: %s" % dest)
    else:
        rc, out = sh(["git", "clone", "--quiet", url, dest])
        if rc != 0:
            print("  CLONE FAILED: %s" % out.strip()[:300])
            return None
        print("  cloned %s -> %s" % (url, dest))
    _rc, head = sh(["git", "log", "--oneline", "-1"], cwd=dest)
    print("  HEAD: %s" % head.strip())
    if fork and repo_url:
        sh(["git", "remote", "add", "upstream", repo_url], cwd=dest)
        sh(["git", "fetch", "--quiet", "upstream"], cwd=dest)
        for branch in ("upstream/main", "upstream/master"):
            rc, n = sh(["git", "rev-list", "--count", "HEAD..%s" % branch],
                       cwd=dest)
            if rc == 0 and n.strip().isdigit():
                print("  fork is %s commits behind %s" % (n.strip(), branch))
                break
    return dest


def step_wheel_vs_repo(installed, repo_src):
    print("\n=== 2. wheel vs repo " + "=" * 50)
    if not (installed and repo_src and os.path.isdir(repo_src)):
        print("  SKIPPED (missing installed dir or repo source dir)")
        return None
    inst, src = rel_files(installed), rel_files(repo_src)
    wheel_only = sorted(f for f in inst - src
                        if not WHEEL_ONLY_EXEMPT.search(f))
    repo_only = sorted(src - inst)
    differing = []
    for f in sorted(inst & src):
        a, b = os.path.join(installed, f), os.path.join(repo_src, f)
        try:
            with open(a, "rb") as fa, open(b, "rb") as fb:
                if fa.read() != fb.read():
                    differing.append(f)
        except OSError:
            differing.append(f + "  (unreadable)")
    print("  installed: %d files    repo: %d files" % (len(inst), len(src)))
    print("  IN WHEEL BUT NOT IN REPO: %d   <-- the finding that matters"
          % len(wheel_only))
    for f in wheel_only[:40]:
        print("      %s" % f)
    print("  same path, different bytes: %d" % len(differing))
    for f in differing[:40]:
        print("      %s" % f)
    print("  in repo only (tests/docs/CI -- normal): %d" % len(repo_only))
    for f in repo_only[:10]:
        print("      %s" % f)
    verdict = ("MATCHES REPO" if not wheel_only and not differing
               else "DIVERGES FROM REPO -- read the files listed above")
    print("  verdict: %s" % verdict)
    return verdict


def step_danger(installed):
    print("\n=== 3. danger shapes " + "=" * 50)
    if not installed:
        print("  SKIPPED")
        return {}
    counts = {}
    hits = []
    for dirpath, _dirs, files in os.walk(installed):
        if "__pycache__" in dirpath:
            continue
        for f in files:
            if not f.endswith(".py"):
                continue
            path = os.path.join(dirpath, f)
            try:
                with open(path, encoding="utf-8", errors="ignore") as fh:
                    lines = fh.read().splitlines()
            except OSError:
                continue
            for i, line in enumerate(lines, 1):
                for label, pat, _why in DANGER_SHAPES:
                    if re.search(pat, line):
                        counts[label] = counts.get(label, 0) + 1
                        at_import = not line[:1].isspace()
                        hits.append((label, os.path.relpath(path, installed),
                                     i, at_import, line.strip()[:110]))
    for label, _pat, why in DANGER_SHAPES:
        n = counts.get(label, 0)
        print("  %-22s %4d   %s" % (label, n, why if n else ""))
    at_import = [h for h in hits if h[3]]
    print("\n  --- of those, %d are at MODULE level (run on import) ---"
          % len(at_import))
    for label, path, ln, _ai, text in at_import[:30]:
        print("      [%s] %s:%d  %s" % (label, path, ln, text))
    print("\n  --- every hit, file:line (read these; most are legitimate) ---")
    for label, path, ln, _ai, text in hits[:60]:
        print("      [%s] %s:%d  %s" % (label, path, ln, text))
    if len(hits) > 60:
        print("      ... and %d more" % (len(hits) - 60))
    return counts


def step_audit(venv, audit_venv):
    print("\n=== 4. dependency audit " + "=" * 50)
    pip = os.path.join(os.path.expanduser(venv), "bin", "pip")
    if not os.path.isfile(pip):
        print("  SKIPPED (no pip at %s)" % pip)
        return
    rc, freeze = sh([pip, "list", "--format=freeze"])
    if rc != 0:
        print("  could not freeze: %s" % freeze[:200])
        return
    pkgs = [l for l in freeze.splitlines() if l.strip()]
    print("  %d packages in the tree" % len(pkgs))
    fd, path = tempfile.mkstemp(suffix=".txt")
    with os.fdopen(fd, "w") as fh:
        fh.write("\n".join(pkgs) + "\n")
    pa = os.path.join(os.path.expanduser(audit_venv), "bin", "pip-audit")
    if not os.path.isfile(pa):
        print("  pip-audit not found at %s -- create it with:" % pa)
        print("     python3 -m venv %s && %s/bin/pip install pip-audit"
              % (audit_venv, audit_venv))
        os.unlink(path)
        return
    _rc, out = sh([pa, "-r", path, "--progress-spinner", "off"], timeout=900)
    os.unlink(path)
    print("\n".join("  " + l for l in out.strip().splitlines()[:40]))
    if re.search(r"^pip\s", out, re.M):
        print("\n  NOTE: findings in `pip` itself are about YOUR VENV, not the")
        print("        package under review. Fix: %s install -U pip" % pip)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("package", help="pip / import name, e.g. graphifyy")
    ap.add_argument("--import-name", default=None,
                    help="import name if it differs from the pip name")
    ap.add_argument("--venv", required=True,
                    help="venv the package is installed in")
    ap.add_argument("--repo", default=None, help="upstream repo URL")
    ap.add_argument("--fork", default=None, help="owner/name of your fork")
    ap.add_argument("--workdir", default=None,
                    help="where to clone (default ~/Code/<name>-review)")
    ap.add_argument("--audit-venv", default="~/.audit_venv")
    ap.add_argument("--src-subdir", default=None,
                    help="package dir inside the repo (default: import name)")
    a = ap.parse_args()

    imp = a.import_name or a.package
    workdir = os.path.expanduser(
        a.workdir or os.path.join("~/Code", "%s-review" % a.package))
    os.makedirs(workdir, exist_ok=True)

    print("=" * 72)
    print("PACKAGE REVIEW: %s   (import %s)" % (a.package, imp))
    print("=" * 72)

    installed = find_installed(a.venv, imp)
    print("installed at: %s" % (installed or "NOT FOUND"))

    repo = step_clone(a.repo, a.fork, workdir) if (a.repo or a.fork) else None
    repo_src = os.path.join(repo, a.src_subdir or imp) if repo else None
    step_wheel_vs_repo(installed, repo_src)
    step_danger(installed)
    step_audit(a.venv, a.audit_venv)

    print("\n" + "=" * 72)
    print("Read the module-level hits and any wheel-only files. Nothing above")
    print("is a verdict; the verdict is yours.")
    print("=" * 72)


if __name__ == "__main__":
    sys.exit(main())
