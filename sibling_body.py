"""SIBLING-BODY CONTAMINATION: did the model recall the neighbouring function?

MEASURED 2026-08-01 on the trace archive (622 patches, 237 resolved / 385 miss).
django-16910 resolved 3 times and missed 13. Three of the misses have an
IDENTICAL structural signature to a resolved patch -- same second loop, same
_filtered_relations walrus, same hasattr guard -- and differ only in that they
ALSO pasted the first loop of Django's OTHER method, _get_defer_select_mask.
Their comments say QuerySet.defer() where the resolved one says only().

So it is a RETRIEVAL error, not a reasoning error: two adjacent, near-identical
methods, and the model lands in the wrong one. Prevalence across the archive at
a threshold of 5 sibling-unique lines: 3.4% of misses, 0.8% of resolved -- a
4.3x enrichment. Also fires on django-15814, django-11019, django-16820.

This must WARN, never REFUSE: 2 of 237 resolved patches trip it, so copying a
sibling's unique lines is sometimes exactly right. The shared TAIL of two
similar functions is legitimate porting -- the resolved 16910 patches copy 12
and 14 lines that way. It is the lines unique to the sibling that are the tell.

Repo-only. No gold patch or FAIL_TO_PASS is read, at any point.
"""
import ast
import os
import re
import subprocess

MIN_SHARED = 5          # below this it is coincidence, not recall
NORM = re.compile(r"\s+")
DECL = r"^\s*(?:async\s+)?def\s+([A-Za-z_]\w*)\s*\("
_DECL = re.compile(DECL, re.M)


def _norm(line):
    return NORM.sub(" ", line).strip()


def _bodies(src):
    """name -> (start_line, end_line, set of normalised body lines)."""
    try:
        tree = ast.parse(src)
    except Exception:
        return {}
    lines = src.splitlines()
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            a = node.lineno
            b = getattr(node, "end_lineno", node.lineno)
            body = set(_norm(x) for x in lines[a:b]
                       if _norm(x) and not _norm(x).startswith("#"))
            out[node.name] = (a, b, body)
    return out


def _pristine(repo_dir, rel_path):
    """The file as it was at HEAD, i.e. before anything this run did."""
    try:
        r = subprocess.run(["git", "show", "HEAD:" + rel_path],
                           cwd=repo_dir, capture_output=True, text=True,
                           timeout=30)
        return r.stdout if r.returncode == 0 else ""
    except Exception:
        return ""


def check(repo_dir, rel_path, written_text, edited_line=None):
    """Did `written_text` come from a DIFFERENT function in the same file?

    Returns {} when there is nothing worth saying, so the caller injects
    nothing rather than injecting noise.
    """
    if not rel_path or not rel_path.endswith(".py") or not written_text:
        return {}
    # The model calls patch() with an absolute path about as often as a
    # repo-relative one. `git show HEAD:/abs/path` fails silently, which made
    # this whole check a no-op on its first live run. Normalise both shapes.
    if os.path.isabs(rel_path):
        try:
            rel_path = os.path.relpath(rel_path, repo_dir)
        except Exception:
            return {}
    if rel_path.startswith("./"):
        rel_path = rel_path[2:]
    if rel_path.startswith(".."):
        return {}                      # outside the repo; nothing to say
    src = _pristine(repo_dir, rel_path)
    if not src:
        return {}
    funcs = _bodies(src)
    if len(funcs) < 2:
        return {}

    written = set(_norm(l) for l in written_text.splitlines()
                  if _norm(l) and not _norm(l).startswith("#"))
    if not written:
        return {}

    # Which function is being edited? Prefer the innermost one containing the
    # edit line; fall back to whichever function the written text most
    # resembles, since a patch can land between functions.
    # A snippet that declares a function names its own target. edited_line is
    # the first differing CHARACTER, which on a whole-method rewrite can fall
    # past the method's end and into the next one -- observed live: reported
    # get_select_mask for a snippet beginning "def _get_only_select_mask".
    target = None
    _decl = _DECL.search(written_text or "")
    if _decl and _decl.group(1) in funcs:
        target = _decl.group(1)
    if target is None and edited_line:
        for name, (a, b, _body) in funcs.items():
            if a <= edited_line <= b:
                if target is None or (b - a) < (funcs[target][1] - funcs[target][0]):
                    target = name
    if target is None:
        best = 0
        for name, (_a, _b, body) in funcs.items():
            n = len(written & body)
            if n > best:
                best, target = n, name
    if target is None:
        return {}

    tbody = funcs[target][2]
    worst = None
    for name, (a, b, body) in funcs.items():
        if name == target:
            continue
        # lines that live in the sibling and NOT in the function being edited.
        # the shared tail is legitimate porting; these are the tell.
        n = len(written & (body - tbody))
        if n >= MIN_SHARED and (worst is None or n > worst[1]):
            worst = (name, n, a, b)
    if worst is None:
        return {}

    name, n, a, b = worst
    copied = sorted(written & (funcs[name][2] - tbody))
    # SHOW THE LINES, do not just count them. Measured on django-16910: the
    # general form fires on 4 of 4 comparable misses AND on both resolved
    # patches, because the resolved ones legitimately port the sibling's
    # shared tail. No threshold separates those without being fitted to this
    # one instance. Naming the actual lines makes a false alarm cheap -- the
    # model reads them, confirms they belong, and moves on.
    shown = copied[:8]
    return {
        "edited_function": target,
        "sibling": name,
        "sibling_lines": "%d-%d" % (a, b),
        "shared_unique_lines": n,
        "written_lines": len(written),
        "copied_lines": shown,
        "note": ("%d of the %d lines you just wrote into %s() appear verbatim "
                 "in %s() (same file, lines %d-%d) and nowhere in %s() before "
                 "your edit. These are two different functions. Some of those "
                 "lines may belong here and some may belong only to %s() -- "
                 "read both and decide line by line. Copied lines: %s"
                 % (n, len(written), target, name, a, b, target, name,
                    " | ".join(shown) + (" ..." if len(copied) > 8 else ""))),
    }
