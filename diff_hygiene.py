"""DIFF HYGIENE: deterministic, binary, no model call.

MEASURED (xarray-5131, 2026-08-02): the model made the semantically correct
fix -- F2P PASSED -- and still graded unresolved because its diff ALSO
deleted a stray blank line and re-spaced a call, and 3 PASS_TO_PASS doctests
compare output verbatim. The bug was fixed; the diff hygiene failed it.

WARNS, never refuses: sometimes the whitespace change IS the fix (5131
itself removes one trailing space). The warning names each cosmetic change
so the model keeps the intended one and reverts the collateral.
"""
import re
import subprocess

def check_diff(diff_text):
    removed = [l[1:] for l in diff_text.splitlines()
               if l.startswith("-") and not l.startswith("---")]
    added = [l[1:] for l in diff_text.splitlines()
             if l.startswith("+") and not l.startswith("+++")]
    blank_removed = sum(1 for r in removed if not r.strip())
    blank_added = sum(1 for a in added if not a.strip())
    # v1 (collapse all whitespace) flagged 2/2 known-good controls; v2
    # (rstrip compare) still caught re-indentation because leading space
    # survives rstrip. MEASURED conclusion: content-vs-whitespace pairing
    # cannot be done reliably at diff level without AST help. v3 keeps only
    # the two rules with a clean causal story for doctest breakage:
    # blank-line deletion and ADDED trailing whitespace. Advisory always.
    ws_only = 0
    trailing = sum(1 for a in added if a != a.rstrip())
    probs = []
    if blank_removed > blank_added:
        probs.append("%d blank line(s) deleted" % (blank_removed - blank_added))
    if ws_only:
        probs.append("%d line(s) changed only in whitespace/spacing" % ws_only)
    if trailing:
        probs.append("%d added line(s) carry trailing whitespace" % trailing)
    if not probs:
        return {}
    return {"problems": probs,
            "note": ("DIFF HYGIENE: beyond your intended fix, the diff "
                     "contains cosmetic changes: " + "; ".join(probs) +
                     ". Doctests and repr-comparing tests fail on exactly "
                     "this. If a whitespace change IS the fix, keep it; "
                     "revert every other cosmetic difference so ONLY "
                     "necessary lines change.")}

def check(repo_dir):
    try:
        d = subprocess.run(["git", "diff"], cwd=repo_dir, capture_output=True,
                           text=True, timeout=60).stdout
        return check_diff(d)
    except Exception:
        return {}


def repair(repo_dir, rel_path):
    """v2 RECONSTRUCTION (Mikey, 2026-08-02: the undo-accidents version ran
    11 times against the correct fix and never yielded an acceptable diff).
    Inverted stance: rebuild the file FROM HEAD, taking from the model ONLY
    the semantic delta -- lines whose stripped content genuinely differs.
    Everything else is HEAD bytes BY CONSTRUCTION. Drift outside the delta
    is not detected and removed; it is unrepresentable."""
    import difflib, os, subprocess
    try:
        head = subprocess.run(["git", "show", "HEAD:" + rel_path],
                              cwd=repo_dir, capture_output=True, text=True,
                              timeout=30)
        if head.returncode != 0:
            return 0
        hp = head.stdout.splitlines(True)
        fp = os.path.join(repo_dir, rel_path)
        cur = open(fp, encoding="utf-8").read().splitlines(True)
    except Exception:
        return 0
    sm = difflib.SequenceMatcher(None, [l.strip() for l in hp],
                                 [l.strip() for l in cur])
    out = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            out.extend(hp[i1:i2])          # HEAD bytes, always
        elif tag in ("insert", "replace"):
            out.extend(cur[j1:j2])         # the model_s semantic delta
        elif tag == "delete":
            if all(not l.strip() for l in hp[i1:i2]):
                out.extend(hp[i1:i2])      # blank deletions are never intent
            # non-blank deletions are honoured: emit nothing
    new = "".join(out)
    if new != "".join(cur):
        open(fp, "w", encoding="utf-8").write(new)
        return 1
    return 0


def repair_syntax(repo_dir, rel_path):
    """Compile-guided single-character typo repair. Fires ONLY when the file
    fails to compile; tries deleting each single character on model-changed
    lines; accepts the first deletion that makes the WHOLE file compile.
    General, deterministic, no instance knowledge, cannot fire on a healthy
    file. Measured motive (5131): the model typed {}.\\\".format( -- one
    spurious backslash, unterminated string, hours lost."""
    import subprocess, os
    fp = os.path.join(repo_dir, rel_path)
    try:
        src = open(fp, encoding="utf-8").read()
    except Exception:
        return 0
    try:
        compile(src, rel_path, "exec")
        return 0
    except SyntaxError:
        pass
    head = subprocess.run(["git", "show", "HEAD:" + rel_path], cwd=repo_dir,
                          capture_output=True, text=True, timeout=30)
    if head.returncode != 0:
        return 0
    hset = set(head.stdout.splitlines())
    lines = src.splitlines(True)
    for i, l in enumerate(lines):
        if l.rstrip("\n") in hset:
            continue
        for j in range(len(l.rstrip("\n"))):
            cand = lines[:i] + [l[:j] + l[j + 1:]] + lines[i + 1:]
            txt = "".join(cand)
            try:
                compile(txt, rel_path, "exec")
            except SyntaxError:
                continue
            open(fp, "w", encoding="utf-8").write(txt)
            return 1
    return 0


def repair_wrap_block(repo_dir, rel_path):
    """Re-indent a suite the model wrapped in a new compound header but left
    at the header's own indent. Compile-guided; fires only on IndentationError
    ('expected an indented block'); only when the header is model-added
    (not in HEAD). Shifts the first body line and its nested suite right by 4.

    Motive django-12453: `with transaction.atomic(...):` inserted above a
    for-loop that was never indented into it. repair_syntax cannot fix this.
    """
    import os as _os
    import subprocess as _sp
    fp = _os.path.join(repo_dir, rel_path)
    try:
        src = open(fp, encoding="utf-8").read()
    except Exception:
        return 0
    try:
        compile(src, rel_path, "exec")
        return 0
    except IndentationError as e:
        lineno = e.lineno or 0
        msg = e.msg or ""
    except SyntaxError:
        return 0
    if "expected an indented block" not in msg:
        return 0
    lines = src.splitlines(True)
    if lineno < 2 or lineno > len(lines):
        return 0
    hdr = lines[lineno - 2]
    if not hdr.rstrip().endswith(":"):
        return 0
    hdr_indent = len(hdr) - len(hdr.lstrip())
    # only act if the header is model-added, not pre-existing code
    head = _sp.run(["git", "show", "HEAD:" + rel_path], cwd=repo_dir,
                   capture_output=True, text=True, timeout=30)
    if head.returncode == 0 and hdr.rstrip("\n") in set(head.stdout.splitlines()):
        return 0
    out = lines[:lineno - 1]
    i = lineno - 1
    shifted = 0
    while i < len(lines):
        ln = lines[i]
        if ln.strip() == "":
            out.append(ln); i += 1; continue
        ind = len(ln) - len(ln.lstrip())
        if shifted == 0:
            # first body line: the un-indented statement the header should own
            if ind != hdr_indent:
                return 0
            out.append("    " + ln); shifted += 1; i += 1
        else:
            # continue through the nested suite; stop when we dedent to or
            # below the header (that line is outside the wrapped block)
            if ind <= hdr_indent:
                break
            out.append("    " + ln); shifted += 1; i += 1
    if shifted == 0:
        return 0
    out += lines[i:]
    txt = "".join(out)
    try:
        compile(txt, rel_path, "exec")
    except SyntaxError:
        return 0
    open(fp, "w", encoding="utf-8").write(txt)
    return shifted
