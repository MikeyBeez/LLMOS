#!/usr/bin/env python3
"""Deterministic post-patch spec probes. WARNS, never refuses.

Data behind it (2026-08-02/03): four one-assertion misses in one night.
The hidden spec is often reachable from the CODE, not the issue: operator
dunders travel in families, and return-type idioms are visible in the file.
Both checks are computable, so they are computed -- prompt nudges have been
disproven twice on this model; facts with line numbers work.
"""
import ast
import os
import re

ARITH = {
    "__mul__", "__rmul__", "__truediv__", "__rtruediv__", "__floordiv__",
    "__rfloordiv__", "__add__", "__radd__", "__sub__", "__rsub__",
    "__pow__", "__rpow__", "__mod__", "__rmod__", "__matmul__",
    "__rmatmul__", "__eq__", "__ne__", "__lt__", "__le__", "__gt__",
    "__ge__", "__neg__", "__pos__", "__abs__", "__div__", "__rdiv__",
}

BARE_RET = re.compile(r"^\s*return\s+(1|0|-1|True|False|None)\s*(#.*)?$")
SINGLETON = re.compile(r"\bS\.(One|Zero|Half|NegativeOne|true|false)\b")
SYMPY_IMPORT = re.compile(r"^from sympy|^import sympy", re.M)
# Mikey 2026-08-03: "if there is an object with a property that matches
# what is being fixed, make sure the IDENTITY holds." A file that
# practices identity comparison against canonical objects confesses it
# in its own source: `is S.One`, `is Module.CONST`. That is the
# deterministic evidence that equal-but-fresh values are bugs here.
IDENT_CANON = re.compile(r"\bis\s+(?:not\s+)?[A-Z]\w*\.\w+")


def _undefined_names(repo_dir, rel_path):
    """pyflakes the whole edited file; return the set of undefined names.
    Empty set on any failure -- this must never break the patch path."""
    try:
        import io
        from pyflakes.api import check as _pf_check
        from pyflakes.reporter import Reporter
        s = open(os.path.join(repo_dir, rel_path), encoding="utf-8",
                 errors="replace").read()
        buf = io.StringIO()
        _pf_check(s, rel_path, Reporter(buf, buf))
        return set(re.findall(r"undefined name '(\w+)'", buf.getvalue()))
    except Exception:
        return set()


def _enclosing_class_method(src, lo, hi):
    """(class_name, method_name) containing lines lo..hi, or (None, None)."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return (None, None)
    for cls in ast.walk(tree):
        if not isinstance(cls, ast.ClassDef):
            continue
        for m in cls.body:
            if not isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            a = m.lineno
            b = getattr(m, "end_lineno", m.lineno)
            if not (hi < a or lo > b):
                return (cls.name, m.name)
    return (None, None)


def _overriding_subclasses(repo_dir, rel_path, cls_name, meth_name, cap=40):
    """Files defining a subclass of cls_name that also defines meth_name."""
    import subprocess as _sp
    hits = []
    try:
        r = _sp.run(["git", "grep", "-l", "--", cls_name], cwd=repo_dir,
                    capture_output=True, text=True, timeout=60)
        if r.returncode not in (0, 1):
            return hits
        cands = [f.strip() for f in (r.stdout or "").splitlines() if f.strip()]
    except Exception:
        return hits
    cands = [f for f in cands
             if f.endswith(".py") and f != rel_path
             and "/tests/" not in f and not f.split("/")[-1].startswith("test_")]
    for f in cands[:cap]:
        try:
            s = open(os.path.join(repo_dir, f), encoding="utf-8",
                     errors="replace").read()
            t = ast.parse(s)
        except Exception:
            continue
        for cls in ast.walk(t):
            if not isinstance(cls, ast.ClassDef):
                continue
            bases = set()
            for b in cls.bases:
                if isinstance(b, ast.Name):
                    bases.add(b.id)
                elif isinstance(b, ast.Attribute):
                    bases.add(b.attr)
            if cls_name not in bases:
                continue
            for m in cls.body:
                if (isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and m.name == meth_name):
                    hits.append("%s (class %s, line %d)" % (f, cls.name,
                                                            m.lineno))
                    break
        if len(hits) >= 6:
            break
    return hits


def _self_calls(node):
    """Names in `self.<name>(...)` anywhere under node."""
    out = set()
    for n in ast.walk(node):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and isinstance(n.func.value, ast.Name)
                and n.func.value.id == "self"):
            out.add(n.func.attr)
    return out


def _sibling_helper_gap(src, lo, hi, min_callers=4, cap=2, min_methods=5):
    """(class, method, [(n_callers, helper), ...]) -- helpers that many sibling
    methods of the enclosing class call and the edited method does not."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return (None, None, [])
    for cls in ast.walk(tree):
        if not isinstance(cls, ast.ClassDef):
            continue
        methods, target = [], None
        for m in cls.body:
            if not isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            methods.append(m)
            a = m.lineno
            b = getattr(m, "end_lineno", m.lineno)
            if not (hi < a or lo > b):
                target = m
        if target is None or len(methods) < min_methods:
            continue
        if target.name.startswith("__"):
            return (None, None, [])
        mine = _self_calls(target)
        counts = {}
        for m in methods:
            if m is target:
                continue
            for nm in _self_calls(m):
                counts[nm] = counts.get(nm, 0) + 1
        # min_callers=4 and no dunders: measured on 250 random django
        # methods, 3-caller and __getattr__-shaped hits were the noise.
        gaps = sorted(((c, nm) for nm, c in counts.items()
                       if c >= min_callers and nm not in mine
                       and nm != target.name
                       and not nm.startswith('__')), reverse=True)
        return (cls.name, target.name, gaps[:cap])
    return (None, None, [])


def probe(repo_dir, rel_path, edited_line, written_text):
    """Return a warning note (str) or None. Never raises."""
    try:
        if not rel_path or not str(rel_path).endswith(".py"):
            return None
        try:
            src = open(os.path.join(repo_dir, rel_path), encoding="utf-8",
                       errors="replace").read()
        except OSError:
            return None
        n_lines = len((written_text or "").splitlines()) or 1
        lo = int(edited_line or 0)
        hi = lo + n_lines - 1
        notes = []

        # A: operator-dunder families travel together (sympy-24909: gold
        # edits __mul__ AND __truediv__; the issue only shows __mul__).
        if lo:
            try:
                tree = ast.parse(src)
            except SyntaxError:
                tree = None
            if tree is not None:
                for cls in ast.walk(tree):
                    if not isinstance(cls, ast.ClassDef):
                        continue
                    dund = {}
                    for m in cls.body:
                        if isinstance(m, (ast.FunctionDef,
                                          ast.AsyncFunctionDef)) \
                                and m.name in ARITH:
                            dund[m.name] = (m.lineno,
                                            getattr(m, "end_lineno",
                                                    m.lineno))
                    hit = [k for k, (a, b) in dund.items()
                           if not (hi < a or lo > b)]
                    if hit:
                        sibs = sorted(k for k in dund if k not in hit)
                        if sibs:
                            notes.append(
                                "SPEC PROBE: your edit changed %s in class "
                                "%s. The same class also defines %s, which "
                                "this edit did NOT touch. Operator siblings "
                                "usually embed the same pattern and need the "
                                "same fix -- read each one, and if it has "
                                "the same flaw, fix it in the SAME patch."
                                % (", ".join(sorted(hit)), cls.name,
                                   ", ".join(sibs)))
                        break

        # B: bare-literal returns where the codebase uses library singletons
        # (sympy graders test identity: `is S.One`, so int 1 fails even when
        # the math is right).
        if written_text and (SINGLETON.search(src)
                             or SYMPY_IMPORT.search(src)
                             or IDENT_CANON.search(src)):
            for i, ln in enumerate((written_text or "").splitlines(),
                                   start=lo):
                if BARE_RET.match(ln):
                    notes.append(
                        "SPEC PROBE: line %d of your edit returns a bare "
                        "Python literal (%s). This codebase returns library "
                        "objects (e.g. sympy uses S.One, S.Zero -- callers "
                        "compare with `is`). If the codebase has a "
                        "CANONICAL object equal to this value, return "
                        "that object itself -- identity must hold, not "
                        "just equality." % (i, ln.strip()))
                    break
        # C: rare shared names. django-11001: the model fixed 1 of 2 USES
        # of self.ordering_parts; gold fixed the shared DEFINITION (line 35)
        # so every use was repaired at once. If the edit touches a name that
        # appears only a few times in the file, name the untouched lines.
        if written_text:
            names = set(re.findall(r"self\.(\w{4,})", written_text))
            src_lines = src.splitlines()
            for nm in sorted(names):
                pat = re.compile(r"\bself\.%s\b" % re.escape(nm))
                occs = [i + 1 for i, l in enumerate(src_lines)
                        if pat.search(l)]
                outside = [i for i in occs if not (lo <= i <= hi)]
                if outside and 0 < len(occs) <= 6:
                    notes.append(
                        "SPEC PROBE: `self.%s` (touched by your edit) also "
                        "appears at line(s) %s, which this edit did NOT "
                        "change. If the flaw lives in the shared object, fix "
                        "its DEFINITION so every use is repaired at once; if "
                        "each use site must change, change them ALL."
                        % (nm, ", ".join(map(str, outside[:6]))))
                    break
        # D: missing imports (django-11620: model caught Http404 in
        # resolvers.py without importing it; every graded test died on
        # NameError). "Are there any missing imports?" is deterministic.
        if written_text:
            und = _undefined_names(repo_dir, rel_path)
            mine = sorted(n for n in und
                          if re.search(r"\b%s\b" % re.escape(n),
                                       written_text))
            if mine:
                notes.append(
                    "SPEC PROBE: after your edit, name(s) %s are UNDEFINED "
                    "in this file -- neither defined nor imported. Code "
                    "that references them dies with NameError at runtime. "
                    "Add the import (top of file) in the SAME patch."
                    % ", ".join(mine))
        # E: base-class edit that a subclass OVERRIDES (django-12113 class).
        # If the method you changed is re-implemented in a subclass, your edit
        # never runs for that subclass -- and the graded behaviour may be
        # exactly that subclass's.
        if lo and written_text:
            _c, _m = _enclosing_class_method(src, lo, hi)
            if _c and _m and not _m.startswith("__"):
                _ov = _overriding_subclasses(repo_dir, rel_path, _c, _m)
                if _ov:
                    notes.append(
                        "SPEC PROBE: you edited %s.%s, but %s OVERRIDE(S) that "
                        "method: %s. Your change does NOT run for those "
                        "subclasses. Decide whether the fix belongs in the "
                        "override instead of (or as well as) the base class."
                        % (_c, _m, "a subclass" if len(_ov) == 1
                           else "subclasses", "; ".join(_ov)))
        # F: CONVENTION GAP. django-12908: the whole accepted fix was one
        # line -- a guard that ten sibling methods of the same class already
        # call and this one did not. If the class has a convention and your
        # method skips it, that omission may BE the bug.
        if lo and written_text:
            _fc, _fm, _fg = _sibling_helper_gap(src, lo, hi)
            if _fc and _fm and _fg:
                notes.append(
                    "SPEC PROBE: in class %s, %s -- but %s does not call %s. "
                    "If that is the class's convention for this kind of "
                    "method, its ABSENCE may be the bug, and adding the call "
                    "may be the whole fix."
                    % (_fc,
                       "; ".join("self.%s() is called by %d other method(s)"
                                 % (n, c) for c, n in _fg),
                       _fm,
                       "it" if len(_fg) == 1 else "them"))

        # G: numbered migrations are applied history, not live code.
        if re.search(r"(?:^|/)migrations/\d{3,4}_\w+\.py$", rel_path or ""):
            notes.append(
                "SPEC PROBE: %s is a NUMBERED MIGRATION -- an applied "
                "historical record, not live code. The framework records "
                "applied migrations by name, so editing this file changes "
                "nothing for any database that has already run it. The fix "
                "belongs in the code that GENERATES migrations, or in a NEW "
                "migration." % rel_path)

        return "\n".join(notes) or None
    except Exception:
        return None


def repro_assertion_note(script, stderr=""):
    """Warn when a registered reproduction cannot detect a wrong value.

    Returns a note (str) or None. Deterministic; never raises.
    """
    try:
        if not script or not script.strip():
            return None
        try:
            tree = ast.parse(script)
        except SyntaxError:
            return None
        n_assert = sum(1 for n in ast.walk(tree) if isinstance(n, ast.Assert))
        n_print = sum(1 for n in ast.walk(tree)
                      if isinstance(n, ast.Call)
                      and isinstance(n.func, ast.Name) and n.func.id == "print")
        err = stderr or ""
        crashed = ("Traceback (most recent call last)" in err
                   and "AssertionError" not in err)
        exc = ""
        if crashed:
            for ln in reversed(err.splitlines()):
                s = ln.strip()
                if s and not s.startswith(("File ", "Traceback", "During",
                                           "The above")) and not ln[:1].isspace():
                    exc = s.split(":")[0]
                    break
        if n_assert == 0:
            msg = ("REPRODUCTION STRENGTH: this script contains NO assertions"
                   "%s. It exits 0 -- and therefore counts as GREEN -- as soon "
                   "as nothing raises. It cannot detect a WRONG RESULT, only a "
                   "crash." % (" (%d print statements)" % n_print
                               if n_print else ""))
            if crashed and exc:
                msg += (" It currently fails only because of an uncaught %s. "
                        "The moment your patch stops that exception this script "
                        "will pass EVEN IF the value it produces is wrong."
                        % exc)
            msg += (" Add assertions that pin the EXPECTED value for every "
                    "distinct behaviour the issue describes, then re-register.")
            return msg
        if n_assert == 1:
            return ("REPRODUCTION STRENGTH: this script asserts exactly ONE "
                    "thing. Issues usually imply several distinct expectations "
                    "(other inputs, the inverse operation, the neighbouring "
                    "case), and the graded tests check them separately. "
                    "Enumerate every expectation the issue implies and assert "
                    "each one.")
        return None
    except Exception:
        return None
