"""Mechanical code probes for the LLMOS fix loop.

Deterministic, AST-based, no model calls, read-only. Purpose: surface facts
the model would otherwise have to notice unaided.

Two measured findings motivate this module:

  1. astropy-14365. The model added re.IGNORECASE to the regex (correct, and
     identical to the gold patch) then submitted. It never found the sibling
     site `if v == "NO"` -- a plain string comparison that no traceback names
     and that the regex flag cannot reach. The graded test failed with
     "DID NOT WARN": a silent data-correctness bug, not a crash.

  2. Prompt nudges do not move ornith. Measured twice (ISOLATE, then
     BLAST_RADIUS): told to call a tool, it called it once in four chances.
     Structure moves it. So this is a TOOL returning facts, never advice.

Design rule: every finding must be a fact about the code with a line number,
never a suggestion. "Line 309 compares a string with == against a literal"
is a fact. "You should make this case-insensitive" is a nudge, and nudges
have already been disproven on this model.
"""
import ast


# callable -> {keyword: why it matters when omitted}
SWITCH_SPECS = {
    "re.compile":    {"flags": "case/multiline/dotall behaviour"},
    "re.match":      {"flags": "case/multiline/dotall behaviour"},
    "re.search":     {"flags": "case/multiline/dotall behaviour"},
    "re.findall":    {"flags": "case/multiline/dotall behaviour"},
    "re.sub":        {"flags": "case/multiline/dotall behaviour"},
    "open":          {"encoding": "defaults to locale, platform-dependent",
                      "newline": "universal-newline translation",
                      "errors": "decode-error policy"},
    "sorted":        {"key": "comparison key", "reverse": "sort direction"},
    "sort":          {"key": "comparison key", "reverse": "sort direction"},
    "subprocess.run":{"shell": "shell vs argv", "check": "silently ignores failure",
                      "text": "bytes vs str"},
    "subprocess.Popen": {"shell": "shell vs argv", "text": "bytes vs str"},
    "json.dumps":    {"sort_keys": "key ordering", "ensure_ascii": "unicode escaping",
                      "default": "unserialisable-object handling"},
    "json.load":     {"object_pairs_hook": "duplicate-key / ordering handling"},
    "round":         {"ndigits": "rounding precision"},
    "split":         {"maxsplit": "split count"},
    "strip":         {"chars": "which characters are stripped"},
}

# string methods whose comparison is case-sensitive
CASE_METHODS = {"startswith", "endswith", "count", "find", "index", "replace"}


def _dotted(node):
    """ast node -> dotted callable name, or None."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    if parts:
        return parts[0]          # e.g. obj.method(...) -> "method"
    return None


def _src(lines, node):
    i = getattr(node, "lineno", 0)
    return lines[i - 1].strip() if 0 < i <= len(lines) else ""


def _has_letters(s):
    return isinstance(s, str) and any(c.isalpha() for c in s)


def switch_audit(tree, lines):
    """Calls that omit a keyword whose default silently decides behaviour."""
    out = []
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        name = _dotted(n.func)
        if not name:
            continue
        spec = SWITCH_SPECS.get(name) or SWITCH_SPECS.get(name.split(".")[-1])
        if not spec:
            continue
        given = {k.arg for k in n.keywords if k.arg}
        missing = [k for k in spec if k not in given]
        if missing:
            out.append({"probe": "switch", "line": n.lineno,
                        "code": _src(lines, n),
                        "fact": "%s() called without %s (default decides: %s)"
                                % (name, ", ".join(missing),
                                   "; ".join(spec[k] for k in missing))})
    return out


def _already_normalised(node):
    """True if this operand has already been case-folded -- v.upper(), s.lower().
    Without this the gate is unsatisfiable: fixing `v == "NO"` into
    `v.upper() == "NO"` would keep matching and the probe would never clear."""
    return (isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("lower", "upper", "casefold"))


def case_sensitivity_audit(tree, lines):
    """Sites whose behaviour depends on the case of a string."""
    out = []
    for n in ast.walk(tree):
        # x == "LITERAL" / x in ("A","B")
        if isinstance(n, ast.Compare):
            if _already_normalised(n.left):
                continue
            for op, comp in zip(n.ops, n.comparators):
                if not isinstance(op, (ast.Eq, ast.NotEq, ast.In, ast.NotIn)):
                    continue
                lits = []
                if isinstance(comp, ast.Constant) and _has_letters(comp.value):
                    lits = [comp.value]
                elif isinstance(comp, (ast.Tuple, ast.List, ast.Set)):
                    lits = [e.value for e in comp.elts
                            if isinstance(e, ast.Constant) and _has_letters(e.value)]
                if lits:
                    out.append({"probe": "case", "line": n.lineno,
                                "code": _src(lines, n),
                                "fact": "string comparison against %s is case-sensitive"
                                        % ", ".join(repr(l) for l in lits[:3])})
        # "x".startswith("Lit")
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) \
                and n.func.attr in CASE_METHODS \
                and not _already_normalised(n.func.value):
            for a in n.args:
                if isinstance(a, ast.Constant) and _has_letters(a.value):
                    out.append({"probe": "case", "line": n.lineno,
                                "code": _src(lines, n),
                                "fact": ".%s(%r) is case-sensitive"
                                        % (n.func.attr, a.value)})
                    break
    return out


def _names_in(node):
    return {x.id for x in ast.walk(node) if isinstance(x, ast.Name)}


def _bound_by(target):
    return {x.id for x in ast.walk(target) if isinstance(x, ast.Name)}


def loop_invariant_audit(tree, lines):
    """Assignments inside a loop that do not depend on anything the loop changes."""
    out = []
    for n in ast.walk(tree):
        if not isinstance(n, (ast.For, ast.While)):
            continue
        loop_vars = _bound_by(n.target) if isinstance(n, ast.For) else set()
        assigned = set()
        for s in ast.walk(n):
            if isinstance(s, ast.Assign):
                for t in s.targets:
                    assigned |= _bound_by(t)
            elif isinstance(s, ast.AugAssign):
                assigned |= _bound_by(s.target)
        changing = loop_vars | assigned
        for s in n.body:
            if isinstance(s, ast.Assign) and not (_names_in(s.value) & changing):
                out.append({"probe": "hoist", "line": s.lineno,
                            "code": _src(lines, s),
                            "fact": "assignment inside loop (line %d) uses nothing "
                                    "the loop changes" % n.lineno})
    return out


def shortcircuit_audit(tree, lines):
    """and/or where a later operand depends on what an earlier one guards."""
    out = []
    for n in ast.walk(tree):
        if not isinstance(n, ast.BoolOp) or len(n.values) < 2:
            continue
        for i, v in enumerate(n.values[:-1]):
            guarded = _names_in(v)
            for later in n.values[i + 1:]:
                deref = {_dotted(x.value) for x in ast.walk(later)
                         if isinstance(x, ast.Attribute)}
                deref |= {_dotted(x.func) for x in ast.walk(later)
                          if isinstance(x, ast.Call)}
                if guarded & {d.split(".")[0] for d in deref if d}:
                    out.append({"probe": "order", "line": n.lineno,
                                "code": _src(lines, n),
                                "fact": "operand order matters: a later operand "
                                        "dereferences a name an earlier one tests"})
                    break
            else:
                continue
            break
    return out


PROBES = (("switch", switch_audit),
          ("case",   case_sensitivity_audit),
          ("hoist",  loop_invariant_audit),
          ("order",  shortcircuit_audit))


def probe_source(source, only=None):
    """Run probes over python source text. Returns list of findings."""
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [{"probe": "parse", "line": e.lineno or 0, "code": "",
                 "fact": "file does not parse: %s" % e.msg}]
    lines = source.split("\n")
    out = []
    for tag, p in PROBES:
        if only and tag not in only:
            continue
        try:
            out.extend(p(tree, lines))
        except Exception as e:                      # a probe must never break the loop
            out.append({"probe": "error", "line": 0, "code": "",
                        "fact": "%s failed: %s" % (p.__name__, type(e).__name__)})
    return sorted(out, key=lambda d: (d["line"], d["probe"]))


def probe_file(path, only=None):
    with open(path, encoding="utf-8", errors="ignore") as fh:
        return probe_source(fh.read(), only=only)


def format_report(findings, limit=40):
    if not findings:
        return "no probe findings"
    by = {}
    for f in findings:
        by.setdefault(f["probe"], []).append(f)
    parts = []
    for k in sorted(by):
        parts.append("%s (%d):" % (k, len(by[k])))
        for f in by[k][:limit]:
            parts.append("  line %-5d %s" % (f["line"], f["fact"]))
            if f["code"]:
                parts.append("           %s" % f["code"][:100])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# SIBLING-SITE SWEEP
#
# The measured failure (astropy-14365): the model fixed the site the traceback
# named and submitted, leaving an identical-class site untouched. Running ALL
# probes over a file is too noisy to help -- qdp.py yields 39 findings, most of
# them true and irrelevant. So the sweep is CLASS-SCOPED: classify the edit that
# was just made, then report only the UNCHANGED sites of that same class.
# For 14365 that turns 39 findings into exactly one: line 309.
# ---------------------------------------------------------------------------
import re as _re

_CASE_RE = _re.compile(r"re\.IGNORECASE|\.lower\(\)|\.upper\(\)|\.casefold\(\)")


def classify_edit(old_snippet, new_snippet):
    """Which probe classes does this edit belong to? Drives the sweep."""
    added = new_snippet or ""
    removed = old_snippet or ""
    cls = set()
    if _CASE_RE.search(added) and not _CASE_RE.search(removed):
        cls.add("case")
    new_kw = set(_re.findall(r"[(,]\s*(\w+)\s*=[^=]", added))
    old_kw = set(_re.findall(r"[(,]\s*(\w+)\s*=[^=]", removed))
    if new_kw - old_kw:
        cls.add("switch")
    na = sorted(x.strip() for x in added.split("\n") if x.strip())
    nr = sorted(x.strip() for x in removed.split("\n") if x.strip())
    if na and na == nr:
        cls.add("order")
    return cls


def sibling_sites(path, classes, exclude_lines=(), limit=8):
    """Unchanged sites of the same class as the edit just made."""
    if not classes:
        return []
    ex = set(exclude_lines)
    found = probe_file(path, only=classes)
    return [f for f in found
            if f["probe"] in classes and f["line"] not in ex][:limit]


def sweep_after_edit(path, old_snippet, new_snippet, edited_lines=(), limit=8):
    """Full sweep: classify the edit, return same-class sites left untouched.
    Returns {} when there is nothing to say -- callers inject only on truth."""
    classes = classify_edit(old_snippet, new_snippet)
    if not classes:
        return {}
    sites = sibling_sites(path, classes, exclude_lines=edited_lines, limit=limit)
    if not sites:
        return {}
    return {
        "edit_class": sorted(classes),
        "unchanged_same_class_sites": [
            {"line": s["line"], "fact": s["fact"], "code": s["code"][:120]}
            for s in sites
        ],
    }
