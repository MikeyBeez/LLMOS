"""Call-graph neighbourhood lookup, backed by graphify (graphifyy 0.9.29).

WHY THIS AND NOT SEARCH. Tested 2026-07-28 on pylint-dev__pylint-7080, whose
gold fix is one line in _is_ignored_file (pylint/lint/expand_modules.py):

  graphify query "recursive ignores ignore-paths"   -- the issue's own title --
  returned 47 nodes, almost all test fixtures (ignores_no_docstring(),
  missing_return_doc.py, ignore_except_pass_by_default.py). The gold target was
  NOT among them. Clustering first made no difference. It is lexical matching
  on node labels, and a test corpus is full of those words.

  graphify affected "_is_in_ignore_list_re" --depth 2  returned, immediately:
      expand_modules()   [calls]  pylint/lint/expand_modules.py:L143
      _is_ignored_file() [calls]  pylint/lint/expand_modules.py:L58   <-- gold
      ._expand_files()   [calls]  pylint/lint/pylinter.py:L797
      test__is_in_ignore_list_re_match()  tests/lint/unittest_expand_modules.py:L24

So graphify answers "what is connected to X" well and "where is the bug" badly.
This module exposes only the half that works: given a symbol the agent has
ALREADY found (via locate, or a traceback frame), return its neighbourhood.

That targets the measured bucket-A failure -- fixed one site, never found the
sibling -- e.g. astropy-14365, where the model added re.IGNORECASE and never
touched `if v == "NO"` at line 309.

Deterministic: tree-sitter AST only, no LLM, no network. Build is ~26s for a
62MB / 1951-file repo, cached in <repo>/graphify-out and reused thereafter.
"""
import os
import subprocess

GRAPHIFY = os.environ.get(
    "GRAPHIFY_BIN", "/home/bard/.graphify_venv/bin/graphify")
BUILD_TIMEOUT = int(os.environ.get("GRAPHIFY_BUILD_TIMEOUT", "600"))
QUERY_TIMEOUT = int(os.environ.get("GRAPHIFY_QUERY_TIMEOUT", "60"))


def _graph_path(repo_dir):
    return os.path.join(repo_dir, "graphify-out", "graph.json")


def available():
    return os.path.isfile(GRAPHIFY)


def ensure_graph(repo_dir, log=print):
    """Build the graph once per checkout. Returns (ok, message)."""
    gp = _graph_path(repo_dir)
    if os.path.isfile(gp) and os.path.getsize(gp) > 1000:
        return True, "cached"
    if not available():
        return False, "graphify binary not found at %s" % GRAPHIFY
    try:
        r = subprocess.run([GRAPHIFY, "update", ".", "--no-cluster"],
                           cwd=repo_dir, capture_output=True, text=True,
                           timeout=BUILD_TIMEOUT)
    except subprocess.TimeoutExpired:
        return False, "graph build timed out after %ds" % BUILD_TIMEOUT
    if not os.path.isfile(gp):
        return False, "graph build produced no graph.json: %s" % (r.stderr or "")[-300:]
    log(" -- graphify: built code graph for %s" % os.path.basename(repo_dir))
    return True, "built"


def _run(args, repo_dir):
    try:
        r = subprocess.run([GRAPHIFY] + args, cwd=repo_dir,
                           capture_output=True, text=True, timeout=QUERY_TIMEOUT)
        return (r.stdout or "") + (("\n" + r.stderr) if r.returncode else "")
    except subprocess.TimeoutExpired:
        return ""


def neighborhood(repo_dir, symbol, depth=2, log=print):
    """Everything that calls, imports or references `symbol`, and what it calls.

    Facts with file:line, never advice -- the agent decides what matters.
    """
    ok, msg = ensure_graph(repo_dir, log=log)
    if not ok:
        return {"error": msg}
    aff = _run(["affected", symbol, "--depth", str(depth)], repo_dir)
    exp = _run(["explain", symbol], repo_dir)
    if not aff.strip() and not exp.strip():
        return {"symbol": symbol,
                "error": "symbol not found in the code graph; check the exact name"}
    return {"symbol": symbol,
            "callers_and_dependents": aff[:3000],
            "definition_and_edges": exp[:2000]}

# ---------------------------------------------------------------------------
# AUTOMATIC INJECTION (2026-07-29)
#
# MEASURED, not assumed: with GRAPHIFY=1 the `neighborhood` tool was in the fix
# tool list for two full instances -- 84 tool calls, a description that spells
# out exactly when to use it -- and the model called it ZERO times. That is the
# third time a persuasive description has moved this model not at all. Offering
# a tool is exhortation. Firing it is structure.
#
# So the neighbourhood is fetched BY THE HARNESS after a successful patch and
# attached to the patch result, the same channel and the same shape as
# sibling_sites. The model does not have to decide to look; it just receives.
#
# WHY THE CLASS AND NOT THE METHOD. First version asked the graph about the
# innermost enclosing def. Measured on django-15061, where the model edited
# MultiWidget.id_for_label:
#
#   affected "id_for_label"                          -> No unique node match
#                                                       (15 nodes share the label)
#   affected "MultiWidget.id_for_label"              -> No unique node match
#                                                       (dotted form is not a selector)
#   affected "django_forms_widgets_multiwidget_id_for_label"
#                                                    -> No affected nodes found
#   affected "MultiWidget"                           -> RangeWidget [inherits],
#        SplitDateTimeWidget [inherits] django/forms/widgets.py:L895,
#        ComplexMultiWidget [inherits] tests/forms_tests/field_tests/
#            test_multivaluefield.py:L27,   <-- an actual FAIL_TO_PASS module
#        ... plus tests/forms_tests/widget_tests/test_multiwidget.py
#
# A method name is ambiguous across a large codebase, and nothing "calls" an
# overridden method by name anyway -- dispatch is dynamic. The CLASS is what
# other code inherits, instantiates and tests. So for an edit inside a class we
# ask about the class, and fall back to the bare function name only for
# module-level functions (pylint's _is_ignored_file, where it works well).
# ---------------------------------------------------------------------------

_DEF_RE = None


def _def_re():
    global _DEF_RE
    if _DEF_RE is None:
        import re
        _DEF_RE = re.compile(r"^(\s*)(?:async\s+)?(def|class)\s+([A-Za-z_]\w*)")
    return _DEF_RE


def enclosing_scopes(file_path, line):
    """(innermost def or None, innermost class or None) containing `line`.

    Text-based on purpose: the file has just been mutated and may not parse. A
    syntax error in the file being edited is exactly when this gets called.
    """
    rx = _def_re()
    try:
        with open(file_path, encoding="utf-8", errors="ignore") as fh:
            lines = fh.read().splitlines()
    except OSError:
        return None, None
    if not lines:
        return None, None
    idx = max(0, min(int(line or 1) - 1, len(lines) - 1))
    func = cls = None
    ceiling = None          # only strictly-less-indented scopes enclose us
    for i in range(idx, -1, -1):
        m = rx.match(lines[i])
        if not m:
            continue
        indent = len(m.group(1).expandtabs(4))
        if ceiling is not None and indent >= ceiling:
            continue
        ceiling = indent
        if m.group(2) == "def":
            if func is None:
                func = m.group(3)
        else:
            if cls is None:
                cls = m.group(3)
            if indent == 0:
                break
        if indent == 0:
            break
    return func, cls


def enclosing_symbol(file_path, line):
    """Back-compatible: the innermost def, else the innermost class."""
    func, cls = enclosing_scopes(file_path, line)
    return func or cls


def _graph_query(repo_dir, symbol, depth, log):
    info = neighborhood(repo_dir, symbol, depth=depth, log=log)
    if not isinstance(info, dict) or info.get("error"):
        return ""
    aff = (info.get("callers_and_dependents") or "")
    # graphify answers "no unique node match" / "no affected nodes found" with
    # exit 0 and prose, so an empty answer has to be recognised, not assumed.
    low = aff.lower()
    if ("no unique node match" in low or "no affected nodes found" in low
            or not aff.strip()):
        return ""
    return aff


def neighborhood_of_edit(repo_dir, rel_path, line, depth=2, log=print):
    """Call-graph neighbourhood of whatever was just edited.

    Class first, then the function -- see the note above. Returns {} when
    there is nothing useful to say, so the caller injects nothing rather than
    injecting noise.
    """
    if not rel_path:
        return {}
    full = (rel_path if os.path.isabs(rel_path)
            else os.path.join(repo_dir, rel_path))
    func, cls = enclosing_scopes(full, line)
    for sym, kind in ((cls, "class"), (func, "function")):
        if not sym:
            continue
        aff = _graph_query(repo_dir, sym, depth, log)
        if aff.strip():
            return {"edited_symbol": sym,
                    "symbol_kind": kind,
                    "other_sites_in_the_call_graph": aff[:1800],
                    "note": ("these inherit from, reference, or are referenced "
                             "by what you just changed; a fix often needs more "
                             "than one site")}
    return {}


_TEST_RE = None


def test_files_near(repo_dir, rel_path, line, depth=2, log=print):
    """Repo-relative test files the graph connects to the edited code.

    For choosing which EXISTING tests to run as a blast-radius check. Path
    proximity has to guess among thousands of test files; the graph names them.
    Measured on django-15061 (edit in MultiWidget.id_for_label): returns
    tests/forms_tests/field_tests/test_multivaluefield.py and
    tests/forms_tests/widget_tests/test_multiwidget.py -- the first is an
    actual FAIL_TO_PASS module, and path proximity from django/forms/widgets.py
    ranks neither.

    Leak-safe: this reads the repo's own call graph, never the answer key.
    """
    global _TEST_RE
    if _TEST_RE is None:
        import re
        _TEST_RE = re.compile(r"((?:[\w.\-/]+/)?(?:tests?|testing)/[\w.\-/]*"
                              r"\.py|[\w.\-/]*test_[\w.\-]*\.py|"
                              r"[\w.\-/]*_test\.py)")
    info = neighborhood_of_edit(repo_dir, rel_path, line, depth=depth, log=log)
    aff = info.get("other_sites_in_the_call_graph") or ""
    out = []
    for m in _TEST_RE.finditer(aff):
        p = m.group(1)
        if p not in out and os.path.isfile(os.path.join(repo_dir, p)):
            out.append(p)
    return out
