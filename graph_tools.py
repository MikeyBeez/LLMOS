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
