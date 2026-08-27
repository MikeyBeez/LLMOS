#!/usr/bin/env python3
"""Contract checks that run BEFORE a campaign, not after it lies to you.

WHY THIS EXISTS. In the week of 2026-08-27 four separate numbers were
computed and then silently wrong or silently dropped, and each one cost
hours or days:

  1. The postmortem patch counter matched the literal tool name "patch"
     while edit_line, insert_lines and rewrite_function also existed, so
     "patches.attempts: 0" was false for every instance that used them --
     and two theories were built on that zero.
  2. The NO-EDIT-YET nudge thresholded on a signal the stuck instances
     never produced, so it never fired, and the silence was read as the
     model ignoring it.
  3. repertoire_fix returned the literal "declared" from its final return,
     the same string its early returns use for SOLVED, so phase2_reason
     was "declared" for all 52 rows of a campaign.
  4. _chat computed finish_reason, trunc_grow, max_tokens and eval_ms and
     phase_run's meta_log kept three fields and dropped all four -- so the
     regrow loop, once identified as the worst unbounded path in the
     harness, could not be measured at all.

Every one of them is mechanically detectable in under a second. None of
them was detected, because nothing looked. A campaign costs 300 x ~45
minutes; this costs one second and runs first.

Not a test suite. Tests check that code does what it says. This checks
that the parts still AGREE WITH EACH OTHER after one of them changed --
which is the failure mode of a system edited by someone who forgets, and
that is the harness's whole design premise.

Usage:  python3 preflight.py            # exits non-zero on any FAIL
        python3 preflight.py --warn     # report only, always exit 0
"""
import ast
import glob
import inspect
import json
import os
import re
import sys
from collections import Counter

LLMOS = os.path.dirname(os.path.abspath(__file__))
SWE = os.path.expanduser("~/swe")
sys.path.insert(0, LLMOS)

FAILS = []
WARNS = []


def fail(check, msg):
    FAILS.append((check, msg))


def warn(check, msg):
    WARNS.append((check, msg))


# --------------------------------------------------------------------------
def check_menu_dispatch_handlers():
    """Every advertised tool must be routable and every route must land.

    Three lists have to agree: FIX_TOOLS (what the model is offered),
    FIX_TOOL2SYS (name -> syscall) and the handlers dict (syscall ->
    function). Adding a tool means editing all three, and the gates strip
    entries from the first one only.
    """
    os.environ.setdefault("EDIT_LINE", "1")
    os.environ.setdefault("EDIT_SURFACE", "1")
    os.environ.setdefault("DIAG_GATE", "1")
    os.environ.setdefault("READINESS_TOOL", "1")
    import swe_fix_tools as T
    repos = sorted(glob.glob(os.path.join(SWE, "work", "*")))
    if not repos:
        warn("menu", "no checkout under ~/swe/work to bind handlers against")
        return
    handlers, _state = T.make_fix_handlers(repos[0], repo="django/django")

    # Targets the TURN LOOP handles itself, so they are correctly absent from
    # the handlers dict. Found by this file's own first run, which reported
    # them as failures: "submit" maps to RETURN (phase_run ends the phase on
    # it) and "recall" is served from phase_run's _catalog of prior tool
    # results. A checker that cries wolf on the two most load-bearing tools
    # in the menu gets switched off in a week, so they are named here.
    LOOP_HANDLED = {"RETURN", "recall"}

    advertised = [t["function"]["name"] for t in T.FIX_TOOLS]
    for name in advertised:
        target = T.FIX_TOOL2SYS.get(name)
        if not target:
            fail("menu", "tool %r is advertised but has no FIX_TOOL2SYS entry, "
                         "so a call to it cannot be routed" % name)
        elif target not in handlers and target not in LOOP_HANDLED:
            fail("menu", "tool %r routes to %r and no handler is registered "
                         "under that name" % (name, target))
    for name, target in T.FIX_TOOL2SYS.items():
        if target not in handlers and target not in LOOP_HANDLED:
            fail("menu", "FIX_TOOL2SYS maps %r -> %r but no such handler "
                         "exists" % (name, target))
    return handlers


# --------------------------------------------------------------------------
def check_edit_tools_are_counted(handlers):
    """The DERIVED edit-tool set must agree with the handlers that write.

    This check changed shape on 2026-08-27, and the reason is the point. It
    used to compare a hand-maintained EDIT_TOOLS tuple in swe_agent_v2
    against the handlers that call _atomic_write -- a drift detector between
    two lists of the same fact. The better fix was to delete one of the
    lists: swe_fix_tools.mutating_tool_names now reads the answer off the
    code that does the writing, so there is nothing to drift.

    That removes a failure mode and introduces a different one. A derivation
    can come back EMPTY or WRONG and silently zero every edit counter
    downstream -- the same bug in nicer clothes. So this is now a
    DIFFERENTIAL TEST: derive the set a second time, by a different method
    (runtime introspection of the bound handlers rather than an AST walk of
    the module source), and require the two to agree. Two independent
    derivations agreeing is evidence; one derivation is an assumption.
    """
    if not handlers:
        return
    try:
        import swe_fix_tools as T
        derived = set(T.mutating_tool_names())
    except Exception as e:
        fail("edit-tools", "mutating_tool_names() raised %s: %s -- every edit "
                           "counter downstream would be wrong or empty"
             % (type(e).__name__, e))
        return

    # Second, independent derivation: bind the handlers and read their source.
    src_by_target = {}
    for target, fn in handlers.items():
        if not target.startswith("swe."):
            continue
        try:
            src_by_target[target] = (getattr(fn, "__name__", ""),
                                     inspect.getsource(fn))
        except (OSError, TypeError):
            continue
    writers = {nm for nm, body in src_by_target.values()
               if "_atomic_write(" in body}
    while True:
        grown = {nm for nm, body in src_by_target.values()
                 if any(("%s(" % w) in body for w in writers)}
        if grown <= writers:
            break
        writers |= grown
    targets = {t for t, (nm, _b) in src_by_target.items() if nm in writers}
    import swe_fix_tools as _T
    independent = {tool for tool, tgt in _T.FIX_TOOL2SYS.items()
                   if tgt in targets}

    if not independent:
        fail("edit-tools", "no handler in the bound registry writes to the "
                           "checkout -- either _atomic_write was renamed or "
                           "the edit tools are gone")
        return
    if derived != independent:
        fail("edit-tools",
             "the two derivations of 'which tools edit files' disagree. "
             "module-source says %s; bound handlers say %s"
             % (sorted(derived), sorted(independent)))


# --------------------------------------------------------------------------
def _dict_keys_at(tree, func_name, needle):
    """String keys of the first dict literal built inside func_name whose
    source line contains needle."""
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name != func_name:
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Dict) and needle in ast.dump(sub):
                return {k.value for k in sub.keys
                        if isinstance(k, ast.Constant) and isinstance(k.value, str)}
    return None


def check_computed_is_kept():
    """Anything _chat measures must survive into the trace.

    This is bug 4. _chat builds a meta dict; phase_run copies SOME of it into
    meta_log, and the rest exists only inside the call that made it. A
    diagnostic you cannot read is not a diagnostic.
    """
    cpu = ast.parse(open(os.path.join(LLMOS, "tool_call_cpu.py"),
                         encoding="utf-8").read())
    agent = ast.parse(open(os.path.join(LLMOS, "swe_agent_v2.py"),
                           encoding="utf-8").read())
    produced = _dict_keys_at(cpu, "_chat", "finish_reason")
    kept = _dict_keys_at(agent, "phase_run", "prompt_tokens")
    if not produced or not kept:
        warn("measurement", "could not locate the meta dicts to compare "
                            "(produced=%s kept=%s)" % (bool(produced), bool(kept)))
        return
    dropped = produced - kept
    if dropped:
        fail("measurement",
             "_chat computes these and phase_run drops them, so nothing "
             "downstream can ever measure them: %s" % ", ".join(sorted(dropped)))


# --------------------------------------------------------------------------
def check_recorded_fields_vary(results_glob=None, min_rows=20):
    """A recorded field that only ever takes one value is not a measurement.

    This is bug 3, and it is checked against DATA rather than source because
    that is how it was actually found. A string field with cardinality 1
    across a whole campaign is either a constant pretending to be a variable
    or a field nobody fills in. Either way it will be read as a finding.
    """
    pattern = results_glob or os.path.join(SWE, "runs", "ornith", "all300.json")
    paths = sorted(glob.glob(pattern))
    if not paths:
        warn("cardinality", "no results file at %s; skipping" % pattern)
        return
    rows = json.load(open(paths[-1]))
    if not isinstance(rows, list) or len(rows) < min_rows:
        warn("cardinality", "only %s rows in %s; need %d to judge"
             % (len(rows) if isinstance(rows, list) else "?",
                os.path.basename(paths[-1]), min_rows))
        return
    for field in sorted({k for r in rows for k in r}):
        vals = [r.get(field) for r in rows]
        if not all(isinstance(v, str) or v is None for v in vals):
            continue
        distinct = Counter(v for v in vals if v is not None)
        if len(distinct) == 1 and sum(distinct.values()) >= min_rows:
            warn("cardinality",
                 "%s is %r for all %d rows of %s -- confirm it CAN take "
                 "another value before believing it"
                 % (field, next(iter(distinct)), sum(distinct.values()),
                    os.path.basename(paths[-1])))


# --------------------------------------------------------------------------
def main():
    warn_only = "--warn" in sys.argv
    handlers = check_menu_dispatch_handlers()
    check_edit_tools_are_counted(handlers)
    check_computed_is_kept()
    check_recorded_fields_vary()

    for kind, msg in WARNS:
        print("WARN  [%s] %s" % (kind, msg))
    for kind, msg in FAILS:
        print("FAIL  [%s] %s" % (kind, msg))
    print("\n%d fail, %d warn" % (len(FAILS), len(WARNS)))
    if FAILS and not warn_only:
        print("refusing to bless this configuration -- a campaign is "
              "300 x ~45 minutes and these numbers would be wrong")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
