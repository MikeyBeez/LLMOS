"""SWE-bench fix-loop toolkit — purpose-shaped tools for the bug-fix phase.

Runs AFTER repo_bootstrap_tools has verified the environment. Each tool
mirrors one step of the ideal loop (reproduce -> locate -> read -> patch
-> verify -> submit) and hides the shell/fs primitives so the model isn't
tempted to burn steps on generic exploration.

VERIFICATION MODEL (rewritten 2026-07-10 — "fix this the right way"):
The agent operates in the STRICT SWE-bench setting: it sees the problem
statement only. FAIL_TO_PASS test ids are NOT given to the model and are
NOT runnable here anyway — most of those tests are added by the scoring
test_patch and do not exist in the working tree during the fix phase.
The old run_failing_test targeted them regardless, which either errored
(pylint: 'not found') or vacuously passed, letting the model declare
victory with an EMPTY DIFF (requests-3362, xarray-5131: patch_bytes=0,
fix_verified=True).

The gate is now red -> green on the agent's OWN reproduction:
  1. reproduce(script): a script that exits NONZERO because of the bug.
     The harness registers the last failing script as THE reproduction.
  2. patch: any edit invalidates prior verification.
  3. verify_fix(): reruns the registered reproduction; ok when exit==0.
  4. submit: accepted only when (a) a reproduction failed at least once
     (seen RED), (b) the same script now passes (GREEN), and (c) the
     git diff of non-test source files is non-empty.
"""
import fnmatch, os, re, shlex, shutil, signal, subprocess, time

from repo_bootstrap_tools import llm_call, _extract_json


def _apply_edit(text, old, new):
    """Apply old->new. Exact unique match first; fall back to whitespace-
    insensitive line matching so a snippet off only by indentation/trailing
    space still lands. Returns (new_text, how) or (None, error_message)."""
    cnt = text.count(old)
    if cnt == 1:
        return text.replace(old, new, 1), "exact"
    if cnt > 1:
        return None, ("old_snippet matches %d places exactly - add surrounding "
                      "context to disambiguate" % cnt)
    tlines = text.split("\n")
    olines = old.split("\n")
    while olines and olines[0].strip() == "": olines.pop(0)
    while olines and olines[-1].strip() == "": olines.pop()
    if not olines:
        return None, "old_snippet is empty"
    want = [l.strip() for l in olines]
    hits = [i for i in range(len(tlines) - len(want) + 1)
            if [tlines[j].strip() for j in range(i, i + len(want))] == want]
    if len(hits) == 1:
        i = hits[0]
        tlines[i:i + len(want)] = new.split("\n")
        return "\n".join(tlines), "whitespace-tolerant"
    if len(hits) > 1:
        return None, ("old_snippet matches %d places (whitespace-insensitive) - "
                      "add more surrounding context" % len(hits))
    return None, _anchor_diagnostic(tlines, want, olines)


def _anchor_diagnostic(tlines, want, olines):
    """Explain WHY an old_snippet did not match: locate the best partial match
    and name the first line that diverges, with what the file actually holds
    there. A constant 'not found' carries no new information, so the model's
    only move is to resend the same text; a divergence point is actionable.
    General: no repo-, file- or instance-specific knowledge."""
    tstrip = [l.strip() for l in tlines]
    starts = [i for i, l in enumerate(tstrip) if l == want[0]]
    if not starts:
        present = sum(1 for w in want if w in tstrip)
        if present == 0:
            return ("old_snippet not found: NONE of its %d lines appear in this "
                    "file. Either the wrong file, or the text came from memory "
                    "rather than from a read_range of this file. read_range the "
                    "region, then copy 1-3 lines verbatim." % len(want))
        return ("old_snippet not found: its first line %r does not appear in "
                "this file (%d of its %d lines do). Re-anchor on 1-3 lines you "
                "can see verbatim in a fresh read_range."
                % (olines[0].strip()[:100], present, len(want)))
    best_i, best_n = starts[0], 0
    for i in starts:
        n = 0
        while n < len(want) and i + n < len(tstrip) and tstrip[i + n] == want[n]:
            n += 1
        if n > best_n:
            best_i, best_n = i, n
    j = best_i + best_n
    sent = want[best_n] if best_n < len(want) else "<snippet ended>"
    have = tstrip[j] if j < len(tstrip) else "<end of file>"
    return ("old_snippet not found. It matched the first %d of its %d lines "
            "starting at file line %d, then diverged at file line %d:\n"
            "  you sent: %s\n"
            "  file has: %s\n"
            "Correct that line, or re-anchor on <=3 lines copied verbatim from "
            "a fresh read_range. Do NOT resend this snippet unchanged."
            % (best_n, len(want), best_i + 1, j + 1, sent[:120], have[:120]))


def _repo_frames(stderr, repo_dir):
    """In-repo traceback frames (relpath:line (fn)) from stderr, skipping venv/
    stdlib, so the agent jumps to the fault site instead of grep-hunting."""
    import re as _re
    out = []
    for m in _re.finditer(r'File "([^"]+)", line (\d+), in (\S+)', stderr or ""):
        path, line, fn = m.group(1), m.group(2), m.group(3)
        low = path.replace("\\", "/")
        if "/.venv/" in low or "site-packages" in low or "/lib/python" in low:
            continue
        try:
            rel = os.path.relpath(path, repo_dir) if os.path.isabs(path) else path
        except Exception:
            rel = path
        if rel.startswith(".."):
            continue
        out.append("%s:%s (%s)" % (rel, line, fn))
    return out[-3:]


_PROX_GENERIC = {"", "py", "tests", "test", "django", "src", "lib", "__init__",
                 "backends", "contrib", "core", "base", "common", "utils", "main",
                 "models", "db", "backend"}


def _prox_tokens(path):
    """Distinctive path tokens (dirs + file), minus generic scaffolding words.
    'django/contrib/sitemaps/__init__.py' -> {'sitemaps'};
    'tests/sitemaps_tests/test_http.py'   -> {'sitemaps', 'http'}."""
    out = set()
    for seg in path.split("/"):
        for t in re.split(r"[_.]", seg):
            if t and t not in _PROX_GENERIC:
                out.add(t)
    return out


def _fault_proximity(test_file, hint_paths):
    """Closeness score between a test file and the fault/source hint paths
    (repo-relative). Higher = nearer. Returns 0 with no hints so ranking is a
    no-op (selection stays in collection order).

    Two signals, so it works for BOTH repo layouts:
      - shared leading path segments / same dir  (sympy-style: source and tests
        share a prefix, sympy/parsing/... <-> sympy/parsing/tests/...);
      - distinctive NAME-TOKEN overlap  (django-style: source under django/... and
        tests under tests/<app>_tests/... share no prefix, but the app name --
        'sitemaps', 'cache' -- appears in both). An exact app-dir match
        (tests/<app>/ or tests/<app>_tests/) is the strongest signal, so a cache
        BACKEND change lands on tests/cache/ rather than a template {% cache %}
        test that merely mentions the word.
    Repo-agnostic and pure: no gold/test-patch data, no I/O."""
    if not hint_paths:
        return 0
    tsegs = test_file.split("/")
    tdir = "/".join(tsegs[:-1])
    ttoks = _prox_tokens(test_file)
    best = 0
    for h in hint_paths:
        if not h:
            continue
        hsegs = h.split("/")
        hdir = "/".join(hsegs[:-1])
        common = 0
        for a, b in zip(tsegs[:-1], hsegs[:-1]):
            if a == b:
                common += 1
            else:
                break
        score = common
        if tdir and tdir == hdir:
            score += 5
        score += 4 * len(ttoks & _prox_tokens(h))
        hdir_toks = [s for s in hsegs[:-1] if s not in _PROX_GENERIC]
        for hs in hdir_toks:
            for ts in tsegs[:-1]:
                if ts == hs or ts == hs + "_tests" or ts == "test_" + hs:
                    score += 6
                    break
        if score > best:
            best = score
    return best


def _rank_test_files(file_nids, hint_paths, limit=6):
    """Pick up to `limit` neighbor-test node ids (one per file), biased toward
    the fault/source hint paths. STABLE: with no hints every score is 0, so the
    result is the first `limit` entries in the original order -- byte-identical
    to the pre-change 'first-N distinct files' selection. Pure/deterministic
    (no I/O): unit-testable and it cannot touch any scoring path."""
    ranked = sorted(
        enumerate(file_nids),
        key=lambda t: (-_fault_proximity(t[1][0], hint_paths), t[0]))
    return [nid for _, (f, nid) in ranked[:limit]]


def _reproduction_strength(script):
    """Classify a reproduction's ACCEPTANCE strength (advisory only, never a gate).

    Returns one of:
      'value_check'      -> asserts an expected value/relation or a specific
                            exception (assert x == y, assertEqual, pytest.raises)
                            -> a meaningful RED->GREEN discriminator.
      'vacuous_constant' -> the only assertion is a literal constant
                            (assert True / assert 1) -> verifies nothing.
      'weak'             -> no assertion, or only truthiness asserts; GREEN means
                            only "no exception was raised".

    Leakage-safe: inspects ONLY the model's own reproduction script text; no
    gold/test-patch/FAIL_TO_PASS data is consulted. Used to steer the model, not
    to change any score or the submit gate.
    """
    if not script:
        return "weak"
    body = "\n".join(re.sub(r"#.*$", "", ln) for ln in script.splitlines())
    asserts = re.findall(r"\bassert\s+(.+)", body)
    value_sig = (
        bool(re.search(r"\bassert\b[^\n]*(==|!=|<=|>=|<|>| in | is | not in )", body))
        or bool(re.search(r"\bassert(Equal|AlmostEqual|In|Is|ListEqual|Dict|Set|Regex|Raises|Greater|Less)\b", body))
        or "np.testing" in body or "pytest.raises" in body or ".raises(" in body
        or "assertRaises" in body)
    only_const = bool(asserts) and all(
        re.match(r"^\(?\s*(True|1)\s*\)?\s*$", a.strip()) for a in asserts)
    if only_const:
        return "vacuous_constant"
    if value_sig:
        return "value_check"
    return "weak"


# Repo-agnostic test / test-infra path detector. A candidate patch that only
# touches test paths can never resolve a SWE-bench instance (the scoring
# FAIL_TO_PASS tests are held out), so such a patch must be refused at edit time
# and must not satisfy the submit gate. pytest keeps its own suite under
# testing/ (not tests/) and conftest.py is test infrastructure; both were
# missed by the older (^|/)tests?/ regex.
_TEST_PATH_RE = re.compile(
    r"(^|/)(tests?|testing)/|(^|/)test_|_test\.py$|(^|/)conftest\.py$")


def _failure_sig(exit_code, stderr):
    """Normalized failure signature: same bug -> same sig across reruns.
    Strips addresses, durations, tmp paths, line numbers."""
    t = (stderr or "")[-1500:]
    t = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", t)
    t = re.sub(r"\b\d+\.\d+s\b", "T", t)
    t = re.sub(r"/tmp/[\w./-]+", "/tmp/PATH", t)
    t = re.sub(r"line \d+", "line N", t)
    return "%s:%s" % (exit_code, hash(t))


def _is_test_path(path):
    """True if `path` is a test or test-infra file (repo-agnostic)."""
    return bool(_TEST_PATH_RE.search(str(path or "")))


def _format_lint(diff_text):
    """Diff-scoped naming check: added Title-case labels should derive from
    attributes the added lines read. Returns (warning_or_None)."""
    import re as _re
    reads = {}
    labels = []
    for ln in diff_text.splitlines():
        if not ln.startswith("+"):
            continue
        for m in _re.finditer(r"\b\w+\.([a-z_][a-z0-9_]*)\b", ln):
            a = m.group(1)
            if a not in ("format", "join", "append", "get", "items",
                         "startswith", "strip", "split", "replace"):
                reads[a] = reads.get(a, 0) + 1
        for m in _re.finditer(r"[\"\']([A-Z][A-Za-z]{2,18})[\"\']", ln):
            labels.append(m.group(1))
    if not labels or not reads:
        return None
    derived = set()
    for a in reads:
        derived.add(a.title().replace("_", ""))
        derived.add(a.replace("_", " ").title())
        derived.add(a.capitalize())
    coined = [l for l in sorted(set(labels)) if l not in derived]
    if not coined:
        return None
    cands = sorted(reads, key=reads.get, reverse=True)[:4]
    return ("FORMAT CHECK: your patch introduces label(s) %s but reads "
            "attribute(s) %s. Labels should be named after the fields they "
            "display (e.g. %s). Rename, or be certain the coined word is the "
            "project's own term." % (coined, cands,
                                     ", ".join(a.title() for a in cands[:2])))


def _coined_from_warning(warning):
    """Labels _format_lint flagged as coined, recovered from its message."""
    if not warning:
        return []
    head = str(warning).split("but reads")[0]
    return re.findall(r"[\"\']([A-Za-z][A-Za-z]{1,18})[\"\']", head)


def _format_objects_to_repro(state):
    """True when the harness's own naming check objects to a label that the
    LOCKED reproduction hard-codes as its acceptance criterion.

    This is the one situation where the frozen verification target is itself
    what the harness is complaining about: complying with the format check
    necessarily turns the locked reproduction red, so the agent is pinned to
    the flagged wording. Capped at one unlock per run -- the latch's whole
    purpose is to stop reproduction thrash, and one rewrite is not thrash.
    """
    if state.get("format_unlocks", 0) >= 1:
        return False
    script = state.get("repro_script") or ""
    if not script:
        return False
    for label in _coined_from_warning(state.get("format_warning")):
        if label in script:
            return True
    return False


def _research_maps(repo_dir, relfile, line, fmap, flow):
    """Append every generated map to the research corpus, for cross-instance
    analysis later (Mikey: collecting them makes the general principle visible).

    RESEARCH ZONE ONLY. The runtime never reads ~/swe/research/. This is an
    export, not a channel -- nothing written here re-enters any run.
    """
    import json as _j
    try:
        d = os.path.expanduser("~/swe/research/maps")
        os.makedirs(d, exist_ok=True)
        iid = os.path.basename(os.path.abspath(repo_dir))
        # keep EVERYTHING: a principle you have not seen yet cannot be
        # recovered from a summary written before you knew what to look for.
        rec = {"iid": iid, "file": relfile, "line": line, "ts": int(time.time()),
               "anatomy": fmap or None,      # enclosing + every symbol, sig, doc
               "flow": flow or None}         # every container, op, key, finding
        with open(os.path.join(d, iid + ".jsonl"), "a", encoding="utf-8") as f:
            f.write(_j.dumps(rec) + "\n")
        return None
    except Exception as e:
        # never break a run -- but never fail silently either. A swallowed
        # collector is indistinguishable from a working one, which is how the
        # disabled probe stayed hidden for a whole run.
        return "%s: %s" % (type(e).__name__, str(e)[:80])


def _fired(state, feature):
    """Mark that a feature actually ran. Counted in fix_state, which IS
    serialised -- unlike tool results, which are budget-truncated out of the
    stored conversation and cannot be recovered afterwards."""
    try:
        f = state.setdefault("features_fired", {})
        f[feature] = f.get(feature, 0) + 1
    except Exception:
        pass


def _seen_before(state, path, new_text):
    """Would this edit RESTORE a file state we have already been in?
    Returns the index of the prior visit, or None. Records nothing."""
    import hashlib
    h = hashlib.md5(new_text.encode("utf-8", "ignore")).hexdigest()
    hist = state.setdefault("state_history", {}).setdefault(path, [])
    return (hist.index(h) if h in hist else None), h, hist


def _remember_state(state, path, text):
    import hashlib
    h = hashlib.md5(text.encode("utf-8", "ignore")).hexdigest()
    state.setdefault("state_history", {}).setdefault(path, []).append(h)


def _anchor_record(state):
    """What was tried and why it failed -- so the next attempt is informed by
    the failures rather than being another draw from the same wrong picture."""
    fa = state.get("failed_anchors") or {}
    if not fa:
        return []
    return ["turn %s  %s  |%s|  -> %s" % (v["turn"], v["file"], v["head"], v["why"])
            for v in sorted(fa.values(), key=lambda x: x["turn"])][:8]


def _syntax_check(full_path, rel):
    """Does this file still parse? Definitive, no style opinions. Returns None
    when fine, else a dict naming the line and what broke."""
    if not full_path.endswith(".py"):
        return None
    try:
        with open(full_path, encoding="utf-8", errors="ignore") as f:
            src = f.read()
        # compile(), NOT ast.parse(). ast.parse only PARSES; errors raised by
        # the symbol table -- "return outside function", "nonlocal at module
        # level", duplicate parameters -- are compile-time and slip straight
        # through it. Measured 2026-07-28 on django-15061: the model left a
        # duplicated `class SplitDateTimeWidget(MultiWidget):` with an orphaned
        # `return` under it. ast.parse accepted the file, we recorded
        # syntax_ok=True, the agent was never told, and grading then died with
        # "SyntaxError: 'return' outside function". The instance regressed
        # identically across three different harness designs today because none
        # of them touched the thing that was actually broken.
        compile(src, full_path, "exec")
        return None
    except SyntaxError as e:
        line = e.lineno or 0
        ctx = []
        try:
            lines = src.splitlines()
            for i in range(max(0, line - 3), min(len(lines), line + 2)):
                ctx.append("%s%5d  %s" % (">>" if i + 1 == line else "  ",
                                          i + 1, lines[i][:110]))
        except Exception:
            pass
        return {"broke_the_file": True,
                "error": "%s at line %s" % (e.msg, line),
                "context": ctx,
                "what_to_do": ("Your patch left %s unparseable, so nothing "
                               "downstream can run. Fix THIS before anything "
                               "else -- usually indentation, an unclosed "
                               "bracket or quote, or a line that was replaced "
                               "without its continuation." % rel)}
    except Exception:
        return None


def render_worksheet(state):
    """Deterministic worksheet from the fix-state ledger. Fixed template, no
    sampled prose (pattern #33): equal state -> byte-identical worksheet."""
    ph = state.get("patch_history") or []
    if state.get("repro_green"):
        repro = "GREEN (your registered reproduction passes)"
    elif state.get("seen_red"):
        repro = "RED registered (verify_fix reruns exactly it)"
    else:
        repro = "none yet - write a script that FAILS because of the bug"
    if state.get("probe_script"):
        probe = "GREEN" if state.get("probe_green") else                 "locked, still RED (the issue's stated property is not fixed)"
    else:
        probe = "none"
    if ph:
        last = ph[-1]
        pline = "%d (last: %s -> %s%s)" % (
            len(ph), last["file"], last["verdict"],
            ", same failure x%d" % state["same_verify_count"]
            if state.get("same_verify_count", 0) > 1 else "")
    else:
        pline = "0"
    # next obligation: what the form says must happen, as one imperative
    if not state.get("seen_red"):
        nxt = "register a failing reproduction"
    elif not state.get("repro_green"):
        if state.get("same_verify_count", 0) >= 2:
            # PROMPT REPERTOIRE (2026-08-08, gated PROMPT_ROTATE, default off).
            #
            # The old behaviour fired ONE alternate imperative and then repeated
            # it forever. Measured today: "NECESSARY BUT NOT SUFFICIENT" was
            # issued 1767 times and "go and read_range" 145 times, and neither
            # changed what the model did. A request the model has already
            # ignored is not evidence of anything -- repeating it is.
            #
            # So rotate, in the order the failure classes actually occur:
            #   0 incomplete  - 58% of multi-hunk misses are right-place,
            #                   second location never found
            #   1 misdiagnosis- the class a fresh retry NEVER rescues (22.9% of
            #                   failed-both were a clean single assertion)
            #   2 revert      - escalation when neither framing moved anything
            # Same select-then-exhaust discipline as the edit repertoire:
            # intelligence picks the order, exhaustion decides when to stop.
            _ROT = (
                ("incomplete",
                 "your reproduction still fails and your edit IS in place. The "
                 "fix is more likely INCOMPLETE than wrong: the accepted fix "
                 "for this kind of bug usually changes more than one place in "
                 "the same file. Find the sibling that needs the same change "
                 "and patch it too, keeping the edit you already made."),
                ("misdiagnosis",
                 "the same failure again. Consider that your edit is CORRECT "
                 "and your THEORY is wrong. Re-read the issue and state in one "
                 "sentence the property it claims should hold, then check "
                 "whether your patch establishes THAT property rather than the "
                 "one you assumed."),
                ("revert",
                 "stop patching. Say which facts you have OBSERVED by running "
                 "something and which you ASSUMED, revert any edit you cannot "
                 "justify from an observation, and re-approach from the "
                 "reproduction."),
            )
            if os.environ.get("PROMPT_ROTATE", "0") == "1":
                _i = (int(state.get("same_verify_count", 2)) - 2) % len(_ROT)
                state["prompt_frame"] = _ROT[_i][0]
                nxt = _ROT[_i][1]
            else:
                nxt = ("your patch is not reaching the failing code path - re-read "
                       "fault_locations and change the PATCH, not the reproduction")
        elif ph:
            nxt = "verify_fix (or improve the patch)"
        else:
            nxt = "read the fault location, then patch"
    elif state.get("probe_script") and not state.get("probe_green"):
        nxt = ("repro passes but the locked probe does not - your fix answers "
               "your theory, not the issue; re-read the issue and patch again")
    else:
        nxt = ("repro and probe are green. BEFORE submitting, state in one "
               "sentence the INVARIANT your patch establishes, then run ONE "
               "VARIANT of your reproduction that differs along the dimension "
               "the issue names (other casing, equal-vs-unequal lengths, other "
               "type, the other branch, a second call site of the same rule). "
               "A patch that only passes the exact script you wrote is "
               "under-constrained: the same concept usually lives at more than "
               "one place in the code. If the variant fails, patch again. "
               "Then run_tests on the neighborhood and submit.")
    lines = ["WORKSHEET (maintained by the harness; observed unless marked [assumed]):"]
    if state.get("triage_goal"):
        lines.append("  trying to get : %s   [assumed at turn 0]"
                     % state["triage_goal"][:200])
    if state.get("triage_repro"):
        lines.append("  valid repro   : %s   [assumed at turn 0]"
                     % state["triage_repro"][:200])
    if state.get("prior_attempts_note"):
        lines.append("  prior attempts: %s   [observed]"
                     % state["prior_attempts_note"][:220])
    if state.get("chain_mechanism"):
        lines.append("  must be true  : %s   [assumed at turn 0 -- check() it]"
                     % state["chain_mechanism"][:200])
    if state.get("chain_change"):
        lines.append("  so change     : %s" % state["chain_change"][:200])
    lines.append("  repro status  : %s" % repro)
    lines.append("  probe status  : %s" % probe)
    lines.append("  patches tried : %s" % pline)
    if state.get("format_warning"):
        lines.append("  format check  : %s" % state["format_warning"][:220])
    # WORKSHEET_VARIANT (2026-08-08, default "control" = byte-identical).
    # "evidence" re-surfaces the last failure text next to the request for the
    # next patch. Everything else about the template is unchanged so the two
    # arms differ by exactly one line.
    if (os.environ.get("WORKSHEET_VARIANT", "control") == "evidence"
            and state.get("last_verify_tail")):
        lines.append("  last failure  : %s   [observed]"
                     % state["last_verify_tail"])
    # SWEBENCH_MODE (2026-08-08, default off). Once edits are landing and the
    # reproduction is still red, the open question is WHERE ELSE in this file,
    # not which file. List the candidate sites rather than exhorting.
    if (os.environ.get("SWEBENCH_MODE", "0") == "1"
            and state.get("seen_red") and not state.get("repro_green")
            and state.get("patch_history") and state.get("_sibling_fn")):
        _sib = state["_sibling_fn"](state.get("last_edit_file") or "",
                                    state.get("last_edit_frag") or "")
        _f = state.get("last_edit_file") or "this file"
        _defs = state["_defs_fn"](state.get("last_edit_file") or "") \
            if state.get("_defs_fn") else []
        if _sib or _defs:
            lines.append("  your edit may be NECESSARY BUT NOT SUFFICIENT: the "
                         "accepted fix for this kind of bug usually changes "
                         "MORE THAN ONE place in %s." % _f)
        if _sib:
            lines.append("      same symbol reused at: %s"
                         % ", ".join("line %d" % _n for _n, _t in _sib))
        if _defs:
            lines.append("      NOW LOOK FOR SIBLINGS WITH DIFFERENT NAMES - "
                         "members of the same family needing the same change "
                         "(the other operator dunders, the other _print_* "
                         "methods, the other branch of the same rule). Every "
                         "definition in this file:")
            lines.append("      " + ", ".join("%s:%d" % (_nm, _n)
                                              for _n, _nm in _defs))
    if os.environ.get("DIAG_GATE", "0") == "1" and state.get("diag"):
        lines.append("  diagnosis     : " + " | ".join(
            "%s %s" % (k.split("_", 1)[0], v)
            for k, v in sorted(state["diag"].items())))
    lines.append("  next          : %s" % nxt)
    if state.get("triage_goal") or state.get("chain_mechanism"):
        lines.append("  --")
        lines.append("  [assumed] lines were written before anything was read. "
                     "Everything else was observed by running something. If "
                     "what you observe contradicts an assumed line, the "
                     "ASSUMPTION is what is wrong -- say so and correct it. "
                     "Do not bend the evidence to fit it.")
    return "\n".join(lines)


def _missing_file_hint(path, repo_dir):
    """A redirecting error for a path that is not a real repo file. The top
    miss cause is the model reading/patching a path it invented -- its inline
    reproduction script, an absolute /work path, or a guessed test file. A bare
    'file not found' does not correct the false belief, so it loops."""
    p = str(path)
    if ("_reproduction" in p or "/work/" in p or p.startswith("/")
            or ".." in p.split("/")):
        return ("not a repo file: %r. reproduction/check scripts run inline -- "
                "they are NOT saved files you can read or patch. Only real "
                "source files under the repo root can be read/edited (e.g. "
                "'django/...', 'src/...'). Use locate(pattern=...) to find the "
                "real path; never guess or invent file paths." % p)
    hint = ("file not found: %r. Do NOT guess file paths -- use "
            "locate(pattern=...) to find the real location, then read_range / "
            "patch that exact path." % p)
    try:
        base = os.path.basename(p)
        if base:
            for root, dirs, files in os.walk(repo_dir):
                dirs[:] = [d for d in dirs if d not in
                           (".git", ".venv", ".condaenv", "node_modules",
                            "__pycache__")]
                if base in files:
                    rel = os.path.relpath(os.path.join(root, base), repo_dir)
                    hint += " (a file named %r exists at %r)" % (base, rel)
                    break
    except OSError:
        pass
    return hint


def _filter_repo_frames(repo_dir, frames):
    """Keep only fault frames that point into REAL repo files.

    CYCLE-2 RERUN FINDING (django-11422): the registered reproduction's only
    frame was <string>:41 -- the repro script itself. It was stored as a
    fault location anyway, which (a) waived the S2 differential on the claim
    that the traceback names in-repo writers, and (b) fed the declare_site
    stack check a frame that resolves to nothing. Junk evidence is worse
    than no evidence: it silently satisfies gates built to demand the real
    thing.
    """
    out = []
    for f in frames or []:
        try:
            head = str(f).split(" ")[0]
            path = head.rsplit(":", 1)[0]
            path = path[2:] if path.startswith("./") else path
            if path and not path.startswith("<") and os.path.isfile(
                    os.path.join(repo_dir, path)):
                out.append(f)
        except Exception:
            continue
    return out


def _mpl_force_draw(script, repo):
    """SYMBOLIC FORCED DRAW (2026-08-11, gated REPRO_FORCE_DRAW, default off).

    Mikey: "You can't just have a rule for drawing. You have to make it
    deterministic. That means you have to do it with symbolic code. Force it
    to draw."  The REPRO_CONTRACT hint nudges where the model puts its
    assertion; THIS makes the draw itself unconditional, harness-side:

      prologue -- pins the Agg backend before anything imports pyplot
                  (inserted after any __future__/comment lines, which must
                  stay first);
      epilogue -- draws EVERY open figure after the model's script body.
                  A script that exits 0 because the bug is invisible pre-draw
                  now crashes at the forced draw instead: nonzero exit, a
                  traceback whose frames point INTO the repo (so
                  fault_locations fills in), and a deterministic RED that
                  needed no model cooperation.

    Scripts that already exit nonzero before the epilogue are unchanged in
    outcome. The wrapped text is what gets REGISTERED, so verify_fix reruns
    the identical instrumented script and green means green-with-draw.
    Applied only in script mode -- a pytest-mode file would execute the
    epilogue at collection time.
    """
    if os.environ.get("REPRO_FORCE_DRAW", "0") != "1":
        return script
    if repo not in ("matplotlib/matplotlib", "mwaskom/seaborn"):
        # seaborn IS matplotlib -- its figures need the same forced draw.
        return script
    pro = ("import matplotlib as _mpl_h\n"
           "_mpl_h.use('Agg', force=True)\n")
    epi = ("\n\n# --- harness epilogue: force a draw of every open figure ---\n"
           "try:\n"
           "    import matplotlib.pyplot as _plt_h\n"
           "    for _n_h in _plt_h.get_fignums():\n"
           "        _plt_h.figure(_n_h).canvas.draw()\n"
           "except SystemExit:\n"
           "    raise\n"
           "except BaseException:\n"
           "    import traceback as _tb_h\n"
           "    _tb_h.print_exc()\n"
           "    import sys as _sys_h\n"
           "    _sys_h.exit(3)\n")
    lines = script.splitlines(True)
    k = 0
    for k, ln in enumerate(lines):
        if not (ln.startswith("from __future__") or ln.startswith("#")
                or not ln.strip()):
            break
    return "".join(lines[:k]) + pro + "".join(lines[k:]) + epi


def make_fix_handlers(repo_dir, env_vars=None, env_kind="uv", repo=None):
    """Return handlers bound to this repo checkout. env_vars carries anything
    the bootstrap phase set (e.g. DJANGO_SETTINGS_MODULE). env_kind selects
    .venv (uv/pip) or .condaenv (conda)."""
    env_vars = dict(env_vars or {})
    env_dir = ".condaenv" if env_kind == "conda" else ".venv"
    state = {"submitted": False, "fix_verified": False,
             "baseline_pass": None,   # neighbor tests passing pre-patch
             "regressions": [],
             "repro_script": None,      # the registered failing script
             "repro_mode": "script",    # "script" (python -c) or "pytest"
             "seen_red": False,         # a reproduction has failed (bug shown)
             "repro_green": False,      # registered script now exits 0
             "probe_script": None,      # LOCKED issue-invariant probe (immutable)
             "probe_green": None,       # probe status after last verify
             "repro_locked": False,     # reproduction proven green: immutable
             "rejected_repro_streak": 0,  # consecutive green-exit repro scripts
             "patch_history": [],
        "failed_anchors": {},
        "features_fired": {},
        "state_history": {},   # file -> [content hashes seen]
        "func_edits": {},      # enclosing function -> edit count
        "must_observe": False,       # [{file, verdict}] verdict set by next verify
        "stuck": 0,                  # STUCK_ESCALATE: consecutive non-progress patches
             "triage_goal": "",         # done_criteria from the understanding pass
             "triage_repro": ""}        # repro_criteria from the understanding pass

    def _run(cmd, timeout=300):
        env = os.environ.copy()
        env.update(env_vars)
        venv_bin = os.path.join(repo_dir, env_dir, "bin")
        env["PATH"] = venv_bin + ":" + env.get("PATH", "")
        if env_kind == "conda":
            env["CONDA_PREFIX"] = os.path.join(repo_dir, env_dir)
        else:
            env["VIRTUAL_ENV"] = os.path.join(repo_dir, env_dir)
        proc = subprocess.Popen(cmd, shell=True, cwd=repo_dir,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, env=env, start_new_session=True)
        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            # shell=True forks /bin/sh which can fork the probe as a grandchild;
            # killing only the direct child (subprocess.run's behaviour) orphans
            # it to PPID=1 where it can spin at 100% CPU forever. Kill the group.
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                proc.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            raise
        return subprocess.CompletedProcess(cmd, proc.returncode, out, err)

    def _diff_nonempty():
        # Gate helper: the working tree must contain at least one changed
        # NON-TEST source file. Editing only tests/conftest can never resolve a
        # SWE-bench instance, so a test-only diff must NOT satisfy the gate.
        r = _run("git diff --name-only", timeout=60)
        files = [f for f in (r.stdout or "").splitlines() if f.strip()]
        return any(not _is_test_path(f) for f in files)

    def _gate():
        ok = (state["seen_red"] and state["repro_green"] and _diff_nonempty())
        # a locked issue-invariant probe is part of "verified": a fix that
        # leaves the issue's own stated property broken cannot pass the gate.
        if ok and state.get("probe_script") and not state.get("probe_green"):
            ok = False
        state["fix_verified"] = ok
        return state["fix_verified"]

    def _capture_baseline(hint_paths=None, graph_files=None):
        """Sample neighbor tests that PASS in the pre-patch tree (cheap: a
        spread of a few files, short timeouts). Called once, before any
        patch, so a later failure is a real regression."""
        try:
            import test_runner as _tr
            ids = _tr.collect_ids(repo_dir, env_kind, env_vars=env_vars)
        except Exception:
            ids = []
        seen, per_file = set(), []
        for nid in ids:
            f = nid.split("::", 1)[0]
            if f not in seen:
                seen.add(f)
                per_file.append((f, nid))
        spread = _rank_test_files(per_file, hint_paths, limit=6)
        # GRAPH_HINTS: test files the call graph connects to the edited code go
        # in FIRST, ahead of anything proximity chose. Proximity has to guess
        # among thousands of test files by path similarity; the graph names
        # them. On django-15061 proximity picked six files that never rendered
        # a label, while the graph named both FAIL_TO_PASS modules.
        if graph_files:
            _gsel = [nid for nid in ids
                     if nid.split("::", 1)[0] in set(graph_files)]
            if _gsel:
                print("   -- GRAPH_HINTS seeded %d ids from %d graph test "
                      "file(s): %s" % (len(_gsel[:40]), len(graph_files),
                                       ", ".join(graph_files[:4])))
                spread = _gsel[:40] + [n for n in spread if n not in _gsel]
            else:
                print("   -- GRAPH_HINTS no-fire (graph named %d file(s), none "
                      "collected: %s)" % (len(graph_files),
                                          ", ".join(graph_files[:4])))
        # BLAST_RADIUS: pull the FULL nearest test file(s) (all their tests), not
        # just one id per file, so the neighbor-test baseline covers the module the
        # change touches. Base status is genuine (pre-patch); FAIL_TO_PASS fails
        # pre-patch so it never enters baseline_pass (no leak).
        check = list(spread)
        if (os.environ.get("BLAST_RADIUS") == "1"
                or os.environ.get("NEIGHBOR_INJECT") == "1") and spread:
            top_files = []
            for _nid in spread[:2]:
                _f = _nid.split("::", 1)[0]
                if _f not in top_files:
                    top_files.append(_f)
            extra = [nid for nid in ids
                     if nid.split("::", 1)[0] in top_files and nid not in check]
            check += extra[:30]
        passing = []
        import test_runner as _tr
        for nid in check:
            try:
                r = _tr.run_tests(repo_dir, env_kind, [nid], env_vars=env_vars,
                                  repo=repo, timeout=120)
                if r["ok"]:
                    passing.append(nid)
            except Exception:
                pass
        state["baseline_pass"] = passing

    def _check_regressions():
        """Rerun baseline-passing tests; any now failing = regression."""
        base = state.get("baseline_pass") or []
        if not base:
            return []
        import test_runner as _tr
        regressed = []
        for nid in base:
            try:
                r = _tr.run_tests(repo_dir, env_kind, [nid], env_vars=env_vars,
                                  repo=repo, timeout=120)
                if not r["ok"]:
                    regressed.append(nid)
            except Exception:
                pass
        state["regressions"] = regressed
        return regressed

    def _exec_repro(script, mode, timeout=300):
        """Run the reproduction with the instrument its problem type needs.
        Returns the completed process. mode=="pytest" runs it through the repo's
        own framework via a temp file OUTSIDE the repo (so it never enters the
        model's git diff). Nonzero exit = RED; for pytest, exit 5 (no tests)
        is caller-checked as INVALID, not red."""
        if mode == "pytest":
            import tempfile
            fd, tmp = tempfile.mkstemp(suffix="_llmos_repro.py", dir="/tmp")
            os.close(fd)
            open(tmp, "w").write(script)
            try:
                return _run(f"{env_dir}/bin/python -m pytest -x -q "
                            f"{shlex.quote(tmp)}", timeout=timeout)
            finally:
                try:
                    os.remove(tmp)
                except OSError:
                    pass
        return _run(f"{env_dir}/bin/python -c {shlex.quote(script)}",
                    timeout=timeout)

    _EXC_LINE = re.compile(r"^((?:[A-Za-z_]\w*\.)*[A-Za-z_]\w*)(?::|$)")
    _TB_CHROME = ("Traceback (most recent", "During handling",
                  "The above exception")

    def _exception_type(err):
        """Class name of the exception that ended the script, or None.

        Parsed, not matched -- see the note on _SETUP_ERRORS.
        """
        for line in reversed((err or "").splitlines()):
            if not line.strip() or line[:1].isspace():
                continue
            if line.startswith(_TB_CHROME):
                continue
            m = _EXC_LINE.match(line.strip())
            if m:
                return m.group(1).rsplit(".", 1)[-1]
        return None

    _SETUP_ERRORS = (
        "ModuleNotFoundError", "ImportError", "ImproperlyConfigured",
        "AppRegistryNotReady", "NameError", "SyntaxError", "IndentationError",
        "TabError", "FileNotFoundError")

    def _repro_quality(r):
        """Why did this script exit nonzero? See the REPRO_QUALITY note.

        Returns (tier, detail). Only "broken" should be refused.
        """
        err = (r.stderr or "")
        out = (r.stdout or "")
        has_tb = "Traceback (most recent call last)" in err
        frames = _repo_frames(err, repo_dir)
        if "AssertionError" in err:
            return "assertion", "AssertionError"
        if not has_tb and out.strip():
            # ran to completion, printed something, chose to exit nonzero
            return "assertion", "explicit nonzero exit with output"
        _exc = _exception_type(err)
        if _exc in _SETUP_ERRORS:
            return "broken", _exc
        if has_tb and not frames:
            return "broken", "traceback never reaches repo source"
        if frames:
            return "repo_exception", frames[0]
        return "unknown", "nonzero exit, no traceback and no output"

    def _promote_if_red_at_base(script, mode):
        """Is this passing script actually a valid reproduction of the bug?

        Red on the unmodified base tree and green with the current patch is the
        definition. Anything else is not promotable. See the module note on
        REPRO_PROMOTE for the measurement that motivated this.

        Leak-safe: runs the MODEL\'s own script against the repo\'s own source.
        Nothing here reads the gold patch or FAIL_TO_PASS.
        """
        import subprocess as _sp
        import tempfile as _tf

        def _git(cmd, t=60):
            return _sp.run(cmd, shell=True, cwd=repo_dir, capture_output=True,
                           text=True, timeout=t)

        diff = _git("git diff").stdout or ""
        if not diff.strip():
            print("   -- REPRO_PROMOTE skipped: no patch applied, so a passing "
                  "script proves nothing", flush=True)
            return False
        fd, pth = _tf.mkstemp(suffix=".promote.patch")
        with os.fdopen(fd, "w") as fh:
            fh.write(diff)

        base_rc, restore_err = None, None
        try:
            _git("git checkout -- .")
            base_rc = _exec_repro(script, mode, timeout=180).returncode
        except Exception as _e:
            print("   -- REPRO_PROMOTE error while testing base: %s: %s"
                  % (type(_e).__name__, _e), flush=True)
        finally:
            # Restore unconditionally. Losing the candidate patch here would be
            # far worse than declining the promotion.
            _git("git checkout -- .")
            _ap = _git("git apply %s" % pth)
            if _ap.returncode != 0:
                restore_err = (_ap.stderr or "").strip()[:180]

        if restore_err is not None:
            print("   -- REPRO_PROMOTE RESTORE FAILED (%s). Working tree is now "
                  "BASE; the candidate diff is preserved at %s"
                  % (restore_err, pth), flush=True)
            state["repro_green"] = False
            state["fix_verified"] = False
            return False
        try:
            os.unlink(pth)
        except OSError:
            pass

        if base_rc is None:
            return False
        if base_rc == 0:
            print("   -- REPRO_PROMOTE rejected: the script passes on the BASE "
                  "tree too, so it does not exercise the bug", flush=True)
            return False
        print("   -- REPRO_PROMOTE accepted: RED at base (exit %s), GREEN with "
              "your patch -- this is a valid reproduction" % base_rc, flush=True)
        return True

    def _enclosing_def(path, line):
        """Name of the def/class containing `line` of a repo file, or None."""
        try:
            _p = path[2:] if path.startswith("./") else path
            with open(os.path.join(repo_dir, _p), encoding="utf-8",
                      errors="ignore") as _fh:
                _ls = _fh.read().splitlines()
            for _l in reversed(_ls[:min(line, len(_ls))]):
                _m = re.match(r"\s*(?:async\s+)?(?:def|class)\s+(\w+)", _l)
                if _m:
                    return _m.group(1)
        except Exception:
            pass
        return None

    def h_differential(pcb, args):
        """DIAGNOSIS LADDER step 2 (2026-08-11, shipped with DIAG_GATE).

        Mikey: the steps must be REQUIRED in the program, and each one either
        runs or is recorded as unnecessary. This is the discriminating
        experiment from matplotlib-23299: same operations WITH and WITHOUT the
        condition the issue names. One comparison carries the diagnosis -- if
        the control is clean, the defect is in STATE the condition changes,
        and the fix site is usually the function that WRITES that state.
        Result is recorded in state["diag"] and shown every turn."""
        bug = str(args.get("bug_script", ""))
        ctl = str(args.get("control_script", ""))
        if not bug.strip() or not ctl.strip():
            return {"error": (
                "provide BOTH bug_script (with the issue-named condition) and "
                "control_script (the SAME operations with that condition "
                "removed -- no rc_context, no flag, no special mode)")}
        if bug.strip() == ctl.strip():
            return {"error": (
                "control_script is identical to bug_script -- remove the "
                "condition the issue names so the comparison isolates it")}
        rb = _exec_repro(bug, "script", timeout=120)
        rc2 = _exec_repro(ctl, "script", timeout=120)
        state.setdefault("diag", {})["S2_differential"] = (
            "done: bug exit %s vs control exit %s"
            % (rb.returncode, rc2.returncode))
        state["diag_differential"] = True
        return {"bug_exit": rb.returncode, "control_exit": rc2.returncode,
                "bug_tail": ((rb.stdout or "") + (rb.stderr or ""))[-800:],
                "control_tail": ((rc2.stdout or "") + (rc2.stderr or ""))[-800:],
                "next": (
                    "Exits differ -> the removed condition is load-bearing: "
                    "find the STATE it changes (compare the raw values in "
                    "both worlds with check), then declare_site the function "
                    "that WRITES that state. Exits equal -> the condition is "
                    "not the trigger; revise the comparison.")}

    def h_declare_site(pcb, args):
        """DIAGNOSIS LADDER step 3: name WHERE the fix goes and WHY, before
        editing. Writer-over-reader is the measured rule: in 41 of 107
        analysed misses the agent edited the function where the symptom
        appears while the accepted fix edits the function that writes the
        state (38%% -- the largest failure class). The declaration is state,
        rendered every turn; editing outside it is challenged once."""
        path = str(args.get("file", ""))
        func = str(args.get("function", "") or "")
        role = str(args.get("role", "") or "")
        why = str(args.get("reason", "") or "")
        full = os.path.join(repo_dir, path)
        if not path or not os.path.isfile(full):
            return {"error": (_missing_file_hint(path, repo_dir)
                              if path else "file is required")}
        if func:
            _txt = open(full, encoding="utf-8", errors="ignore").read()
            if not re.search(r"^\s*(?:async\s+)?(?:def|class)\s+%s\b"
                             % re.escape(func), _txt, re.M):
                _defs = re.findall(r"^\s*(?:async\s+)?(?:def|class)\s+(\w+)",
                                   _txt, re.M)
                return {"error": "no def/class named %r in %s" % (func, path),
                        "definitions_in_file": _defs[:40],
                        "hint": "nothing was recorded; pick a name from the "
                                "list or omit function"}
        state["diag_site"] = {"file": path, "function": func, "role": role}
        note = "declared: %s%s%s -- %s" % (
            path, (":" + func) if func else "",
            (" (%s)" % role) if role else "", (why[:120] or "no reason given"))
        state.setdefault("diag", {})["S3_site"] = note
        out = {"recorded": note}
        # WRONG-FILE CLASS (2026-08-13, cycle 5, pytest-7220): the ladder ran
        # end to end, S3 declared _code/code.py:_makepath with a plausible
        # reason, and gold edits nodes.py. fault_locations was EMPTY (pytest-
        # style failure, no crash frames), so nothing confronted the
        # declaration -- while nodes.py sat unread in the model's own locate
        # results (the 66%-of-misses pattern: the gold name passes through
        # the run). Confront it mechanically: non-test files the model's own
        # searches matched but it never read come back as a stated fact at
        # declare time. A fact, not a block -- declaring anyway stays legal,
        # it just cannot happen in ignorance any more.
        if os.environ.get("DIAG_GATE", "0") == "1":
            _read = set(state.get("files_read") or [])
            _decln = path.lstrip("./")
            _alts, _apats = [], []
            for _rec in (state.get("locate_files") or []):
                _hitp = False
                for _p in _rec.get("files", []):
                    if (_p != _decln and _p not in _read
                            and "test" not in _p.lower()
                            and _p not in _alts):
                        _alts.append(_p)
                        _hitp = True
                if _hitp and _rec.get("pattern") not in _apats:
                    _apats.append(_rec.get("pattern"))
            if _alts:
                out["alternatives"] = (
                    "your own locate search(es) (%s) ALSO matched these "
                    "non-test files, and you have READ NONE of them: %s. "
                    "The writer of the buggy state is often in one of these. "
                    "Read the relevant ones and re-declare, or proceed only "
                    "with a reason the writer cannot be there."
                    % (", ".join(repr(_x) for _x in _apats[:3]),
                       ", ".join(_alts[:6])))
                state["diag"]["S3_site"] = (
                    state["diag"]["S3_site"]
                    + " | unread alternatives shown: "
                    + ", ".join(_alts[:4]))
        # CYCLE-2 FINDING (django-11422): the ladder ran end to end and the
        # model still declared BaseReloader.__init__ while its own registered
        # traceback ran through iter_modules_and_files -- the gold function.
        # S1's evidence and S3's declaration were never connected. Connect
        # them mechanically: resolve the registered fault frames to their
        # enclosing functions; a declaration outside that set gets the list
        # back as a stated fact. A fact, not a block -- declaring elsewhere
        # stays legal, it just can no longer happen in ignorance.
        _fls = state.get("fault_locations") or []
        _stack = []
        for _fl_ent in _fls[:8]:
            try:
                _fp, _fline = str(_fl_ent).rsplit(":", 1)
                _nm = _enclosing_def(_fp, int(_fline))
            except Exception:
                continue
            if _nm and _nm not in [s.split(" ")[0] for s in _stack]:
                _stack.append("%s (%s)" % (_nm, _fl_ent))
        if func and _stack and func not in [s.split(" ")[0] for s in _stack]:
            out["stack_check"] = (
                "your registered reproduction's traceback ran through these "
                "repo functions: %s. Your declared site (%s) is NOT among "
                "them. The writer of the buggy state is usually in that "
                "stack -- reconsider the site, or proceed only with a reason "
                "the stack does not reach it." % ("; ".join(_stack), func))
        if role == "reader":
            out["caution"] = (
                "readers are usually the SYMPTOM site. If any function "
                "WRITES the state this reader consumes, the accepted fix is "
                "usually there -- confirm no writer candidate exists before "
                "editing here.")
        return out

    def h_reproduce(pcb, args):
        """Run a reproduction script. A script that exits NONZERO because of
        the bug becomes the registered reproduction (RED)."""
        script = str(args.get("python_script", ""))
        state["stuck"] = 0   # a probe ran -> unstick
        state["repro_attempts"] = state.get("repro_attempts", 0) + 1
        # REPRO CONTRACT (2026-08-11, gated REPRO_CONTRACT, default off).
        # Measured on the idiom retest: ADVICE ADOPTION IS PROPORTIONAL TO
        # HOW MECHANICAL THE ADVICE IS. The one-line "use Agg" idiom was
        # adopted from the first reproduction (5/7 instances); the structural
        # "render before asserting" idiom was adopted 2/7 -- one instance
        # wrote EIGHT non-rendering reproductions with the rule in CAPITALS
        # in its context. Same lesson as the 1767 ignored undo-refusal
        # advisories. So enforce the structure at the TOOL, the way h_patch
        # refuses test edits: the FIRST pyplot reproduction with no render is
        # refused with the reason; a resubmission is accepted unchanged,
        # because some matplotlib bugs (constructor raises) genuinely need no
        # draw and a hard block would trap those.
        if (os.environ.get("REPRO_CONTRACT", "0") == "1"
                and repo == "matplotlib/matplotlib"
                and not state.get("_contract_hinted")
                and re.search(r"pyplot|plt\.", script)
                and not re.search(r"savefig|canvas\.draw|draw_idle"
                                  r"|print_figure|draw\(\)", script)):
            state["_contract_hinted"] = True
            return {"error": (
                "not run: this reproduction never RENDERS. In matplotlib "
                "nothing is observable until the canvas draws -- property "
                "reads before a draw see the pre-render state and cannot go "
                "green over a correct fix. Add fig.canvas.draw() or "
                "fig.savefig(io.BytesIO(), format=png) before the "
                "assertion and resubmit. If this bug truly raises before "
                "any draw, resubmit the script UNCHANGED and it will run.")}
        if state.get("repro_locked") and _format_objects_to_repro(state):
            # One narrow exemption to the latch. The harness's own format
            # check has flagged a coined label, and that same label is the
            # acceptance assertion of the frozen reproduction -- so the
            # harness is simultaneously telling the agent to rename it and
            # refusing to let it. Observed on pallets__flask-5063 iter5:
            # three compliance attempts, three forced reverts, shipped the
            # flagged label. Allow the target to be rewritten once.
            state["repro_locked"] = False
            state["format_unlocks"] = state.get("format_unlocks", 0) + 1
        if state.get("repro_locked"):
            # The registered reproduction has already gone RED -> GREEN under
            # verify_fix. It is the only artifact that can satisfy the submit
            # gate, so it is immutable from here on: a later script that
            # happens to exit nonzero must not be allowed to overwrite it and
            # reset repro_green (observed: an inverted "demonstrate the bug"
            # script with an unconditional sys.exit(1) replaced a green
            # reproduction, made verify_fix structurally unreachable, and burned
            # the remaining budget to an empty diff).
            return {"error": (
                "reproduce LOCKED: the registered reproduction has already "
                "passed verify_fix, so it is now the fixed verification "
                "target and cannot be replaced. Writing another reproduction "
                "cannot advance the fix and risks discarding a verified "
                "state. If your patch is complete, call submit. If you changed "
                "the source since the last check, call verify_fix — it reruns "
                "the REGISTERED reproduction and is the only check that counts.")}
        if state.get("rejected_repro_streak", 0) >= 3 and state["seen_red"]:
            return {"error": (
                "reproduce PAUSED: your last 3 scripts all exited 0 while a "
                "failing reproduction is ALREADY registered. Writing more "
                "reproductions cannot advance the fix. Read fault_locations, "
                "change the PATCH, then verify_fix. reproduce unlocks after "
                "your next patch or read_range.")}
        _mode = "pytest" if (args.get("as_pytest")
                             or bool(re.search(r"^\s*def test_", script, re.M))
                             ) else "script"
        if _mode == "script":
            script = _mpl_force_draw(script, repo)
        r = _exec_repro(script, _mode, timeout=180)
        registered = False
        _invalid_pytest = (_mode == "pytest" and r.returncode == 5)  # no tests
        _tier, _why = ("unknown", "")
        if r.returncode != 0 and not _invalid_pytest:
            _tier, _why = _repro_quality(r)
        if (os.environ.get("REPRO_QUALITY") == "1" and _tier == "broken"
                and not state["seen_red"]):
            # The script did not get far enough to say anything about the bug.
            # Registering it would make it the permanent verification target --
            # exactly what cost django-16873 five repertoire segments.
            print("   -- REPRO_QUALITY refused a broken script (%s)" % _why,
                  flush=True)
            return {"exit": r.returncode,
                    "stdout": (r.stdout or "")[-2000:],
                    "stderr": (r.stderr or "")[-2000:],
                    "registered_as_reproduction": False,
                    "note": ("This script did not fail because of the bug -- it "
                             "failed to RUN (%s). A reproduction has to reach "
                             "the code under test and disagree with it, ideally "
                             "with an assert. Fix the script setup and run "
                             "reproduce again." % _why)}
        if r.returncode != 0 and not _invalid_pytest:
            state["repro_tier"] = _tier
            state["repro_script"] = script
            state["repro_mode"] = _mode
            state["seen_red"] = True
            state["repro_green"] = False
            state["rejected_repro_streak"] = 0
            registered = True
            state.setdefault("diag", {})["S1_reproduce"] = (
                "done: red registered (exit %s)" % r.returncode)
            if state["baseline_pass"] is None:   # pre-patch: valid baseline
                _hint_paths = [fl.split(":", 1)[0] for fl in
                               _repo_frames(r.stderr or "", repo_dir)]
                _capture_baseline(_hint_paths)
        result = {"exit": r.returncode,
                  "stdout": (r.stdout or "")[-2000:],
                  "stderr": (r.stderr or "")[-2000:],
                  "registered_as_reproduction": registered}
        _fl = _repo_frames(r.stderr or "", repo_dir)
        if _fl:
            result["fault_locations"] = _fl
            _fl_repo = _filter_repo_frames(repo_dir, _fl)
            if _fl_repo:
                state["fault_seen"] = True
                state["fault_locations"] = _fl_repo
                # ORDERING HOLE (cycle-2 rerun): declare_site at event 18,
                # first reproduce at event 20 -- a declaration made before
                # red exists dodges the declare-time stack check entirely.
                # So the check also runs HERE, when the evidence arrives:
                # a standing off-stack declaration gets the stack handed
                # back in this result and flagged in the diag record.
                _site = state.get("diag_site") or {}
                _sfn = _site.get("function")
                if _sfn and os.environ.get("DIAG_GATE", "0") == "1":
                    _stk = []
                    for _fe in _fl_repo[:8]:
                        try:
                            _fp, _fln = str(_fe).split(" ")[0].rsplit(":", 1)
                            _nm = _enclosing_def(_fp, int(_fln))
                        except Exception:
                            continue
                        if _nm and _nm not in _stk:
                            _stk.append(_nm)
                    if _stk and _sfn not in _stk:
                        result["site_check"] = (
                            "this traceback ran through: %s. Your declared "
                            "fix site (%s) is NOT among them -- the writer "
                            "of the buggy state is usually in this stack. "
                            "Consider declare_site again."
                            % (", ".join(_stk), _sfn))
                        state.setdefault("diag", {})["S3_site"] = (
                            str(state.get("diag", {}).get("S3_site", ""))
                            + " | OFF-STACK vs registered traceback (%s)"
                            % ", ".join(_stk))
        if registered:
            result["repro_tier"] = _tier
            result["note"] = ("This failing script is now the registered "
                              "reproduction. After you patch, verify_fix will "
                              "rerun EXACTLY this script — it must exit 0.")
            if os.environ.get("REPRO_STRENGTH") == "1":
                try:
                    import spec_probe as _sp
                    _rn = _sp.repro_assertion_note(script, r.stderr or "")
                    if _rn:
                        result["reproduction_strength_warning"] = _rn
                        print(" -- REPRO_STRENGTH: %s" % _rn[:110], flush=True)
                except Exception as _re:
                    print(" -- REPRO_STRENGTH error: %s: %s"
                          % (type(_re).__name__, _re), flush=True)
        elif not state["seen_red"]:
            result["note"] = ("Script exited 0 — the bug is not demonstrated. "
                              "Write a script that FAILS (nonzero exit, e.g. "
                              "an assert or uncaught exception) because of "
                              "the reported bug.")
        elif (os.environ.get("REPRO_PROMOTE") == "1"
              and state.get("promotions", 0)
                  < int(os.environ.get("REPRO_PROMOTE_MAX", "3") or 3)
              and _promote_if_red_at_base(script, _mode)):
            # It passed HERE and failed at BASE: a real reproduction that the
            # current patch fixes. Without this branch a correct replacement for
            # a wrong reproduction is unreachable -- see REPRO_PROMOTE note.
            state["repro_script"] = script
            state["repro_mode"] = _mode
            state["seen_red"] = True
            state["repro_green"] = True
            # A promotion IS a red->green transition, proven on both trees, so
            # it freezes the verification target exactly as verify_fix does.
            # Without this the freshly promoted reproduction could be
            # overwritten by any later script that happens to exit nonzero.
            state["repro_locked"] = True
            state["rejected_repro_streak"] = 0
            state["promotions"] = state.get("promotions", 0) + 1
            result["registered_as_reproduction"] = True
            result["note"] = (
                "PROMOTED. This script FAILS on the unmodified base tree and "
                "PASSES with your patch applied, so it is a valid reproduction "
                "that your fix resolves. It is now the registered reproduction "
                "and it is GREEN.")
        else:
            state["rejected_repro_streak"] = state.get("rejected_repro_streak", 0) + 1
            # Rejected AFTER a red reproduction is already registered. This
            # branch used to attach NO note, so the model got a bare
            # registered_as_reproduction:False with nothing to act on and
            # re-ran the same script until its budget died (pattern #6:
            # never reject a tool call without the specific diagnostic).
            # Say what happened AND name the tool that actually advances.
            _empty = not (r.stdout or "").strip()
            result["note"] = (
                "Script exited 0, so it was NOT registered"
                + (" (it also printed nothing — it may not be exercising the "
                   "bug at all)" if _empty else "")
                + ". That is not a problem: a reproduction is ALREADY registered "
                "from an earlier turn and it still stands. Calling reproduce "
                "again does NOT re-check your fix and will keep returning this. "
                "To check whether your patch works, call verify_fix — it reruns "
                "the REGISTERED reproduction and is the only check that counts. "
                "Call reproduce again ONLY if you mean to REPLACE the registered "
                "reproduction with a different script that FAILS (nonzero exit) "
                "because of the bug.")
        return result

    def h_locate(pcb, args):
        state["must_observe"] = False
        """grep for a pattern, then ask the LLM which hit is most likely the
        actual site to investigate."""
        pat = str(args.get("pattern", ""))
        glob_pat = args.get("file_glob") or ""
        # grep --include matches basenames only, so a path-style glob like
        # "lib/matplotlib/axis.py" silently matches nothing. Grep by the
        # basename component instead, then filter hits by path.
        path_glob = "/" in glob_pat
        if path_glob:
            base = glob_pat.rsplit("/", 1)[-1] or "*.py"
            cmd = f'grep -RIn --include={shlex.quote(base)} {shlex.quote(pat)} .'
        elif glob_pat:
            cmd = f'grep -RIn --include={shlex.quote(glob_pat)} {shlex.quote(pat)} .'
        else:
            cmd = f'grep -RIn --include="*.py" {shlex.quote(pat)} .'
        r = _run(cmd, timeout=60)
        lines = (r.stdout or "").splitlines()
        if path_glob:
            norm = glob_pat.lstrip("./")
            def _hit_ok(ln):
                hit_path = ln.split(":", 1)[0].lstrip("./")
                return (fnmatch.fnmatch(hit_path, norm)
                        or fnmatch.fnmatch(hit_path, "*/" + norm)
                        or hit_path.endswith(norm))
            lines = [ln for ln in lines if _hit_ok(ln)]
        lines = lines[:40]
        # CYCLE-6 (2026-08-13, pytest-7220 rerun): the alternatives fact
        # FIRED and was POISONED -- the locates had also matched across
        # venv/site-packages, and the capped alternatives list was consumed
        # entirely by setuptools internals, burying nodes.py (the gold file,
        # present in TWO of the model's own locate records). Third instance
        # of the junk-evidence law (_filter_repo_frames, GROW_ECHO): junk
        # silently drowns gates built to demand the real thing. Only repo
        # source files count.
        _JUNK = ("venv/", "site-packages/", ".tox/", "node_modules/",
                 ".egg-info", "__pycache__/", "build/lib", "dist/")
        _lf = []
        for _ln in lines:
            _p = _ln.split(":", 1)[0].lstrip("./")
            if (_p.endswith(".py") and _p not in _lf
                    and not any(_j in _p for _j in _JUNK)):
                _lf.append(_p)
        if _lf:
            _hist = state.setdefault("locate_files", [])
            _hist.append({"pattern": pat, "files": _lf[:12]})
            del _hist[:-6]
        result = {"matches": lines, "match_count": len(lines),
                  "truncated": len(lines) == 40}
        if len(lines) > 1:
            hits_blob = "\n".join(lines[:30])
            ranking = llm_call(
                system=("You rank grep hits by likelihood of being the actual "
                        "bug site vs test file / comment / unrelated match. "
                        "Answer JSON."),
                prompt=(f"grep pattern: {pat!r}\n\nHits:\n{hits_blob}\n\n"
                        'Return JSON: {"top_hit":"path/to/file.py:LINE", '
                        '"reason":"why this one", '
                        '"discard":["path:LINE reasons to skip"]}'))
            parsed = _extract_json(ranking) or {}
            # top_hit_only: keep the judge's DECISION, never its sampled prose.
            # The wording of "reason"/"discard" varies run-to-run even when the
            # decision is identical, and embedding it in the tool result makes
            # the whole downstream trajectory nondeterministic (observed:
            # determinism check diverged first at exactly this field).
            th = parsed.get("top_hit")
            if isinstance(th, str) and th.strip():
                result["top_hit"] = th.strip()
        return result

    def h_read_range(pcb, args):
        state["must_observe"] = False
        state["rejected_repro_streak"] = 0
        path = str(args.get("file", ""))
        start = max(1, int(args.get("start", 1)))
        end = int(args.get("end", start + 40))
        full = os.path.join(repo_dir, path)
        if not os.path.isfile(full):
            return {"error": _missing_file_hint(path, repo_dir)}
        _fr = state.setdefault("files_read", [])
        _np = path.lstrip("./")
        if _np not in _fr:
            _fr.append(_np)
            del _fr[:-50]
        try:
            with open(full, encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
        except OSError as e:
            return {"error": str(e)}
        total = len(lines)
        end = min(end, total)
        window = "".join(lines[start-1:end])
        return {"path": path, "start": start, "end": end, "total_lines": total,
                "content": window[:6000]}

    def _region_now(path, center=None, snippet=None, span=10):
        """Read the CURRENT bytes around a target and return them numbered.

        Every repeat/undo refusal in h_patch tells the model to go and look.
        Measured on sympy: 192 of 390 patch calls failed, and 145 of those
        were repeats or undos -- the model does not go and look. Worse, the
        must_observe branch DEADLOCKS: the harness demands an observation,
        the model patches again instead, and both sides repeat until the
        walk gives up. Handing back the bytes IS the observation, so the
        loop cannot survive on a stale mental model.

        Gated THRASH_ECHO, default off. Returns None when disabled or when
        the region cannot be located, in which case callers fall through to
        their original refusal unchanged.
        """
        if os.environ.get("THRASH_ECHO", "0") != "1":
            return None
        try:
            with open(os.path.join(repo_dir, path), encoding="utf-8",
                      errors="ignore") as _fh:
                _ls = _fh.read().splitlines()
            if center is None and snippet:
                _first = next((s.strip() for s in snippet.splitlines()
                               if s.strip()), "")
                if _first:
                    for _i, _l in enumerate(_ls, 1):
                        if _first in _l:
                            center = _i
                            break
            if center is None:
                return None
            _a = max(1, int(center) - span)
            _b = min(len(_ls), int(center) + span)
            if _b < _a:
                return None
            return {"file": path, "lines": "%d-%d" % (_a, _b),
                    "text": chr(10).join("%5d| %s" % (_n, _ls[_n - 1])
                                         for _n in range(_a, _b + 1))}
        except Exception as _e:
            print(" -- region echo failed: %s" % type(_e).__name__, flush=True)
            return None

    _SIB_KW = set(
        "def class return if else elif for while in not and or is None True "
        "False self import from as with try except pass raise lambda print "
        "len str int float bool dict list set tuple type object".split())

    def _sibling_sites(path, frag, limit=5):
        """SWEBENCH_MODE: deterministic within-file scan for OTHER places the
        same concept lives.

        Measured 2026-08-08 over 593 archived failed patches: on multi-hunk
        gold instances 58% are PARTIAL -- the model edited one correct site
        and never found the second -- against only 20% that missed entirely.
        Wrong-location rates are identical for wins and misses (12% each), so
        on multi-hunk bugs the failure is not misplacement, it is abandonment.

        The harness ALREADY tells the model its edit is "NECESSARY BUT NOT
        SUFFICIENT" on every undo refusal -- that message fired 1767 times and
        did not work, the same way "go and read_range" failed 145 times.
        Advice is not the missing piece; LOCATIONS are. So hand back the lines.

        This is Mikey's own sibling-site corollary made mechanical: a single
        grep of a file already open would have handed ornith the second site.
        """
        if os.environ.get("SWEBENCH_MODE", "0") != "1":
            return []
        try:
            toks = [t for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}",
                                          frag or "") if t not in _SIB_KW]
            if not toks:
                return []
            toks = list(dict.fromkeys(toks))
            with open(os.path.join(repo_dir, path), encoding="utf-8",
                      errors="ignore") as _fh:
                _ls = _fh.read().splitlines()
            done = {l.strip() for l in (frag or "").splitlines() if l.strip()}
            scored = []
            for _n, _l in enumerate(_ls, 1):
                if _l.strip() in done or not _l.strip():
                    continue
                hits = sum(1 for t in toks if t in _l)
                if hits:
                    scored.append((hits, -_n, _n, _l.strip()[:100]))
            scored.sort(reverse=True)
            return [(n, txt) for _h, _neg, n, txt in scored[:limit]]
        except Exception as _e:
            print(" -- sibling scan failed: %s" % type(_e).__name__, flush=True)
            return []

    def h_patch(pcb, args):
        """Surgical edit. Any successful patch invalidates prior verification —
        the reproduction must be rerun."""
        path = str(args.get("file", ""))
        old = str(args.get("old_snippet") or "")
        new = str(args.get("new_snippet", ""))
        state["last_edit_file"] = path
        # THE NEW SIDE ONLY. After the edit lands the site on disk matches
        # "new", so excluding new is what stops the just-edited line coming
        # back top-ranked as its own sibling (the bug caught end-to-end on
        # 2026-08-08).  Excluding "old" as well looks symmetric and is
        # actively harmful: a genuine sibling is usually a line whose text is
        # IDENTICAL to the one just fixed -- the same guard inside the next
        # operator dunder, the same comparison in the next _print_* method --
        # so excluding the old text suppresses exactly the lines the scan
        # exists to surface.  edit_line reaches h_patch in line mode with no
        # old_snippet, so it never hit this; snippet mode did.
        state["last_edit_frag"] = new or ""
        # DIAGNOSIS GATE (2026-08-11, gated DIAG_GATE, default off).
        # Mikey: "can we make these steps required in the program? ... a
        # general algorithm that will either try each step or say it's
        # unnecessary. And the results should be kept as state."
        # Every step ends in exactly one recorded outcome -- done, waived
        # with the reason, or skipped-after-warning -- in state["diag"],
        # which the worksheet renders every turn and the trace persists.
        # Each refusal is one-shot, so the ladder cannot deadlock: the
        # model is challenged once per step, never trapped (the
        # pallets-5063 format-latch lesson).
        if os.environ.get("DIAG_GATE", "0") == "1":
            _dg = state.setdefault("diag", {})
            _w = state.setdefault("_diag_warned", {})
            if "S1_reproduce" not in _dg:
                if state.get("seen_red"):
                    _dg["S1_reproduce"] = "done: red reproduction registered"
                elif state.get("repro_attempts", 0) >= 2:
                    _dg["S1_reproduce"] = (
                        "waived: %d attempts, bug not triggerable here"
                        % state.get("repro_attempts", 0))
                elif not _w.get("s1"):
                    _w["s1"] = True
                    return {"error": (
                        "DIAGNOSIS 1/3 -- REPRODUCE FIRST: no red "
                        "reproduction is registered (%d attempt(s) so far). "
                        "Write a script that fails BECAUSE of the bug. If it "
                        "truly cannot be triggered here, two attempts record "
                        "the waiver and this gate steps aside."
                        % state.get("repro_attempts", 0))}
                else:
                    _dg["S1_reproduce"] = "skipped after warning"
            if "S2_differential" not in _dg:
                if state.get("fault_seen"):
                    _dg["S2_differential"] = (
                        "waived: crash traceback names in-repo frames -- the "
                        "stack already points at the state and its writers")
                elif not _w.get("s2"):
                    _w["s2"] = True
                    return {"error": (
                        "DIAGNOSIS 2/3 -- DIFFERENTIAL: before editing, run "
                        "differential(bug_script=..., control_script=...) -- "
                        "the same operations WITHOUT the condition the issue "
                        "names. If the control is clean, the bug is in STATE "
                        "that condition changes, and the fix belongs in the "
                        "function that WRITES it. If a controlled comparison "
                        "is impossible for this bug, patching again records "
                        "this step as skipped.")}
                else:
                    _dg["S2_differential"] = "skipped after warning"
            if "S3_site" not in _dg:
                if not _w.get("s3"):
                    _w["s3"] = True
                    return {"error": (
                        "DIAGNOSIS 3/3 -- DECLARE THE SITE: call "
                        "declare_site(file=..., function=..., role=writer|"
                        "reader, reason=...) before editing. Candidate sites "
                        "are the functions that WRITE the state your "
                        "diagnosis found, ranked above the function where "
                        "the symptom appears.")}
                else:
                    _dg["S3_site"] = "skipped after warning"
            else:
                _site = state.get("diag_site") or {}
                _sf = _site.get("file")
                if (_sf and _sf != path
                        and not _w.get("site_" + path)):
                    _w["site_" + path] = True
                    return {"error": (
                        "declared fix site is %s but you are editing %s -- "
                        "either edit the declared site or declare_site again "
                        "(changing sites is allowed; the change is recorded)."
                        % (_sf, path))}
        if _is_test_path(path):
            return {"error": "refusing to edit a test file — fix the source, "
                             "not the tests"}
        full = os.path.join(repo_dir, path)
        if not os.path.isfile(full):
            return {"error": _missing_file_hint(path, repo_dir)}
        try:
            with open(full, encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except OSError as e:
            return {"error": str(e)}
        # NEIGHBOR_INJECT baseline ordering (2026-07-27). The neighbor baseline
        # must exist AND be pre-patch. Its only other capture point is inside
        # reproduce(), which ornith often does not call before its first patch
        # (52 reproduce vs 97 patch across compare3) -- so the inject silently
        # no-fired on every such instance. Capture HERE: still strictly before
        # any mutation, and the file being patched is the ideal proximity hint.
        # Deliberately NOT done inside neighbor_tests: there the patch is already
        # applied, so the broken neighbor would look base-failing and the
        # regression would be hidden.
        if (os.environ.get("NEIGHBOR_INJECT") == "1"
                and state.get("baseline_pass") is None):
            # Where in the PRE-patch file is this edit? start_line is optional
            # in the schema, so derive it from where old_snippet sits. The graph
            # was built on the pre-patch tree, so pre-patch lines are the right
            # coordinates.
            _gfiles = None
            if os.environ.get("GRAPH_HINTS") == "1":
                try:
                    _pl = args.get("start_line")
                    if not _pl and old:
                        _k = text.find(old)
                        if _k >= 0:
                            _pl = text[:_k].count("\n") + 1
                    if not _pl and old:
                        # h_patch matches the anchor FUZZILY, so an exact find
                        # misses whenever indentation or trailing whitespace
                        # differs. Compare the first real line stripped.
                        _tl = text.splitlines()
                        _first = next((x.strip() for x in old.splitlines()
                                       if x.strip()), "")
                        if _first:
                            for _n, _line in enumerate(_tl, 1):
                                if _line.strip() == _first:
                                    _pl = _n
                                    break
                        if not _pl:
                            # last resort: a def/class named anywhere in the
                            # snippet, found by its declaration in the file
                            import re as _re
                            _names = _re.findall(
                                r"^\s*(?:async\s+)?(?:def|class)\s+(\w+)",
                                old, _re.M)
                            for _nm in _names:
                                _rx = _re.compile(
                                    r"^\s*(?:async\s+)?(?:def|class)\s+%s\b"
                                    % _re.escape(_nm))
                                for _n, _line in enumerate(_tl, 1):
                                    if _rx.match(_line):
                                        _pl = _n
                                        break
                                if _pl:
                                    break
                    if _pl:
                        import graph_tools as _gt
                        _gfiles = _gt.test_files_near(repo_dir, path, _pl)
                        print("   -- GRAPH_HINTS %s:%s -> %d test file(s)"
                              % (path, _pl, len(_gfiles or [])))
                    else:
                        print("   -- GRAPH_HINTS skipped (no pre-patch line "
                              "for %s)" % path)
                except Exception as _e:
                    print("   -- GRAPH_HINTS error: %s: %s"
                          % (type(_e).__name__, _e))
            try:
                _capture_baseline([path], graph_files=_gfiles)
            except Exception:
                pass
            _bp = state.get("baseline_pass")
            print("   -- NEIGHBOR_INJECT baseline captured: %s tests (hint=%s)"
                  % (len(_bp) if _bp is not None else "FAILED", path))

        # STUCK CIRCUIT-BREAKER (env STUCK_ESCALATE=1). Consecutive non-progress
        # patches (failed anchor / undo-refusal / no-op) are the flail signature;
        # a real write or a check/reproduce/verify_fix resets `stuck`. Redirect at
        # >=3, hard-latch at >=5. Offline-validated: fires only on the two flails.
        if os.environ.get("STUCK_ESCALATE") == "1" and state.get("stuck", 0) >= 3:
            _fired(state, "stuck_break")
            if state.get("stuck", 0) >= 5:
                return {"error": (
                    "PATCH LOCKED: your last 5+ patch attempts made no progress "
                    "(anchor mismatch, or undo of an edit already in place). You "
                    "cannot patch again until you OBSERVE. Run reproduce() or "
                    "verify_fix to test the edits you already have."),
                    "do_this_instead": (
                        "reproduce() -> if still RED, the remaining fault is in a "
                        "DIFFERENT file than the one you keep editing; read_range "
                        "that other file. Do NOT re-send a patch to this file.")}
            return {"error": (
                "NO PROGRESS: your last 3 patches did not change the file (anchor "
                "did not match, or would undo an edit already in place). Stop "
                "re-sending patches -- the edit you need is either already in, or "
                "belongs in a different file."),
                "do_this_instead": (
                    "Run ONE check() that PRINTS the current value at the target, "
                    "or reproduce() to see if you are already done. A probe resets "
                    "this and lets you patch again.")}
        state["stuck"] = state.get("stuck", 0) + 1   # tentative; zeroed on a real write
        # LINE-ANCHORED MODE: two integers cannot be mis-escaped. Text anchors
        # containing a backslash fail 66% of the time (25/38 measured); line
        # anchors cannot fail that way at all.
        _sl, _el = args.get("start_line"), args.get("end_line")
        if _sl is not None:
            try:
                _sl = int(_sl); _el = int(_el if _el is not None else _sl)
            except (TypeError, ValueError):
                return {"error": "start_line/end_line must be integers"}
            _lines = text.splitlines(True)
            if _sl < 1 or _el < _sl or _el > len(_lines):
                return {"error": ("line range %d-%d is outside %s (file has %d "
                                  "lines). read_range first."
                                  % (_sl, _el, path, len(_lines)))}
            _replaced = "".join(_lines[_sl - 1:_el])
            _nl = chr(10)
            _ins = new if (not new or new.endswith(_nl)) else new + _nl
            _newtext = "".join(_lines[:_sl - 1]) + _ins + "".join(_lines[_el:])
            _prior, _h, _hist = _seen_before(state, path, _newtext)
            if _prior is not None and _hist and _h != _hist[-1]:
                _fired(state, "undo_refused")
                return {"error": (
                    "REFUSED: this patch would RETURN %s to a state it was "
                    "already in (visit %d of this run). The failure CHANGED "
                    "when you moved past that state -- which proves your edit "
                    "does something necessary. Undoing it reverses known "
                    "progress and re-creates the OLD failure. Your edit is "
                    "NECESSARY BUT NOT SUFFICIENT: keep it, and ALSO fix what "
                    "is still wrong. If a rule arrived with an earlier patch "
                    "result (applies_here), implement THAT construction now."
                    % (path, _prior + 1)),
                "already_tried": _anchor_record(state)}
            if not _hist:
                _remember_state(state, path, text)   # the pre-edit baseline
            with open(full, "w", encoding="utf-8") as f:
                f.write(_newtext)
            _remember_state(state, path, _newtext)
            state["repro_green"] = False
            state["fix_verified"] = False
            state["same_verify_count"] = 0
            state["last_verify_sig"] = None
            state["rejected_repro_streak"] = 0
            state["must_observe"] = False
            state["stuck"] = 0
            state["patch_attempts"] = state.get("patch_attempts", 0) + 1
            state["patch_history"].append({"file": path, "verdict": "unverified"})
            _syn = _syntax_check(full, path)
            _fired(state, "patch_line_anchored")
            out = {"edited": path, "mode": "line_anchored",
                   "lines": "%d-%d" % (_sl, _el),
                   "edited_line": _sl,
                   "you_replaced_exactly": _replaced,
                   "note": ("verification invalidated - run verify_fix. CHECK "
                            "you_replaced_exactly: if that is not the text you "
                            "meant to change, read_range again and re-patch the "
                            "correct lines.")}
            # MAGIC-STRING CHECK (Mikey) -- before the maps. A string written in a
            # discriminator position must AGREE with the package; measured fail
            # rate when a fix involves one: 75% vs 51%.
            try:
                import magic_tool as _mg
                _ms = _mg.check_snippet(repo_dir, new or "")
                if _ms:
                    _fired(state, "magic_string_check")
                    out["magic_strings"] = _ms
            except Exception:
                pass
            if _syn:
                out["syntax"] = _syn
            try:
                import symmap as _sm
                _fm = _sm.file_map(repo_dir, path, near_line=_sl)
                if _fm:
                    _fired(state, "map_anatomy")
                    out["file_map"] = {"you_edited_inside":
                                       _fm["enclosing"] or "(module level)",
                                       "at_line": _sl}
                _fl = _sm.flow_map(repo_dir, path, near_line=_sl)
                if _fl:
                    _fired(state, "map_value_flow"); out["value_flow"] = [{k: v for k, v in c.items() if v}
                                         for c in _fl[:2]]
                # COMPOSITION MAP: for code that BUILDS output, whether it composes
                # blocks or splices strings is decisive and STATIC. A function that
                # concatenates rendered text has already discarded the alignment
                # information it needs, and looking at the result never reveals that.
                _enc = (_fm or {}).get("enclosing")
                if _enc and any(w in (_enc or "").lower() or w in path.lower()
                                for w in ("print", "repr", "render", "format",
                                          "pretty", "latex", "str")):
                    try:
                        import compose_map as _cm
                        _cr = _cm.compose_map(full, _enc)
                        if _cr and _cr.get("verdict") and not _cr.get("error"):
                            _fired(state, "map_composition"); out["composition"] = {k: v for k, v in _cr.items() if v}
                            # FORM-SELECTION MAP (Mikey): which forms can this function produce
                            # and what CONDITION selects each. Wrong-form bugs live in the
                            # predicate, not the template.
                            import form_map as _fmap
                            _fr = _fmap.form_map(full, _enc)
                            if _fr and not _fr.get("error") and _fr.get("branches"):
                                _fired(state, "map_form_selection")
                                out["form_selection"] = _fr
                            # PREDICATE-CHURN ESCALATION: repeated edits to the same SELECTOR
                            # with failing verification mean no predicate alone can satisfy the
                            # requirement -- one of the routed-to branches must change as well.
                            _fe = state.setdefault("func_edits", {})
                            _fe[_enc] = _fe.get(_enc, 0) + 1
                            if _fe[_enc] >= 3 and _fr and _fr.get("branches"):
                                _fired(state, "escalation_two_site")
                                _targets = sorted({b.get("selects", "")[:60]
                                                   for b in _fr["branches"]
                                                   if "returns" in (b.get("selects") or "")})[:4]
                                out["escalation"] = {
                                    "situation": ("You have edited the conditions of this "
                                                  "form-selecting function %d times and verification "
                                                  "still fails. When each predicate edit fixes one "
                                                  "case and breaks another, NO predicate alone can "
                                                  "work: the partition you need does not exist while "
                                                  "the branches stay as they are." % _fe[_enc]),
                                    "do_this": ("WIDEN THE EDIT to two sites. Decide which inputs "
                                                "belong in which form, write the predicate for THAT "
                                                "partition, and then MODIFY THE BRANCH whose renderer "
                                                "cannot yet handle the inputs your predicate routes "
                                                "to it. The branch targets are listed in "
                                                "form_selection above -- the second edit belongs in "
                                                "one of them."),
                                    "branch_targets": _targets}
                            # CALLER MAP (Mikey): who calls the function you are editing, and
                            # how. A signature change is judged by its callers; the calling
                            # convention usually lives at the call sites, not in the body.
                            import caller_map as _cmap
                            _cl = _cmap.caller_map(repo_dir, _enc)
                            if _cl and _cl.get("callers"):
                                _fired(state, "map_callers")
                                out["callers"] = _cl
                    except Exception:
                        pass
                # KNOWLEDGE TRIGGERS (Mikey): rules tied to the TOOL, fired by symbolic
                # match on repo/path/function/argument -- so the rule arrives WITH the
                # action instead of decaying in a turn-0 wall of text.
                try:
                    import swe_triggers as _tg
                    _rules = _tg.fire(repo or "", "patch", path,
                                      (_fm or {}).get("enclosing") or "",
                                      (old or "") + " " + (new or ""))
                    if _rules:
                        _fired(state, "knowledge_trigger")
                        out["applies_here"] = _rules
                except Exception:
                    pass
                _research_maps(repo_dir, path, _sl, _fm, _fl)
            except Exception as _e:
                out["file_map_error"] = str(_e)[:120]
            return out

        _key = " ".join(old.split())[:400]
        _prior = state["failed_anchors"].get(_key)
        if _prior:
            _fired(state, "anchor_repeat_refused")
            state["must_observe"] = True     # the repeat IS the trigger
            return {"error": ("you already sent this exact snippet at turn %s "
                              "and it did not match. Sending it again cannot "
                              "work -- the file is not what you think it is."
                              % _prior["turn"]),
                    "already_tried": _anchor_record(state),
                    "do_this_instead": (
                        "STOP patching and go look. read_range the region "
                        "again (the file has changed if any earlier patch "
                        "landed), or check() to print the exact lines from "
                        "disk. Compare what you get against the snippet above "
                        "character by character -- the difference is usually "
                        "whitespace. Only patch once you have SEEN the real "
                        "text.")}
        if state.get("must_observe"):
            # THRASH BREAK (2026-08-08, gated THRASH_ECHO). This branch used to
            # deadlock: it demands an observation the model will not spend a
            # turn making, so it patches again and is refused again. Observe
            # FOR it -- hand back the current bytes and count that as the
            # observation, clearing the flag so the walk can progress.
            _reg = _region_now(path, snippet=old)
            if _reg is not None:
                state["must_observe"] = False
                return {"error": ("a previous anchor was refused as a repeat. "
                                  "Below is what that region ACTUALLY "
                                  "contains on disk right now. Your snippet "
                                  "did not match THIS text."),
                        "already_tried": _anchor_record(state),
                        "current_file_region": _reg,
                        "do_this_instead": ("anchor on text you can SEE above, "
                                            "or use edit_line with a line "
                                            "number taken from it. Do not "
                                            "resend the old snippet.")}
            return {"error": ("a previous anchor was refused as a repeat and "
                              "nothing has been observed since."),
                    "already_tried": _anchor_record(state),
                    "do_this_instead": ("read_range or check() first, then "
                                        "patch. One observation is enough.")}
        new_text, _how = _apply_edit(text, old, new)
        if new_text is not None:
            _prior, _h, _hist = _seen_before(state, path, new_text)
            if _prior is not None and _hist and _h != _hist[-1]:
                _fired(state, "undo_refused")
                return {"error": (
                    "REFUSED: this patch would RETURN %s to a state it was "
                    "already in (visit %d of this run). The failure CHANGED "
                    "when you moved past that state -- which proves your edit "
                    "does something necessary. Undoing it reverses known "
                    "progress and re-creates the OLD failure. Your edit is "
                    "NECESSARY BUT NOT SUFFICIENT: keep it, and ALSO fix what "
                    "is still wrong. If a rule arrived with an earlier patch "
                    "result (applies_here), implement THAT construction now."
                    % (path, _prior + 1)),
                "already_tried": _anchor_record(state)}
        if new_text is None:
            state["patch_attempts"] = state.get("patch_attempts", 0) + 1
            state["failed_anchors"][_key] = {
                "turn": state["patch_attempts"],
                "file": path, "head": old.strip().splitlines()[0][:90]
                        if old.strip() else "", "why": str(_how)[:120]}
            if len(state["failed_anchors"]) > 1:
                state["must_observe"] = True
            return {"error": _how, "already_tried": _anchor_record(state)}
        if not state.get("state_history", {}).get(path):
            _remember_state(state, path, text)   # pre-edit baseline
        with open(full, "w", encoding="utf-8") as f:
            f.write(new_text)
        _remember_state(state, path, new_text)
        state["stuck"] = 0
        state["repro_green"] = False
        state["fix_verified"] = False
        state["same_verify_count"] = 0
        state["last_verify_sig"] = None
        state["rejected_repro_streak"] = 0
        state["patch_attempts"] = state.get("patch_attempts", 0) + 1
        state["patch_history"].append(
            {"file": str(args.get("file", "?")), "verdict": "unverified"})
        _syn = _syntax_check(full, path)
        # Where did the replacement land? The harness knows; the model was not
        # required to say (start_line is optional in the schema). Runner-side
        # consumers -- GRAPHIFY_INJECT, sibling sweep -- need a line, and
        # deriving it here is the one place it is certainly correct.
        # The edit point is the first character where old and new text differ.
        # NOT new_text.index(new): a short generic snippet -- a bare
        # return, a lone pass -- occurs earlier in the file too, and
        # that lookup then silently returns the wrong line
        # (measured on django-15061: reported 772, actual 852).
        _i, _lim = 0, min(len(text), len(new_text))
        while _i < _lim and text[_i] == new_text[_i]:
            _i += 1
        _eline = text[:_i].count("\n") + 1 if new_text != text else None
        out = {"edited": path, "old_bytes": len(old), "new_bytes": len(new),
               "delta_bytes": len(new) - len(old), "edited_line": _eline,
               "match": _how, "note": "verification invalidated — run verify_fix"}
        if _syn:
            out["syntax"] = _syn
        # MAGIC-STRING CHECK (Mikey) -- before the maps. A string written in a
        # discriminator position must AGREE with the package; measured fail
        # rate when a fix involves one: 75% vs 51%.
        try:
            import magic_tool as _mg
            _ms = _mg.check_snippet(repo_dir, new or "")
            if _ms:
                _fired(state, "magic_string_check")
                out["magic_strings"] = _ms
        except Exception:
            pass
        # THE MAP, built right before/with every patch (Mikey): mechanical AST,
        # cached per checkout, outside the repo tree. No choice point -- the
        # agent cannot patch without receiving an observation of what it changed.
        try:
            _idx = text.find(old) if old else -1
            _line = text[:_idx].count("\n") + 1 if _idx >= 0 else None
            import symmap as _sm
            _fm = _sm.file_map(repo_dir, path, near_line=_line)
            if _fm:
                _rows = ["%5d  %s%s" % (
                    r["line"], r["sig"] or r["name"],
                    ("  -- " + r["doc"]) if r["doc"] else "")
                    for r in _fm["symbols"]]
                out["file_map"] = {
                    "you_edited_inside": _fm["enclosing"] or "(module level)",
                    "at_line": _line,
                    "anatomy": _rows,
                    "how_to_use": ("This is %s, mechanically. Before you patch "
                                   "again, use it: does the symbol you changed "
                                   "actually do what you assumed, and is the "
                                   "thing you are looking for defined elsewhere "
                                   "in this file? If your next move is another "
                                   "patch with no check in between, you are "
                                   "guessing." % path)}
            # what the code DOES WITH THE VALUE: containers written/read near
            # the edit, their key expressions, and whether the writers agree.
            _fl = _sm.flow_map(repo_dir, path, near_line=_line)
            if _fl:
                _fired(state, "map_value_flow"); out["value_flow"] = [
                    {k: v for k, v in c.items() if v} for c in _fl[:2]]
            # COMPOSITION MAP: for code that BUILDS output, whether it composes
            # blocks or splices strings is decisive and STATIC. A function that
            # concatenates rendered text has already discarded the alignment
            # information it needs, and looking at the result never reveals that.
            _enc = (_fm or {}).get("enclosing")
            if _enc and any(w in (_enc or "").lower() or w in path.lower()
                            for w in ("print", "repr", "render", "format",
                                      "pretty", "latex", "str")):
                try:
                    import compose_map as _cm
                    _cr = _cm.compose_map(full, _enc)
                    if _cr and _cr.get("verdict") and not _cr.get("error"):
                        _fired(state, "map_composition"); out["composition"] = {k: v for k, v in _cr.items() if v}
                        # FORM-SELECTION MAP (Mikey): which forms can this function produce
                        # and what CONDITION selects each. Wrong-form bugs live in the
                        # predicate, not the template.
                        import form_map as _fmap
                        _fr = _fmap.form_map(full, _enc)
                        if _fr and not _fr.get("error") and _fr.get("branches"):
                            _fired(state, "map_form_selection")
                            out["form_selection"] = _fr
                        # PREDICATE-CHURN ESCALATION: repeated edits to the same SELECTOR
                        # with failing verification mean no predicate alone can satisfy the
                        # requirement -- one of the routed-to branches must change as well.
                        _fe = state.setdefault("func_edits", {})
                        _fe[_enc] = _fe.get(_enc, 0) + 1
                        if _fe[_enc] >= 3 and _fr and _fr.get("branches"):
                            _fired(state, "escalation_two_site")
                            _targets = sorted({b.get("selects", "")[:60]
                                               for b in _fr["branches"]
                                               if "returns" in (b.get("selects") or "")})[:4]
                            out["escalation"] = {
                                "situation": ("You have edited the conditions of this "
                                              "form-selecting function %d times and verification "
                                              "still fails. When each predicate edit fixes one "
                                              "case and breaks another, NO predicate alone can "
                                              "work: the partition you need does not exist while "
                                              "the branches stay as they are." % _fe[_enc]),
                                "do_this": ("WIDEN THE EDIT to two sites. Decide which inputs "
                                            "belong in which form, write the predicate for THAT "
                                            "partition, and then MODIFY THE BRANCH whose renderer "
                                            "cannot yet handle the inputs your predicate routes "
                                            "to it. The branch targets are listed in "
                                            "form_selection above -- the second edit belongs in "
                                            "one of them."),
                                "branch_targets": _targets}
                        # CALLER MAP (Mikey): who calls the function you are editing, and
                        # how. A signature change is judged by its callers; the calling
                        # convention usually lives at the call sites, not in the body.
                        import caller_map as _cmap
                        _cl = _cmap.caller_map(repo_dir, _enc)
                        if _cl and _cl.get("callers"):
                            _fired(state, "map_callers")
                            out["callers"] = _cl
                except Exception:
                    pass
            # KNOWLEDGE TRIGGERS (Mikey): rules tied to the TOOL, fired by symbolic
            # match on repo/path/function/argument -- so the rule arrives WITH the
            # action instead of decaying in a turn-0 wall of text.
            try:
                import swe_triggers as _tg
                _rules = _tg.fire(repo or "", "patch", path,
                                  (_fm or {}).get("enclosing") or "",
                                  (old or "") + " " + (new or ""))
                if _rules:
                    _fired(state, "knowledge_trigger")
                    out["applies_here"] = _rules
            except Exception:
                pass
            _cerr = _research_maps(repo_dir, path, _line, _fm, _fl)
            if _cerr:
                out["map_collection_error"] = _cerr
        except Exception as _e:
            out["file_map_error"] = str(_e)[:120]
        return out

    def lock_probe(script):
        """Run the triage-written invariant probe once, pre-theory. RED (nonzero
        exit) locks it as the immutable verification target. GREEN means the
        probe does not demonstrate the bug -- discard it (recorded), fall back.
        Called by the runner, never exposed to the model as a tool."""
        script = (script or "").strip()
        if not script:
            return "none"
        try:
            r = _run(f"{env_dir}/bin/python -c {shlex.quote(script)}", timeout=180)
        except Exception as e:
            print(" -- probe errored at lock time (%s); discarded" % e, flush=True)
            return "error"
        if r.returncode != 0:
            state["probe_script"] = script
            state["probe_green"] = False
            print(" -- invariant probe LOCKED (red confirmed)", flush=True)
            return "locked"
        print(" -- invariant probe exited 0 pre-fix; discarded (bad probe)", flush=True)
        return "green_prefix"

    def h_verify_fix(pcb, args):
        """Rerun the registered reproduction. GREEN when it exits 0."""
        shutil.rmtree(os.path.join(repo_dir, ".hypothesis"), ignore_errors=True)
        state["stuck"] = 0   # a probe ran -> unstick
        if not state["repro_script"]:
            return {"ok": False,
                    "error": ("no registered reproduction — use reproduce() "
                              "with a script that fails because of the bug "
                              "BEFORE patching")}
        r = _exec_repro(state["repro_script"],
                        state.get("repro_mode", "script"), timeout=300)
        # pytest exit 5 (no tests collected) is not a pass; only exit 0 is green.
        green = (r.returncode == 0)
        state["repro_green"] = green
        if green:
            # First red -> green transition freezes the target (see h_reproduce).
            state["repro_locked"] = True
        if green and state.get("probe_script"):
            pr = _run(f"{env_dir}/bin/python -c "
                      f"{shlex.quote(state['probe_script'])}", timeout=180)
            state["probe_green"] = (pr.returncode == 0)
        regressed = _check_regressions() if green else []
        gate_ok = _gate()
        # WORKSHEET EVIDENCE (2026-08-08). verify_fix already RETURNS stdout to
        # the model, but only once -- it then decays into history while the
        # worksheet, which is regenerated every turn, reports only that the
        # failure repeated ("same failure x2") and never what it said. Persist
        # the last meaningful line so the evidence variant can re-surface it.
        _tl = [_l for _l in ((r.stdout or "") + chr(10)
                             + (r.stderr or "")).splitlines() if _l.strip()]
        state["last_verify_tail"] = _tl[-1].strip()[:160] if _tl else ""
        result = {"ok": green, "exit": r.returncode,
                  "stdout": (r.stdout or "")[-2000:],
                  "stderr": (r.stderr or "")[-1500:],
                  "regressions": regressed,
                  "gate": {"seen_red": state["seen_red"],
                           "repro_green": state["repro_green"],
                           "diff_nonempty": _diff_nonempty(),
                           "no_regressions": not regressed,
                           "fix_verified": gate_ok}}
        if state.get("probe_script"):
            result["probe_green"] = state.get("probe_green")
            result["gate"]["probe_green"] = state.get("probe_green")
            if green and state.get("probe_green") is False:
                result["probe_note"] = (
                    "Your reproduction passes, but the LOCKED issue-invariant "
                    "probe still FAILS. The probe tests the property the issue "
                    "itself states; your patch fixes your theory of the bug, "
                    "not the reported bug. Re-read the issue and the probe "
                    "output; do not submit until the probe is green.")
        if green:
            try:
                _d = _run("git diff", timeout=30)
                state["format_warning"] = _format_lint(_d.stdout or "")
            except Exception:
                state["format_warning"] = None
            if state.get("format_warning"):
                result_format_note = state["format_warning"]
            else:
                result_format_note = None
        else:
            result_format_note = None
        if result_format_note:
            result["format_check"] = result_format_note
        _strength = _reproduction_strength(state.get("repro_script") or "")
        result["repro_strength"] = _strength
        if green and _strength != "value_check":
            if _strength == "vacuous_constant":
                result["repro_note"] = (
                    "Your reproduction's only assertion is a constant (e.g. "
                    "assert True) and verifies nothing about the output; GREEN "
                    "here means only 'no exception was raised'. If this bug is "
                    "about producing a CORRECT value/format, rewrite the "
                    "reproduction to assert the EXPECTED result before submitting.")
            else:
                result["repro_note"] = (
                    "Your reproduction has no value assertion; GREEN here means "
                    "only that no exception was raised. If the bug is about "
                    "producing a CORRECT value/format (not just avoiding a crash), "
                    "add an assertion on the expected result before submitting.")
        if regressed:
            result["warning"] = (f"your patch broke {len(regressed)} test(s) "
                                 f"that passed before: {regressed[:3]} — a "
                                 "correct fix should not break working tests. "
                                 "Investigate before submitting.")
        if state["patch_history"]:
            state["patch_history"][-1]["verdict"] = (
                "repro GREEN" if green else "repro still red")
        if not green:
            _sig = _failure_sig(r.returncode, r.stderr)
            if _sig == state.get("last_verify_sig"):
                state["same_verify_count"] = state.get("same_verify_count", 1) + 1
            else:
                state["same_verify_count"] = 1
            state["last_verify_sig"] = _sig
            result["same_failure_as_last"] = state["same_verify_count"] > 1
            result["verify_attempt"] = state["same_verify_count"]
            if state["same_verify_count"] >= 3:
                result["note"] = (
                    "IDENTICAL failure %d times in a row. Verifying again "
                    "without changing the patch will return this same result. "
                    "Read fault_locations, change the PATCH, then verify."
                    % state["same_verify_count"])
            _fl = _repo_frames(r.stderr or "", repo_dir)
            if _fl:
                result["fault_locations"] = _fl
            result["diagnosis"] = llm_call(
                system=("You explain a failing reproduction for a bug-fix "
                        "agent. Be specific about the traceback and what to "
                        "change next."),
                prompt=(f"Reproduction script:\n{state['repro_script'][:1500]}\n\n"
                        f"stdout:\n{result['stdout']}\nstderr:\n{result['stderr']}\n\n"
                        "In 2-4 sentences: what does the failure show, which "
                        "code is the likely fault, what should the fix look "
                        "like?"))
        return result

    def h_run_tests(pcb, args):
        """Run existing suite test(s) as a regression check (NOT the gate).
        Delegates to the single deterministic test_runner."""
        tid = str(args.get("test_id", ""))
        if not tid:
            return {"error": "test_id required"}
        import test_runner as _tr
        res = _tr.run_tests(repo_dir, env_kind, [tid], env_vars=env_vars,
                            repo=repo, timeout=600, diagnose=True)
        out = {"ok": res["ok"], "exit": res["exit"],
               "stdout": res["stdout"], "installed": res.get("installed", [])}
        if res.get("diagnosis"):
            out["diagnosis"] = res["diagnosis"]
        return out

    def h_neighbor_tests(pcb, args):
        """Run the repo's EXISTING tests around the changed code so the agent can
        see (and repair) its own regressions. Reports how many nearby base-passing
        tests still pass and which the patch BROKE. Leak-safe: baseline_pass are
        tests that were green pre-patch, so the graded FAIL_TO_PASS (red pre-patch)
        is never in the set."""
        import test_runner as _tr
        base = state.get("baseline_pass") or []
        if not base:
            return {"error": ("no neighborhood baseline yet -- run reproduce() "
                              "first; that captures the nearby tests that pass "
                              "before your change.")}
        try:
            r = _tr.run_tests(repo_dir, env_kind, base, env_vars=env_vars,
                              repo=repo, timeout=300)
        except Exception as _e:
            return {"error": "could not run neighbor tests: %s" % type(_e).__name__}
        if r.get("ok"):
            return {"neighbor_tests": len(base), "regressed": 0,
                    "note": ("all %d existing tests around your change still pass "
                             "-- no regression in your blast radius." % len(base))}
        _f = re.findall(r"(?:FAILED|ERROR)\s+([^\s:]+(?:::\S+)?)", r.get("tail") or "")
        regressed = list(dict.fromkeys(_f))
        state["neighbor_regressed"] = regressed
        return {"neighbor_tests": len(base),
                "regressed": len(regressed) or "some",
                "which": regressed[:8],
                "detail": (r.get("tail") or "")[-700:],
                "note": ("your patch BROKE existing test(s) that passed before. "
                         "KEEP your fix and ALSO make these pass -- a fix that "
                         "regresses the neighborhood is not done.")}

    def h_check(pcb, args):
        state["must_observe"] = False
        state["stuck"] = 0   # a probe ran -> unstick
        """Answer ONE small question. Registers nothing, gates nothing."""
        snippet = args.get("snippet") or args.get("python") or ""
        if not snippet.strip():
            return {"error": "check needs a `snippet`: a few lines of python "
                             "that PRINT the one fact you want to know."}
        _pre = _run("git status --porcelain", timeout=30).stdout or ""
        # check() is READ-ONLY by contract. Snapshot the agent's in-progress
        # source patch so a snippet that reverts/overwrites it (e.g. a
        # `git checkout <file>` inside the check -- the known losing move,
        # observed looping scikit-learn-25638 to patch_bytes=0) is undone
        # automatically instead of silently wiping the WIP patch.
        _wip_snap = {}
        for _ln in _pre.splitlines():
            _f = _ln[3:].strip()
            if _f and not _is_test_path(_f):
                try:
                    with open(os.path.join(repo_dir, _f), "rb") as _fh:
                        _wip_snap[_f] = _fh.read()
                except OSError:
                    pass
        r = _run("%s/bin/python -c %s" % (env_dir, shlex.quote(snippet)),
                 timeout=60)
        state["checks_run"] = state.get("checks_run", 0) + 1
        _post = _run("git status --porcelain", timeout=30).stdout or ""
        out = {"exit": r.returncode,
               "stdout": (r.stdout or "")[-2000:],
               "stderr": (r.stderr or "")[-800:]}
        if not (r.stdout or "").strip() and r.returncode == 0:
            out["note"] = ("ran clean but printed nothing -- a check is only "
                           "useful if it PRINTS the fact. Add a print().")
        if _post != _pre:
            # the snippet MUTATED the repository -- check is for reading, and
            # edits through it bypass every safeguard patch provides.
            _fired(state, "check_modified_repo")
            # a read-only check must not be able to destroy the WIP patch:
            # restore the exact pre-check bytes of any source file it altered.
            _restored = []
            for _f, _blob in _wip_snap.items():
                _fp = os.path.join(repo_dir, _f)
                try:
                    with open(_fp, "rb") as _fh:
                        _cur = _fh.read()
                    if _cur != _blob:
                        with open(_fp, "wb") as _fh:
                            _fh.write(_blob)
                        _restored.append(_f)
                except OSError:
                    pass
            if _restored:
                out["patch_restored"] = (
                    "your check ALTERED source you had already patched (%s) -- "
                    "check() is read-only, so your in-progress patch was "
                    "automatically RESTORED. Never `git checkout`/revert your "
                    "own patch inside a check; use patch to change code."
                    % ", ".join(_restored[:4]))
            _changed = sorted({ln[3:].strip() for ln in _post.splitlines()}
                              - {ln[3:].strip() for ln in _pre.splitlines()})
            _tests = [f for f in _changed if _is_test_path(f)]
            if _tests:
                for f in _tests:
                    _run("git checkout -- %s" % shlex.quote(f), timeout=30)
                out["repo_mutation"] = (
                    "your snippet EDITED TEST FILES (%s). Those changes were "
                    "REVERTED: the scorer strips all test edits before running "
                    "the hidden tests, so a test you write can never count. "
                    "Fix the SOURCE." % ", ".join(_tests[:3]))
            else:
                out["repo_mutation"] = (
                    "your snippet MODIFIED the repository (%s). check() is for "
                    "READING; edits made here bypass syntax checking, the maps "
                    "and the anchor record. Use patch for changes. If you just "
                    "REVERTED your own patch: that is the known losing move -- "
                    "a correct patch with a red self-check scores as a WIN, a "
                    "reverted one scores as nothing."
                    % ", ".join(_changed[:4] or ["files changed"]))
        return out

    def h_submit(pcb, args):
        """Terminal call. HARD requirement: a real (non-test) diff. The internal
        checks (reproduction red->green, invariant probe) are ADVISORY -- they
        inform, they do not veto, because they are self-authored and a failing
        self-check on a CORRECT patch would otherwise make the agent undo good
        work (observed: an agent reverting to an empty diff after 73 turns)."""
        verified = _gate()          # still computed + recorded for ranking
        if not _diff_nonempty():
            return {"error": ("cannot submit: there is no change to submit. "
                              "The working tree has no non-test source diff. "
                              "Patch the source, then submit.")}
        unmet = []
        if not state["seen_red"]:
            unmet.append("no failing reproduction was ever registered")
        elif not state["repro_green"]:
            unmet.append("your reproduction is still red")
        if state.get("probe_script") and not state.get("probe_green"):
            unmet.append("the locked invariant probe is still red")
        # (The sibling gate used to live here. It never ran: `submit` maps to
        # RETURN in FIX_TOOL2SYS, so phase_run returns before dispatching
        # h_submit at all -- exactly the trap atlas_doc.py already warned
        # about. The real gate is _fix_gate in swe_agent_v2.py.)
        out = {"submitted": True, "summary": args.get("summary", ""),
               "fix_verified": verified}
        state["submitted"] = True
        if unmet:
            out["submit_advice"] = (
                "ACCEPTED with unmet internal checks: %s. These are advisory. "
                "If your patch is correct but the self-check cannot be made to "
                "pass (some bugs cannot be observed by a self-authored test), "
                "submitting is right -- do NOT revert a patch you believe is "
                "correct just to satisfy an internal check."
                % "; ".join(unmet))
        return out

    def h_edit_line(pcb, args):
        """LINE-SCOPED FRAGMENT EDIT (2026-08-08, gated EDIT_LINE=1).

        describe-don't-transcribe. The model names ONE line and the fragment
        on it to change; the harness writes the bytes. The model never retypes
        the line's indentation or its untouched remainder -- which is where
        xarray-5131 died: the working fix was produced 31 times across 12
        attempts and every green was refused because re-typing drifted
        whitespace, with hints proven powerless (4 of 4 failed).

        WHY NOT THE EXISTING patch MODES: old_snippet needs GLOBAL uniqueness,
        which a short fragment like '<' or 'None' never has; line mode makes
        you retype the whole line including indentation. Scoping uniqueness to
        ONE line makes short fragments safe -- and "change a comparison
        operator" is 73 of the 300 gold patches.

        FAILURE IS A LOUD NO-OP: a fragment that is missing, or repeated on
        that line, writes NOTHING and returns the line's exact bytes via
        repr() so tabs and trailing spaces are visible. Wrong input yields no
        edit rather than a corrupt file -- the opposite of line-range
        replacement, where a mistyped indent is written happily and surfaces
        later as an IndentationError.

        The candidate file is compiled IN MEMORY first, so an edit that would
        break the parse is refused rather than applied then reverted. The
        write itself is delegated to h_patch in line mode, so verification
        invalidation, neighbour baselines, sibling sweeps and trigger firing
        behave exactly as for a normal patch.
        """
        path = str(args.get("file", ""))
        try:
            ln = int(args.get("line"))
        except (TypeError, ValueError):
            return {"error": "line must be an integer (1-based); read_range "
                             "gives you the number"}
        old = str(args.get("old") or "")
        new = str(args.get("new", ""))
        if not old:
            return {"error": "old must be a non-empty fragment on that line"}
        if _is_test_path(path):
            return {"error": "refusing to edit a test file - fix the source, "
                             "not the tests"}
        full = os.path.join(repo_dir, path)
        if not os.path.isfile(full):
            return {"error": _missing_file_hint(path, repo_dir)}
        try:
            with open(full, encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except OSError as e:
            return {"error": str(e)}
        _lines = text.splitlines(True)
        if ln < 1 or ln > len(_lines):
            return {"error": ("line %d is outside %s (file has %d lines). "
                              "read_range first." % (ln, path, len(_lines)))}
        raw = _lines[ln - 1]
        body = raw.rstrip(chr(10))
        eol = raw[len(body):]
        hits = body.count(old)
        if hits == 0:
            return {"error": "fragment not found on line %d" % ln,
                    "line_is": repr(body),
                    "hint": "nothing was written; copy the fragment exactly "
                            "as it appears above"}
        if hits > 1:
            return {"error": ("fragment occurs %d times on line %d - lengthen "
                              "it until it is unique ON THAT LINE"
                              % (hits, ln)),
                    "line_is": repr(body),
                    "hint": "nothing was written"}
        new_body = body.replace(old, new, 1)
        if path.endswith(".py"):
            _cand = ("".join(_lines[:ln - 1]) + new_body + eol
                     + "".join(_lines[ln:]))
            try:
                compile(_cand, path, "exec")
            except SyntaxError as _se:
                return {"error": ("refused: that edit would stop %s parsing "
                                  "(%s at line %s)"
                                  % (path, _se.msg, _se.lineno)),
                        "would_have_been": repr(new_body),
                        "hint": "nothing was written"}
        res = h_patch(pcb, {"file": path, "start_line": ln, "end_line": ln,
                            "new_snippet": new_body + eol})
        if isinstance(res, dict) and not res.get("error"):
            res["edited_line"] = ln
            res["before"] = repr(body)
            res["after"] = repr(new_body)
        return res

    def _file_defs(path, limit=40):
        """Every def/class in the edited file, with line numbers.

        The harness cannot tell which names are SIBLINGS -- _print_sinc shares
        no token with _print_sinh, __mul__ none with __truediv__ -- and a token
        scan that tries to guess finds only re-uses of the SAME symbol. That
        semantic judgement is the model's strength, not the harness's. So split
        the work: the harness supplies the complete, mechanical list of names in
        the file; the model decides which of them are family.
        """
        if os.environ.get("SWEBENCH_MODE", "0") != "1":
            return []
        try:
            with open(os.path.join(repo_dir, path), encoding="utf-8",
                      errors="ignore") as _fh:
                out = []
                for _n, _l in enumerate(_fh.read().splitlines(), 1):
                    m = re.match(r"\s*(?:async\s+)?(def|class)\s+([A-Za-z_]\w*)",
                                 _l)
                    if m:
                        out.append((_n, m.group(2)))
                return out[:limit]
        except Exception as _e:
            print(" -- defs scan failed: %s" % type(_e).__name__, flush=True)
            return []

    state["_sibling_fn"] = _sibling_sites
    state["_defs_fn"] = _file_defs
    handlers = {
        "swe.reproduce":   h_reproduce,
        "swe.differential": h_differential,
        "swe.declare_site": h_declare_site,
        "swe.locate":      h_locate,
        "swe.read_range":  h_read_range,
        "swe.patch":       h_patch,
        "swe.edit_line":   h_edit_line,
        "swe.verify_fix":  h_verify_fix,
        "swe.run_tests":   h_run_tests,
        "swe.neighbor_tests": h_neighbor_tests,
        "swe.check":       h_check,
        "swe.submit":      h_submit,
    }
    handlers["_lock_probe"] = lock_probe   # runner-only; stripped from tool menu
    handlers["_diff_nonempty"] = _diff_nonempty  # runner-only; the fix-phase gate
    handlers["_check_regressions"] = _check_regressions  # runner-only; given tests
    def _sibling_sweep(old_snippet, new_snippet, edited_line=None, rel_path=None):
        """Runner-only. Classify the edit just applied and report UNCHANGED
        sites of the same class in the same file. repo_dir is only in scope
        here, which is why this lives beside the other runner-only handlers
        rather than in phase_run (same reason neighbor_tests does)."""
        import os as _os
        import code_probes as _cp
        if not rel_path:
            return {}
        path = _os.path.join(repo_dir, rel_path)
        if not _os.path.isfile(path):
            return {}
        # v2 (2026-07-28). v1 stored the LAST sweep result and was wiped by the
        # next unrelated patch, so by submit time the list was always empty and
        # the gate never fired once in 10 instances / 46 evaluations.
        # v2 tracks which CLASSES are live per file and RE-PROBES fresh every
        # time. Deliberately not tracking line numbers: any edit above a site
        # shifts them, and a stale line number would look like "fixed".
        # Self-clearing: once the sites are actually changed the probe stops
        # reporting them (it ignores already-normalised comparisons), so the
        # gate becomes satisfiable rather than a trap.
        live = state.setdefault("sibling_classes", {})
        classes = _cp.classify_edit(old_snippet, new_snippet)
        if classes:
            live[rel_path] = sorted(set(live.get(rel_path, [])) | set(classes))

        outstanding, edit_classes = [], set()
        for _rp in list(live):
            _p = _os.path.join(repo_dir, _rp)
            if not _os.path.isfile(_p):
                live.pop(_rp)
                continue
            _cls = live[_rp]
            _found = [f for f in _cp.probe_file(_p, only=_cls)
                      if f["probe"] in _cls]
            if not _found:
                live.pop(_rp)          # every site of this class is now clean
                continue
            edit_classes |= set(_cls)
            for f in _found[:8]:
                outstanding.append({"file": _rp, "line": f["line"],
                                    "fact": f["fact"], "code": f["code"][:120]})

        state["sibling_outstanding"] = (
            {"edit_class": sorted(edit_classes),
             "unchanged_same_class_sites": outstanding[:8]}
            if outstanding else {})
        return state["sibling_outstanding"]

    def _revert_tree():
        """Runner-only. Put the source back to base so the next operation in
        the repertoire starts from a clean tree instead of inheriting the
        previous attempt\'s damage. Tracked files only -- the venv and any
        scratch reproduction scripts are untracked and survive."""
        import subprocess as _sp
        _sp.run("git checkout -- .", shell=True, cwd=repo_dir,
                capture_output=True, timeout=60)
        state["probe_green"] = False
        state["repro_green"] = False      # the fix is gone; so is its evidence
        return {"reverted": True}

    def h_neighborhood(pcb, args):
        """Call graph around a symbol the agent has ALREADY located.

        Measured 2026-07-28: graphify `query` from an issue title does NOT find
        the bug (47 nodes, all test fixtures, gold target absent), but
        `affected` from a known symbol returns the caller chain with exact
        file:line immediately. So this exposes only the half that works --
        expansion, not search. Targets the bucket-A failure: fixed one site,
        never found the sibling (astropy-14365 added re.IGNORECASE and left
        `if v == "NO"` at line 309 untouched).

        Deterministic: tree-sitter AST, no LLM, no network. ~26s to build the
        graph once per checkout, cached thereafter.
        """
        import graph_tools as _gt
        sym = str(args.get("symbol", "")).strip()
        if not sym:
            return {"error": "symbol is required, e.g. symbol='_is_ignored_file'"}
        return _gt.neighborhood(repo_dir, sym,
                                depth=int(args.get("depth", 2) or 2))

    def _capture_diff():
        """Runner-only. The current source diff as text, so a candidate patch
        can survive the revert between repertoire operations."""
        import subprocess as _sp
        r = _sp.run("git diff", shell=True, cwd=repo_dir,
                    capture_output=True, text=True, timeout=60)
        return r.stdout or ""

    def _restore_diff(text):
        """Runner-only. Put a saved candidate back on a clean tree."""
        import subprocess as _sp, tempfile, os as _os
        _sp.run("git checkout -- .", shell=True, cwd=repo_dir,
                capture_output=True, timeout=60)
        if not (text or "").strip():
            return {"restored": False}
        fd, path = tempfile.mkstemp(suffix=".patch")
        with _os.fdopen(fd, "w") as fh:
            fh.write(text)
        r = _sp.run("git apply %s" % path, shell=True, cwd=repo_dir,
                    capture_output=True, text=True, timeout=60)
        _os.unlink(path)
        return {"restored": r.returncode == 0, "err": (r.stderr or "")[:200]}

    def _neighborhood_of_edit(rel_path, line):
        """Runner-only. Call-graph neighbourhood of the symbol just edited.

        Returns {} when there is nothing worth saying, so the caller injects
        nothing rather than injecting noise.
        """
        try:
            import graph_tools as _gt
            return _gt.neighborhood_of_edit(repo_dir, rel_path, line)
        except Exception:
            return {}

    def _seed_reproduction(problem_statement):
        """Runner-only. Register the REPORTER's reproduction, if they wrote one.

        Returns a dict describing what happened -- always, so a no-fire is
        visible in the log rather than silent.
        """
        try:
            import repro_extract as _rx
        except Exception as _e:
            return {"seeded": False, "why": "repro_extract unavailable: %s" % _e}
        if state.get("seen_red"):
            return {"seeded": False, "why": "a reproduction is already registered"}
        blocks = _rx.code_blocks(problem_statement or "")
        if not blocks:
            return {"seeded": False, "why": "no runnable code in the issue text"}
        skipped = []
        for blk in blocks[:3]:
            if blk["kind"] == "testcase":
                # needs a home inside the repo test package for its relative
                # imports; _exec_repro runs pytest from /tmp on purpose.
                skipped.append("testcase (needs in-repo placement)")
                continue
            src = blk["source"]
            try:
                r = _exec_repro(src, "script", timeout=180)
            except Exception as _e:
                skipped.append("exec failed: %s" % _e)
                continue
            if r.returncode == 0:
                skipped.append("%s exits 0 (does not show the bug)" % blk["kind"])
                continue
            tier, why = _repro_quality(r)
            if tier == "broken":
                skipped.append("%s is broken (%s)" % (blk["kind"], why))
                continue
            state["repro_script"] = src
            state["repro_mode"] = "script"
            state["repro_tier"] = tier
            state["seen_red"] = True
            state["repro_green"] = False
            state["seeded_from_issue"] = True
            if state.get("baseline_pass") is None:
                try:
                    _capture_baseline([fl.split(":", 1)[0] for fl in
                                       _repo_frames(r.stderr or "", repo_dir)])
                except Exception:
                    pass
            return {"seeded": True, "kind": blk["kind"], "tier": tier,
                    "why": blk["why"], "chars": len(src)}
        return {"seeded": False, "why": "; ".join(skipped) or "no usable block"}

    handlers["_seed_reproduction"] = _seed_reproduction         # runner-only
    def _sibling_body_check(rel_path, written_text, edited_line=None):
        """Runner-only. Did the model just paste a NEARBY function's body?

        Returns {} when there is nothing worth saying.
        """
        try:
            import sibling_body as _sb
            return _sb.check(repo_dir, rel_path, written_text, edited_line)
        except Exception:
            return {}

    handlers["_sibling_body_check"] = _sibling_body_check       # runner-only

    def _spec_probe(rel_path, written_text, edited_line=None):
        try:
            import spec_probe as _sp
            return _sp.probe(repo_dir, rel_path, edited_line, written_text)
        except Exception as _e:
            # Never swallow silently: a crash and an honest no-fire must not
            # look the same in the log. COVERAGE_GAP ran as a no-op for a
            # whole run behind exactly this pattern.
            print(" -- _spec_probe error: %s: %s" % (type(_e).__name__, _e),
                  flush=True)
            return None

    handlers["_spec_probe"] = _spec_probe                       # runner-only

    def _coverage_gap():
        """Added lines the registered reproduction never executes.

        Returns {relpath: [line, ...]} (capped) or {} when there is no gap
        or no way to measure one. Never raises. Stdlib `trace` only, so it
        works in every instance venv, conda 3.6 through uv 3.12.
        """
        try:
            script = state.get("repro_script")
            if not script:
                return {}
            d = _run("git -C %s diff --unified=0" % shlex.quote(repo_dir),
                     timeout=60).stdout or ""
            added, cur = {}, None
            for ln in d.splitlines():
                if ln.startswith("+++ b/"):
                    p = ln[6:].strip()
                    cur = p if p.endswith(".py") else None
                elif ln.startswith("@@") and cur:
                    m = re.search(r"\+(\d+)(?:,(\d+))?", ln)
                    if m:
                        s = int(m.group(1))
                        c = int(m.group(2) or 1)
                        added.setdefault(cur, set()).update(range(s, s + c))
            if not added:
                return {}
            import json as _json
            import tempfile
            fd, sp = tempfile.mkstemp(suffix="_llmos_cov_src.py", dir="/tmp")
            os.close(fd)
            open(sp, "w").write(script)
            fd, op = tempfile.mkstemp(suffix="_llmos_cov.json", dir="/tmp")
            os.close(fd)
            if state.get("repro_mode") == "pytest":
                runner = ("import pytest\n"
                          "t.runfunc(pytest.main, [%r, '-x', '-q'])\n" % sp)
            else:
                runner = ("import runpy\n"
                          "t.runfunc(runpy.run_path, %r, "
                          "run_name='__main__')\n" % sp)
            wrapper = (
                "import trace, json\n"
                "t = trace.Trace(count=1, trace=0)\n"
                "try:\n"
                + "".join("    " + l + "\n"
                          for l in runner.strip().splitlines() if l) +
                "except BaseException:\n"
                "    pass\n"
                "out = {}\n"
                "for (fn, line) in t.results().counts:\n"
                "    out.setdefault(fn, []).append(line)\n"
                "json.dump(out, open(%r, 'w'))\n" % op)
            _run("%s/bin/python -c %s" % (env_dir, shlex.quote(wrapper)),
                 timeout=420)
            try:
                cov = _json.load(open(op))
            except Exception:
                return {}
            finally:
                for f in (sp, op):
                    try:
                        os.remove(f)
                    except OSError:
                        pass
            hit = {}
            rd = os.path.realpath(repo_dir)
            for fn, lines in cov.items():
                rf = os.path.realpath(fn)
                if rf.startswith(rd + os.sep):
                    hit.setdefault(os.path.relpath(rf, rd),
                                   set()).update(lines)
            gap = {}
            for f, lns in added.items():
                missed = sorted(lns - hit.get(f, set()))
                try:
                    fl = open(os.path.join(repo_dir, f), encoding="utf-8",
                              errors="replace").read().splitlines()
                    missed = [l for l in missed
                              if l <= len(fl) and fl[l - 1].strip()
                              and not fl[l - 1].strip().startswith("#")]
                except OSError:
                    pass
                if missed:
                    gap[f] = missed[:12]
            return gap
        except Exception as _cge:
            print(" -- COVERAGE GAP error (returning empty): %s: %s"
                  % (type(_cge).__name__, _cge), flush=True)
            return {}

    handlers["_coverage_gap"] = _coverage_gap                   # runner-only

    def _diff_hygiene():
        try:
            import diff_hygiene as _dhm
            return _dhm.check(repo_dir)
        except Exception:
            return {}

    handlers["_diff_hygiene"] = _diff_hygiene                   # runner-only

    def _diff_repair(rel_path):
        try:
            import diff_hygiene as _dhm
            _n = _dhm.repair(repo_dir, rel_path)
            _n += _dhm.repair_syntax(repo_dir, rel_path)
            _n += _dhm.repair_wrap_block(repo_dir, rel_path)
            return _n
        except Exception:
            return 0

    handlers["_diff_repair"] = _diff_repair                     # runner-only
    handlers["_neighborhood_of_edit"] = _neighborhood_of_edit   # runner-only
    handlers["swe.neighborhood"] = h_neighborhood        # agent-facing
    handlers["_capture_diff"] = _capture_diff            # runner-only
    handlers["_restore_diff"] = _restore_diff            # runner-only
    handlers["_revert_tree"] = _revert_tree              # runner-only
    handlers["_sibling_sweep"] = _sibling_sweep          # runner-only
    handlers["_capture_baseline"] = _capture_baseline    # runner-only
    return handlers, state


FIX_TOOLS = [
    {"type": "function", "function": {
        "name": "check",
        "description": (
            "Answer ONE small question about the code, right now, cheaply. Runs "
            "a few lines of python in the venv and gives you back what they "
            "PRINT. Registers nothing and verifies nothing -- it cannot help or "
            "hurt your reproduction, so use it freely. THIS IS THE TOOL FOR "
            "CHECKING A MECHANISM: is this object the same object as that one "
            "(use `is`, print the ids); which branch does this flag actually "
            "take; what key is this really stored under; what does this "
            "attribute contain before and after. Run several small checks until "
            "you KNOW what is wrong -- do not run the whole reproduction "
            "repeatedly hoping to infer it. Cheap and specific beats expensive "
            "and vague."),
        "parameters": {"type": "object", "properties": {
            "snippet": {"type": "string", "description": (
                "A few lines of python that PRINT the fact you want. Keep it "
                "under ~15 lines and print explicitly.")}},
            "required": ["snippet"]}}},
    {"type": "function", "function": {
        "name": "reproduce",
        "description": (
            "Run a reproduction inside the (verified) venv that demonstrates the "
            "bug by EXITING NONZERO. The last failing reproduction becomes the "
            "registered one that verify_fix reruns after your patch. Do this "
            "FIRST. TWO INSTRUMENTS — pick the one that can actually show THIS "
            "bug: (a) default: a plain script (python -c) for crashes, wrong "
            "return values, exceptions; (b) as_pytest=true: your script is a "
            "PYTEST TEST FILE (define test_ functions), run through the project's "
            "own framework. Use as_pytest for bugs that only manifest INSIDE the "
            "framework — import machinery, collection, fixtures, plugin behavior, "
            "test-runner semantics — which a standalone script cannot reproduce."),
        "parameters": {"type": "object", "properties": {
            "python_script": {"type": "string",
                              "description": "the reproduction: a script (default) "
                                             "or a pytest test file (if as_pytest)"},
            "as_pytest": {"type": "boolean",
                          "description": "run the reproduction via `pytest` instead "
                                         "of `python -c`; use for framework-internal "
                                         "bugs (imports/collection/fixtures/plugins)"},
        }, "required": ["python_script"]}}},
    {"type": "function", "function": {
        "name": "locate",
        "description": (
            "grep across the repo for a symbol/message/pattern. Returns file:line "
            "matches (up to 40) plus an LLM ranking of the likeliest bug site."),
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string"},
            "file_glob": {"type": "string",
                          "description": "Optional glob to scope the search, e.g. '*.py'."},
        }, "required": ["pattern"]}}},
    {"type": "function", "function": {
        "name": "read_range",
        "description": (
            "Read lines [start, end] of a specific file. Follow locate — grep gives you "
            "the line number, read_range opens the exact window."),
        "parameters": {"type": "object", "properties": {
            "file":  {"type": "string"},
            "start": {"type": "integer"},
            "end":   {"type": "integer"},
        }, "required": ["file", "start", "end"]}}},
    {"type": "function", "function": {
        "name": "edit_line",
        "description": (
            "Change ONE fragment on ONE line. You give the line number and the "
            "exact fragment to replace; the harness rewrites the bytes. You do "
            "NOT retype the line's indentation or the rest of the line, so "
            "whitespace cannot drift. PREFER THIS over patch for in-place "
            "edits: changing an argument, an operator, a literal, a name. The "
            "fragment must occur EXACTLY ONCE on that line - it need NOT be "
            "unique in the file, which is what makes short fragments like '<' "
            "or 'None' usable. If it is missing or repeated, NOTHING is "
            "written and you get the line's exact bytes back. An edit that "
            "would stop the file parsing is refused, not applied. Any "
            "successful edit invalidates verification - rerun verify_fix."),
        "parameters": {"type": "object", "properties": {
            "file": {"type": "string"},
            "line": {"type": "integer",
                     "description": "1-based line number from read_range"},
            "old":  {"type": "string",
                     "description": "exact fragment on that line to replace"},
            "new":  {"type": "string",
                     "description": "replacement (empty string deletes it)"},
        }, "required": ["file", "line", "old", "new"]}}},
    {"type": "function", "function": {
        "name": "patch",
        "description": (
            "Replace source text in a SOURCE file (test files are refused). "
            "TWO WAYS TO ANCHOR. (a) old_snippet: must match EXACTLY and be "
            "unique. (b) start_line + end_line: replaces those lines outright. "
            "PREFER LINE ANCHORING whenever the text contains backslashes, "
            "quotes or escape sequences -- reproducing such a line verbatim "
            "through a JSON argument usually fails on over-escaping, while two "
            "line numbers cannot be mis-encoded. read_range gives you the "
            "numbers. Line mode returns the exact text it replaced so you can "
            "confirm you hit the right lines. Any patch invalidates "
            "verification — rerun verify_fix afterwards."),
        "parameters": {"type": "object", "properties": {
            "file":         {"type": "string"},
            "start_line":   {"type": "integer"},
            "end_line":     {"type": "integer"},
            "old_snippet":  {"type": "string"},
            "new_snippet":  {"type": "string"},
        }, "required": ["file", "new_snippet"]}}},
    {"type": "function", "function": {
        "name": "verify_fix",
        "description": (
            "Rerun the registered reproduction script. ok=true when it exits 0 "
            "(the bug no longer occurs). submit is only accepted after this "
            "passes on a script that previously FAILED."),
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "run_tests",
        "description": (
            "Run an existing test file or test id from the repo's suite as a "
            "REGRESSION check (did my patch break something nearby?). This is not "
            "the verification gate — verify_fix is."),
        "parameters": {"type": "object", "properties": {
            "test_id": {"type": "string",
                        "description": "e.g. 'path/to/test_file.py::test_name'"},
        }, "required": ["test_id"]}}},
    {"type": "function", "function": {
        "name": "submit",
        "description": (
            "Terminal call. ONLY accepted after: a reproduction failed (RED), you "
            "patched source, and verify_fix passed (GREEN) with a non-empty diff."),
        "parameters": {"type": "object", "properties": {
            "summary": {"type": "string",
                        "description": "1-3 sentence summary of the fix."},
        }, "required": ["summary"]}}},
    {"type": "function", "function": {
        "name": "differential",
        "description": (
            "DIAGNOSIS: run the SAME operations twice -- once under the "
            "condition the issue names (bug_script) and once WITHOUT it "
            "(control_script). If the exits differ, that condition is "
            "load-bearing: the bug lives in STATE it changes, and the fix "
            "site is usually the function that WRITES that state, not the "
            "one where the symptom appears."),
        "parameters": {"type": "object", "properties": {
            "bug_script": {"type": "string"},
            "control_script": {"type": "string",
                               "description": "same operations with the "
                                              "issue-named condition removed"},
        }, "required": ["bug_script", "control_script"]}}},
    {"type": "function", "function": {
        "name": "declare_site",
        "description": (
            "DIAGNOSIS: declare WHERE the fix will go and WHY, before "
            "patching. role='writer' if the function writes the state your "
            "diagnosis found, 'reader' if it only consumes it. Recorded as "
            "state, shown every turn; edits outside the declared file are "
            "challenged once."),
        "parameters": {"type": "object", "properties": {
            "file": {"type": "string"},
            "function": {"type": "string"},
            "role": {"type": "string", "enum": ["writer", "reader"]},
            "reason": {"type": "string"},
        }, "required": ["file", "reason"]}}},
]

# EDIT_LINE gate (2026-08-08, default OFF). The tool menu is a behaviour
# surface -- adding a tool changes routing for every instance -- so this ships
# switched off like BANK_AUDIT and RANK_ORACLE. The handler stays registered
# either way; only the advertised schema is withheld.
if os.environ.get("EDIT_LINE", "0") != "1":
    FIX_TOOLS = [_t for _t in FIX_TOOLS
                 if _t.get("function", {}).get("name") != "edit_line"]

# DIAG_GATE menu gate: the diagnosis tools appear only when the ladder is on.
if os.environ.get("DIAG_GATE", "0") != "1":
    FIX_TOOLS = [_t for _t in FIX_TOOLS
                 if _t.get("function", {}).get("name") not in
                 ("differential", "declare_site")]


FIX_TOOL2SYS = {
    "check": "swe.check",
    "reproduce":   "swe.reproduce",
    "differential": "swe.differential",
    "declare_site": "swe.declare_site",
    "locate":      "swe.locate",
    "read_range":  "swe.read_range",
    "patch":       "swe.patch",
    "edit_line":   "swe.edit_line",
    "verify_fix":  "swe.verify_fix",
    "run_tests":   "swe.run_tests",
    "neighbor_tests": "swe.neighbor_tests",
    "neighborhood":   "swe.neighborhood",
    "submit":      "RETURN",   # terminal
}


FIX_SYSTEM_PROMPT = (
    "The environment is verified and ready. YOUR JOB: produce a patch to the SOURCE that fixes the bug, then submit. ONLY the project's real tests decide pass/fail -- reproduce, check, verify_fix and run_tests are OPTIONAL scaffolding to help YOU; use them or skip them, it does not matter. Do NOT spend turns reading or exploring without patching: the moment you can name the fix, make the edit. If a patch, reproduction, or check comes back with a SYNTAX ERROR or is incomplete, FIX that one error and re-run it -- NEVER abandon it and go back to exploring. When in doubt between reading more and patching, PATCH; you can always adjust after. A suggested loop:\n"
    "  1. reproduce — write a reproduction that FAILS (nonzero exit: uncaught "
    "exception or assert) because of the reported bug. This registers your "
    "reproduction. If your script exits 0, it does not demonstrate the bug — "
    "rewrite it.\n"
    "  2. locate — reproduce and verify_fix return fault_locations (the "
    "in-repo traceback frames); read_range those FIRST. Only grep for a "
    "symbol/message if fault_locations is empty or insufficient.\n"
    "  3. read_range — open the exact window around the match.\n"
    "  4. patch — surgical replacement in SOURCE files, small and targeted. "
    "Test files are refused.\n"
    "  5. verify_fix — reruns YOUR registered reproduction; it must now exit 0.\n"
    "  6. run_tests — run a nearby existing test file to check you broke nothing.\n"
    "  7. If verify_fix still fails, return to step 3 with the new evidence.\n"
    "  8. submit — requires a real source diff. The reproduction/probe checks "
    "are ADVISORY: if your patch is correct but the self-check cannot be "
    "made to pass, submit anyway. NEVER revert a patch you believe is "
    "correct in order to satisfy an internal check.\n\n"
    "NAMING: when your change produces user-facing text -- a column header, "
    "a label, a message, a key -- name it after the FIELD it displays, using "
    "the codebase's own vocabulary, NOT the issue reporter's wording. If a "
    "column shows the value of `rule.subdomain`, its header is \"Subdomain\", "
    "not a synonym you invented. When an object exposes ALTERNATIVE fields "
    "for the same slot (e.g. `rule.host` vs `rule.subdomain`, chosen by a "
    "mode flag), read the code to find EVERY such field and cover each with "
    "its own field-derived label. The maintainers name things after their "
    "own attributes. "
    "THE SAME APPLIES WHEN THE ISSUE PROPOSES AN API -- a new parameter, "
    "keyword or option. The reporter's suggested name is a SUGGESTION, not a "
    "specification; maintainers name things to match their own code. Before "
    "adopting a proposed name, read the signature you are extending and the "
    "options already on it: if they are booleans named for the property they "
    "control, add a boolean named for the property -- do not add a string "
    "mode borrowed from a builtin just because the reporter wrote it that "
    "way. Match the type, shape and naming style of what is already there.\n\n"
    """BUG SHAPE -- before you patch, name which of these the defect is. Most bugs are ONE decision with its boundary in the wrong place, and the shape tells you what to change:
  - guard too loose: a check accepts a case it should reject -- returns a value where it should raise, or applies a rewrite that is not always valid. Fix: ADD the clause that excludes the bad case.
  - guard too tight: a check rejects a case it should accept -- crashes on valid input, or a character, type or branch is not handled. Fix: WIDEN the check or add the missing branch.
  - wrong branch / fall-through: an operation returns an identity or default (1, 0, None, or the input unchanged) because it took a default path. Fix: find the dispatch (an __mul__, an isinstance chain) and repair the branch for this operand.
  - lost grouping / precedence: a built result drops parentheses it needed, so it means something else (a/b/c instead of a/(b/c)). Fix: parenthesize the sub-expression the construction rule left unwrapped.
  - mode divergence: the same input gives different results down two paths (evaluate=True vs False, two entry points, two orders). Fix: find the rule that fires on only one path and make both paths agree.
To localize any of these: reproduce, read the fault_locations, and find the SINGLE decision -- a comparison, an isinstance, a branch, a parenthesization -- whose boundary is wrong, and in which direction. Change that boundary, not the surrounding logic. If you cannot say in one sentence which decision is wrong and which way it should move, you have not localized yet.

"""
    """MINIMAL FIRST -- before anything fancier, your first verified candidate MUST be the bare fix: the single smallest edit that addresses the reported behavior, with NOTHING added -- no extra branch, no defensive case, no refactor, no rename. Then confirm it TWO ways before you trust it: verify_fix (your own reproduction goes green) AND run_tests on a neighboring existing test file (proof you did not break OTHER cases -- a fix can pass the reported case while quietly regressing others). These two checks are ADVISORY -- only the project tests decide, and a real source diff is ALL that submit requires. Once you have a diff you believe fixes the bug, submit AS-IS: do NOT improve, generalize, harden, or keep exploring. If a check fails because your SCRIPT is malformed, fix the script and re-run; if it fails because the PATCH is wrong, adjust the patch one element at a time -- but a failing self-check NEVER means revert the patch and go back to reading. A fix the issue reporter SUGGESTS -- a proposed diff, snippet or one-liner -- is a HYPOTHESIS to test exactly this way, never code to copy on trust: suggested fixes are frequently over- or under-broad.

"""
    "Make the smallest change that fixes the issue. Every turn MUST call "
    "exactly one tool."
)


# --- Blast-radius discipline (A/B, env-gated 2026-07-26) ---------------------
if os.environ.get("BLAST_RADIUS") == "1":
    FIX_SYSTEM_PROMPT = FIX_SYSTEM_PROMPT + (
        "\n\nCHECK YOUR BLAST RADIUS BEFORE YOU SUBMIT:\n"
        "A fix that makes the bug's test pass but breaks another existing test is "
        "NOT a fix. You have a `neighbor_tests` tool that runs the repo's own "
        "tests around the code you changed. After you patch and before you "
        "submit, call neighbor_tests. If it reports a regression, you broke a "
        "test that passed before -- KEEP your fix and ALSO repair it (your change "
        "altered behavior or a signature it depends on). Submit only once the bug "
        "is fixed AND the neighborhood still passes.")


# --- Scientific-debugging discipline (A/B, env-gated 2026-07-26) --------------
if os.environ.get("ISOLATE_DISCIPLINE") == "1":
    FIX_SYSTEM_PROMPT = FIX_SYSTEM_PROMPT + (
        "\n\nOVERRIDE -- DEBUG BY EXPERIMENT, NOT BY READING:\n"
        "Do NOT theorize from reading code. Every hypothesis about the bug is "
        "cheap to test, so TEST it. Loop:\n"
        "  (a) state ONE hypothesis: 'the fault is in <function>, which returns "
        "X but should return Y'.\n"
        "  (b) IMMEDIATELY run a `check` probe that constructs the minimal input "
        "and PRINTS the suspect value (call the function; print what it returns "
        "vs. what it should). One probe that prints the divergence is worth ten "
        "file reads.\n"
        "  (c) read the probe OUTPUT; keep or revise the hypothesis.\n"
        "  (d) patch ONLY after a probe has pinned the fault to a specific line "
        "or branch -- then verify with another probe.\n"
        "HARD RULES: never read_range more than TWICE in a row without running a "
        "probe in between; never patch a location you have not first confirmed "
        "with a probe that printed the wrong value there. A 5-line experiment "
        "that prints an intermediate value beats re-reading the source or "
        "re-running the full suite. If you have reasoned more than a few "
        "sentences without running a probe, stop and run one.")



# ---- recall tool (2026-07-25): retrieve the full output of an earlier call ----
from repo_bootstrap_tools import RECALL_TOOL as _RECALL_TOOL
FIX_TOOLS = FIX_TOOLS + [_RECALL_TOOL]
FIX_TOOL2SYS["recall"] = "recall"

# CYCLE-1 FINDING (2026-08-12, single-example loop, matplotlib-23299 rerun):
# with DIAG_GATE on, the model never patched at all -- 25 locate calls, zero
# edits, ladder tools untouched. The gate fires AT the edit, so a run that
# never edits never meets it, and nothing told the model the ladder exists.
# Enforcement without navigation. So when the gate is on, the system prompt
# names the ladder explicitly, up front.
if os.environ.get("DIAG_GATE", "0") == "1":
    FIX_SYSTEM_PROMPT = FIX_SYSTEM_PROMPT + (
        "\n\nDIAGNOSIS LADDER (enforced -- your patches are gated on it): "
        "BEFORE your first patch, complete or explicitly pass three steps, "
        "in order, EARLY:\n"
        "  A. reproduce -- register a RED script (fails because of the bug).\n"
        "  B. differential -- run the SAME operations WITHOUT the condition "
        "the issue names; if the control is clean, the bug is in STATE that "
        "condition changes and the fix belongs in the function that WRITES "
        "that state. (Auto-waived when a crash traceback already names "
        "in-repo frames.)\n"
        "  C. declare_site(file=..., function=..., role=writer|reader, "
        "reason=...) -- say where the fix goes and why. Writers outrank "
        "readers.\n"
        "The worksheet shows the ladder state each turn under diagnosis. "
        "Long locate/read_range exploration before the ladder is wasted "
        "budget: run A-B-C, then patch.")

# ---- neighbor_tests tool (BLAST_RADIUS, 2026-07-26): run the existing tests
# around the changed code so the agent sees and fixes its own regressions -------
_NEIGHBOR_TOOL = {"type": "function", "function": {
    "name": "neighbor_tests",
    "description": (
        "Run the repo's EXISTING tests around the code you changed, to see your "
        "blast radius. Reports which nearby base-passing tests still pass and "
        "which your patch BROKE. Call it AFTER you patch and before you submit: a "
        "fix that makes the bug's test pass but breaks a neighboring test is NOT "
        "done. Requires a registered reproduction first (that captures the "
        "baseline)."),
    "parameters": {"type": "object", "properties": {}}}}
if os.environ.get("BLAST_RADIUS") == "1":
    FIX_TOOLS = FIX_TOOLS + [_NEIGHBOR_TOOL]

_NEIGHBORHOOD_TOOL = {"type": "function", "function": {
    "name": "neighborhood",
    "description": (
        "Given a symbol you have ALREADY found (from locate, or a traceback "
        "frame), list everything that calls, imports or references it, and what "
        "it calls -- with exact file:line. Use it after you locate the fault to "
        "find the OTHER places that need the same change: a fix often has more "
        "than one site and the traceback names only the first. This is a lookup "
        "in a precomputed code graph, not a search -- give it a real symbol "
        "name, not a description."),
    "parameters": {"type": "object", "properties": {
        "symbol": {"type": "string",
                   "description": "exact symbol name, e.g. '_is_ignored_file'"},
        "depth":  {"type": "integer",
                   "description": "traversal depth, default 2"},
    }, "required": ["symbol"]}}}

if os.environ.get("GRAPHIFY") == "1":
    FIX_TOOLS = FIX_TOOLS + [_NEIGHBORHOOD_TOOL]
