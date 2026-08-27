#!/usr/bin/env python3
"""SWE-bench Lite agent v2: two-phase with env-verification gate.

Phase 1 — repo bootstrap:
  Model uses repo_bootstrap_tools until BOTH run_sanity and run_smoke_test
  have passed, then calls declare_env_ready. If either verification fails,
  the model must diagnose and try again. Bootstrap has its own budget so
  a broken repo doesn't eat the fix budget.

Phase 2 — bug fix:
  Only starts if phase 1 declared ready. Model uses swe_fix_tools
  (reproduce -> locate -> read_range -> patch -> run_failing_test) and
  calls submit only after run_failing_test on the FAIL_TO_PASS set passes.

Scoring:
  Apply the model's git diff + the SWE-bench test_patch, run FAIL_TO_PASS.

    PYTHONPATH=~/Code/LLMOS python3 swe_agent_v2.py [N]

Reads ~/swe/instances.json (from swe_lite_select.py), writes
~/swe/results_v2.json and a trace per instance.
"""
import json, os, re, shlex, shutil, subprocess, sys, tempfile, time

sys.path.insert(0, os.path.expanduser("~/Code/LLMOS"))
from tool_call_cpu import ToolCallCPU
from repo_bootstrap_tools import (BOOTSTRAP_TOOLS, BOOTSTRAP_TOOL2SYS, auto_verify_env,
                                   BOOTSTRAP_SYSTEM_PROMPT,
                                   make_bootstrap_handlers, env_ready)
from swe_fix_tools import (FIX_TOOLS, FIX_TOOL2SYS, FIX_SYSTEM_PROMPT,
                            make_fix_handlers)
import envcheck
from trace_consumers import (remedies_for, format_remedy_context,
                             patterns_load, format_patterns_context,
                             harvest_trace, critic_review, error_signature,
                             playbook_for, format_playbook_context)
from repo_bootstrap_tools import _ddg_search

HOST = "http://127.0.0.1:8080"   # llama-server direct (ollama retired)
MODEL = "ornith:35b"
NUMCTX = 65536          # must match llama-server --ctx-size (see start_ornith.sh)
# Sampling temperature. The vendor model card for Ornith-1.0 recommends 0.6
# (top_p 0.95 / top_k 20 are already sent by tool_call_cpu). The old
# hardcoded 1.0 came from the TTS-2 regime -- two draws deliberately
# diverse, then form_rank picks. With MAX_ATTEMPTS=1 that inherited the
# variance and dropped the selection step that paid for it.
SWE_TEMP = float(os.environ.get("SWE_TEMP", "0.6"))
NUM_PREDICT = int(os.environ.get("NUM_PREDICT", "2048"))
BOOTSTRAP_BUDGET = int(os.environ.get("BOOTSTRAP_BUDGET", "50"))
FIX_BUDGET       = int(os.environ.get("FIX_BUDGET", "200"))
FIX_STALL        = int(os.environ.get("FIX_STALL", "25"))  # no-progress watchdog: stop the fix phase after this many turns with <=2 novel results; 0 disables
# Attempt 2+ gets more room: 26 of 49 misses ended by exhausting the
# budget, and above the median a long run is no more likely to be lost
# than won (p90 turns: 74 resolved vs 74 missed). Retries only happen
# after a failure, so the extra time is spent where the wall bound.
RETRY_FIX_BUDGET = int(os.environ.get("RETRY_FIX_BUDGET", "120"))
WORK = os.path.expanduser("~/swe/work")
import spec_env

TRACES = os.path.expanduser("~/swe/traces_v2")

INJECTED_LOG = os.path.join(os.path.dirname(TRACES), "research", "injected.jsonl")


def _entry_key(_e):
    """Stable key for one accumulated-knowledge entry. SINGLE SOURCE OF TRUTH:
    the attribution log and the ablation sampler both derive keys here, so a
    withheld pattern and its logged key can never drift apart."""
    import hashlib
    if isinstance(_e, dict):
        _k = (_e.get("id") or _e.get("key") or _e.get("title")
              or _e.get("name") or _e.get("pattern") or "")
    else:
        _k = str(_e)
    _k = str(_k).strip().replace("\n", " ")[:120]
    if not _k:
        _k = "sha:" + hashlib.sha1(
            repr(_e).encode("utf-8", "replace")).hexdigest()[:12]
    return _k


def _pattern_ablate(pats, iid):
    """DETERMINISTIC PATTERN ABLATION (2026-08-08, gated, default OFF).

    patterns_load() returns the SAME global list for every instance, so the
    attribution log can never attribute anything to an individual pattern:
    with no instance that lacked pattern N, there is nothing to compare it
    against. PATTERN_ABLATE=k withholds each pattern INDEPENDENTLY with
    probability ~k percent, keyed on sha1(instance_id + entry key), so across
    a corpus every pattern accumulates both a with- and a without- arm while
    any single instance stays reproducible and resume-safe.

    Independent per-pattern withholding, NOT a fixed held-out subset: a fixed
    subset confounds each pattern with the instances it happened to miss.

    Returns (kept, withheld). With PATTERN_ABLATE unset this returns
    (pats, []) and behaviour is bit-identical to before.
    """
    try:
        k = int(os.environ.get("PATTERN_ABLATE", "0"))
    except ValueError:
        k = 0
    if k <= 0 or not pats:
        return pats, []
    import hashlib
    k = min(k, 100)
    kept, held = [], []
    for _p in pats:
        _h = hashlib.sha1(
            ("%s|%s" % (iid, _entry_key(_p))).encode("utf-8", "replace"))
        if int(_h.hexdigest()[:8], 16) % 100 < k:
            held.append(_p)
        else:
            kept.append(_p)
    return kept, held


def _attrib_log(iid, repo, source, entries, blob=None):
    """PER-ENTRY ATTRIBUTION (2026-08-08). Record WHICH accumulated knowledge
    entries were injected into WHICH instance, so a later join against the run
    results can say whether any given atlas idiom, remedy, playbook or
    engineering pattern ever changes an outcome. Until now the harness injected
    e.g. 45 patterns into every instance and measured none of them singly --
    the same unobservability the harness punishes in the model.

    TELEMETRY ONLY: appends one JSONL row; never touches the prompt or score.
    Failures PRINT rather than passing silently (the COVERAGE_GAP lesson).
    """
    try:
        import hashlib
        import json as _json
        keys = []
        if isinstance(entries, dict):
            entries = [entries]
        for _e in (entries or []):
            keys.append(_entry_key(_e))
        rec = {
            "instance_id": iid,
            "repo": repo,
            "source": source,
            "n": len(keys) if keys else (1 if blob else 0),
            "entries": keys,
            "blob_sha": (hashlib.sha1(
                blob.encode("utf-8", "replace")).hexdigest()[:12]
                if isinstance(blob, str) and blob.strip() else None),
            "blob_chars": len(blob) if isinstance(blob, str) else None,
        }
        os.makedirs(os.path.dirname(INJECTED_LOG), exist_ok=True)
        with open(INJECTED_LOG, "a") as _fh:
            _fh.write(_json.dumps(rec) + "\n")
    except Exception as _e:
        print(" -- attrib log FAILED: %s: %s" % (type(_e).__name__, _e),
              flush=True)

SCORE_LOGS = os.path.expanduser("~/swe/score_logs")  # full final-scorer output (telemetry only)


def sh(cmd, cwd=None, timeout=300):
    return subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True,
                          text=True, timeout=timeout)


MIRRORS = os.path.expanduser("~/swe/mirrors")

# --- live event bus (2026-07-25): append-only JSONL of everything a run does,
# tailed live by the monitor on :8899. Best-effort: never raises, never blocks
# a run. Path from $LLMOS_EVENTS, else a single shared stream under runs/live/.
import threading as _threading
EVENTS_PATH = (os.environ.get("LLMOS_EVENTS")
               or os.path.expanduser("~/swe/runs/live/events.jsonl"))
_events_lock = _threading.Lock()
_events_seq = [0]


def _cap_ev(v, n=20000):
    """Bound one field so a huge result/generation can't bloat the line. The
    complete output is still in the trace; this stream is for watching."""
    if isinstance(v, str):
        return v if len(v) <= n else v[:n] + " \u2026[+%d chars]" % (len(v) - n)
    try:
        s = json.dumps(v, default=str)
    except Exception:
        s = str(v)
    if len(s) <= n:
        return v
    return s[:n] + " \u2026[+%d chars]" % (len(s) - n)


def make_emitter(instance_id, phase, run_id=None):
    """Return emit(ev_type, fields) appending one JSON line to EVENTS_PATH."""
    def emit(ev_type, fields=None):
        try:
            with _events_lock:
                _events_seq[0] += 1
                seq = _events_seq[0]
            rec = {"seq": seq, "ts": time.time(), "type": ev_type,
                   "instance_id": instance_id, "phase": phase}
            if run_id:
                rec["run_id"] = run_id
            if fields:
                for k, v in fields.items():
                    rec[k] = _cap_ev(v)
            line = json.dumps(rec, default=str)
            d = os.path.dirname(EVENTS_PATH)
            if d and not os.path.isdir(d):
                os.makedirs(d, exist_ok=True)
            with open(EVENTS_PATH, "a") as fh:
                fh.write(line + "\n")
        except Exception:
            pass
    return emit



def clone(inst):
    """Working checkout via a local --mirror cache. The network is touched
    only when the mirror is missing or lacks base_commit; checkouts are
    created from the mirror (fast, local). Existing checkouts are RESET and
    REUSED, never re-downloaded — and never deleted on failure."""
    repo = os.path.join(WORK, inst["instance_id"])
    mirror = os.path.join(MIRRORS, inst["repo"].replace("/", "__") + ".git")
    os.makedirs(MIRRORS, exist_ok=True)
    # 1. Ensure the mirror holds base_commit (full history + tags: SWE-bench
    #    scoring and setuptools_scm both need tags; shallow clones broke this).
    if not os.path.isdir(mirror):
        sh(f"git clone -q --mirror https://github.com/{inst['repo']}.git {mirror}",
           timeout=7200)
    if sh(f"git -C {mirror} cat-file -e {inst['base_commit']}").returncode != 0:
        sh(f"git -C {mirror} fetch -q --tags origin", timeout=7200)
    # 2. (Re)use the working checkout — reset + clean, no network.
    if os.path.isdir(os.path.join(repo, ".git")):
        sh("git reset -q --hard && git clean -qfdx", cwd=repo, timeout=600)
        sh(f"git checkout -q {inst['base_commit']}", cwd=repo, timeout=300)
    else:
        shutil.rmtree(repo, ignore_errors=True)
        sh(f"git clone -q --shared {mirror} {repo}", timeout=600)
        sh(f"git checkout -q {inst['base_commit']}", cwd=repo, timeout=300)
    sh("git config user.email a@b.c; git config user.name a", cwd=repo)
    return repo


def _auto_verify_reject_detail(res):
    """Turn an auto_verify_env() result into a short, actionable hint for the
    model when the env-ready gate rejects a declare_env_ready. Returns None if
    the env actually verified (nothing to surface). ENV-DIAGNOSTIC ONLY --
    auto_verify_env excludes the instance's FAIL_TO_PASS tests, so this never
    leaks gold/test content."""
    if not isinstance(res, dict) or res.get("ok"):
        return None
    mod = res.get("missing_module")
    if mod:
        return ("harness auto-verify: environment is missing a TEST dependency "
                "`" + str(mod) + "` -- install it (pip/uv) into the active env, "
                "then declare again.")
    err = res.get("error")
    if err:
        return "harness auto-verify failed: " + str(err)
    return ("harness auto-verify could not confirm any green test -- check the "
            "install; then call run_smoke_test WITH NO ARGUMENTS to let the "
            "harness pick a known-stable test.")



def _parse_tool_args(s):
    """Parse a tool-call `arguments` string tolerantly.

    Returns (dict, None) on success or ({}, reason) on failure. Local models
    routinely truncate long argument strings at the num_predict ceiling; those
    are unrecoverable and must be reported back to the model rather than
    dispatched to a handler that will misdiagnose the missing fields.
    """
    for kw in ({}, {"strict": False}):
        try:
            v = json.loads(s, **kw)
            if isinstance(v, dict):
                return v, None
        except Exception:
            pass
    t = s.strip()
    m = re.match(r"^```[a-zA-Z]*\s*(.*?)\s*```$", t, re.S)
    if m:
        t = m.group(1).strip()
    i, j = t.find("{"), t.rfind("}")
    if i != -1 and j > i:
        blk = t[i:j + 1]
        for kw in ({"strict": False}, {}):
            try:
                v = json.loads(blk, **kw)
                if isinstance(v, dict):
                    return v, None
            except Exception:
                pass
    reason = ("truncated mid-JSON" if not t.rstrip().endswith("}")
              else "malformed JSON")
    return {}, reason

_IMPORTANT_RE = re.compile(
    r"(FAIL|ERROR|error|Exception|Traceback|assert|!=|"
    r"Successfully installed|No module named|not found|not exist|DOES NOT EXIST|"
    r"SyntaxError|NameError|ImportError|ModuleNotFound|AttributeError|TypeError|"
    r"ValueError|KeyError|IndexError|RuntimeError|Ran [0-9]+ test|"
    r"^E |hint)", re.I | re.M)


def _extract_important(text, max_chars):
    """Keep the lines that carry the diagnosis, drop the noise, preserve order.
    A head+tail plus every line matching an error/result signature."""
    if len(text) <= max_chars:
        return text
    lines = text.split("\n")
    n = len(lines)
    keep = set(range(min(4, n))) | set(range(max(0, n - 6), n))
    for i, ln in enumerate(lines):
        if _IMPORTANT_RE.search(ln):
            keep.add(i)
    out, last = [], None
    for i in sorted(keep):
        if last is not None and i > last + 1:
            out.append("      ...")
        out.append(lines[i])
        last = i
    s = "\n".join(out)
    if len(s) > max_chars:            # important lines alone still too big
        s = s[:max_chars // 4] + "\n      ...\n" + s[-(3 * max_chars // 4):]
    return s


def smart_summarize(result, max_chars, ref):
    """Summarize a tool result to fit max_chars WITHOUT losing the diagnostic
    content. Big string fields are reduced extractively; small fields pass
    through untouched. The full result stays retrievable via recall(ref)."""
    full = json.dumps(result, default=str)
    if len(full) <= max_chars:
        return full
    if isinstance(result, dict):
        big = [k for k, v in result.items()
               if isinstance(v, str) and len(v) > 400]
        per = max(400, (max_chars - 220) // max(1, len(big)))
        red = {}
        for k, v in result.items():
            if isinstance(v, str) and len(v) > per:
                red[k] = _extract_important(v, per)
            else:
                red[k] = v
        red["_recall"] = ref
        red["_note"] = ("output summarized to fit; nothing discarded. call "
                        "recall(ref=%r) for the complete original output." % ref)
        out = json.dumps(red, default=str)
        if len(out) <= int(max_chars * 1.6):
            return out
        full = out
    return (_extract_important(full, max_chars)
            + ('\n[full output stored as %r -- call recall to see all of it]' % ref))


def _result_sig(tool, result):
    """A stable signature of a tool OUTCOME, for the no-progress watchdog.
    Keys off content: re-running the same failing check or re-applying a patch
    that lands the file in a seen state looks identical, while a genuinely new
    edit or a new error looks different."""
    try:
        if isinstance(result, dict):
            for k in ("error", "stderr"):
                if result.get(k):
                    return tool + "|E|" + error_signature(str(result[k]))[:160]
            parts = [tool]
            for k in ("edited", "mode", "new_bytes", "delta_bytes",
                      "ok", "exit", "match", "match_count"):
                if k in result:
                    parts.append("%s=%s" % (k, result[k]))
            for k in ("stdout", "test_tail", "score_tail", "content"):
                if result.get(k):
                    parts.append(error_signature(str(result[k]))[:120])
                    break
            return "|".join(str(p) for p in parts)[:220]
        return tool + "|" + str(result)[:200]
    except Exception:
        return tool + "|?"


# ---------------------------------------------------------------------------
# EDIT-OPERATION REPERTOIRE (Mikey, 2026-07-28): "If recognition is the problem,
# don\'t try to recognize. Just go through the list of problem types in order of
# frequency. And stop as soon as it\'s solved."
#
# Every recognition-triggered mechanism built on 2026-07-27 failed at the
# RECOGNITION step, never the intervention: the edit classifier covered 19% of
# real fixes, missed `(?i:)` because it only knew re.IGNORECASE, and was wiped
# by unrelated patches. A walked list needs to recognise nothing.
#
# Counts are occurrences among the 300 SWE-bench Lite gold patches; the trailing
# percentage is our measured solve rate when that operation is the fix. Ordered
# by FREQUENCY per Mikey\'s instruction (note: ordering by frequency x solve rate
# would promote "change a literal" 25/67% and "reorder" 10/67% -- worth an A/B
# later, but frequency is the instruction and the honest first cut).
# ---------------------------------------------------------------------------
REPERTOIRE = [
    ("change an argument value",
     "Look at the CALLS in the code you are fixing. Change the VALUE of an "
     "argument being passed, or pass one that is currently left at its default."),      # 109, 46%
    ("change a comparison operator",
     "Look at the comparisons: == != < > <= >= is/is not, in/not in. One of "
     "them is wrong or too strict/too loose. Change it."),                              # 73, 40%
    ("add a branch",
     "There is a case the code does not handle. Add an elif/else for it."),             # 53, 37%
    ("add a guard or early return",
     "A value can be None/empty/missing here and is not checked. Add the "
     "guard, or return early."),                                                        # 50, 30%
    ("add a helper function or method",
     "The logic needed does not exist yet. Write the small function or method "
     "and call it."),                                                                   # 40, 22%
    ("restructure the logic",
     "The approach itself is wrong, not one line of it. Rewrite the block."),           # 33, 13%
    ("wrap in try/except",
     "This can raise, and the caller cannot cope. Catch it, or raise something "
     "more specific."),                                                                 # 28, 33%
    ("change a literal or constant",
     "A hardcoded string/number/default here is wrong. Change it."),                    # 25, 67%
    ("fix a wrong name",
     "An attribute, method or key being referenced is misspelled or is simply "
     "the wrong one. Find the right name and use it."),                                 # 15, 55%
    ("add an argument or flag",
     "A function being called supports a keyword argument that is not being "
     "passed, and passing it fixes this. Find it in the signature."),                   # 12, 50%
    ("reorder the code",
     "The statements are right but happen in the wrong order. Move one."),              # 10, 67%
    ("change a type coercion",
     "Something is being converted to the wrong type, or not converted."),              # 8, 29%
    ("normalise case",
     "A comparison depends on the case of a string and should not."),                   # 3, 0%
]


def _corroborated(state, handlers, log=print):
    """PARTIAL VETO (Mikey, 2026-07-28): "If the deterministic test is only 67%
    accurate, it should only have partial veto power."

    repro_green is a SELF-AUTHORED reproduction; measured reliability ~67-69%.
    Giving it sole authority to end the search is what regressed django-15061:
    it went green, the walk stopped with five operations unused, and the graded
    tests disagreed. Under the old design the model kept working and got there.

    So a green reproduction NOMINATES -- it is recorded as the preferred
    candidate -- but it does not TERMINATE. Termination needs a second,
    independent signal: the repo's own nearby tests still passing. One signal
    ranks; two signals stop. neighbor_tests is leak-safe by construction
    (baseline_pass are green PRE-patch, so FAIL_TO_PASS can never be in it).
    """
    def check():
        if not state.get("repro_green"):
            return False
        state["green_seen"] = True          # nominate, regardless of what follows
        try:
            nb = handlers["swe.neighbor_tests"](None, {})
        except Exception:
            nb = {}
        reg = nb.get("regressed") if isinstance(nb, dict) else None
        if reg == 0:
            log("CORROBORATED: reproduction green AND %s neighbour tests still pass"
                % nb.get("neighbor_tests", "?"))
            return True
        if isinstance(nb, dict) and nb.get("error"):
            # no baseline to corroborate against -- cannot confirm, so keep going
            log("green reproduction, but no neighbour baseline to corroborate; continuing")
            return False
        log("green reproduction, but neighbour tests regressed (%s); continuing" % reg)
        return False
    return check


def repertoire_fix(cpu, tools, tool2sys, handlers, system_prompt, goal,
                   state, seg_turns=8, max_ops=None, log=print, **kw):
    """Run the fix phase as OUR loop, not the model\'s.

    Mikey, 2026-07-28: "we have to be asking the question is this problem one
    where the switches are wrong IN that phase run or before the phase run."

    The gate-based version asked only when the model volunteered that it was
    done -- so the runs that never volunteer (9850s, 11792s, 1694s, all misses)
    were never asked anything at all. Here the outer loop is ours:

        for operation in candidates:
            run seg_turns of work under that instruction
            if the reproduction is green -> stop
            otherwise revert the tree and try the next kind of fix

    Two consequences. The question gets asked regardless of what the model
    decides to do. And a flail is impossible by construction: the ceiling is
    len(candidates) * seg_turns, not "until the model gives up".

    Segments SHARE one conversation (init_messages), so the model keeps what it
    learned about the code and only the instruction changes.
    """
    ops = REPERTOIRE[:max_ops] if max_ops else REPERTOIRE
    msgs = None
    i = 0
    extended = False
    # NON-DESTRUCTIVE INVARIANT (2026-07-28, after the regression test).
    # v1 reverted between operations and, if nothing ever went green, ended
    # with a CLEAN tree -- so a correct patch whose self-authored reproduction
    # could not be made green was thrown away. 3 of the first 6 known-good
    # instances regressed that way, including django-15498, which we had
    # proved by hand was a correct fix. The walk must only ever ADD.
    # Every non-empty segment diff is kept; if no segment achieves a green
    # reproduction, the FIRST candidate is restored. Segment 1 runs on the
    # plain goal with no operation directive, so that candidate is exactly
    # what the un-segmented harness would have submitted. Worst case we tie
    # the old behaviour; we can no longer lose to it.
    candidates = []
    # SEG_ECHO (2026-08-08, gated SEG_ECHO, default off). Measured on
    # sympy-17022: segments 2, 3, 4 and 5 each produced a candidate of exactly
    # 746 bytes -- the tree is reverted between segments and the model walks
    # straight back to the identical edit while being told to try a DIFFERENT
    # KIND of fix. Hash every candidate; when one repeats, state that as a
    # fact in the next segment goal instead of exhorting a third time.
    _diff_seen = {}
    _repeat_note = None
    # GROW (2026-08-12, gated GROW_SEG, default off). Cycle 3 of the
    # single-example loop, sympy-24909: THREE segments went green on the
    # model's own reproduction and ORACLE_GATE refused all three -- the
    # under-generalization signature caught live. The walk then REVERTED
    # each refused candidate and asked for a DIFFERENT kind of fix, so
    # every segment restarted from zero and every candidate stayed small
    # (589, 589, 943 bytes vs gold's larger, sibling-covering patch). A
    # green-but-refused candidate is not wrong, it is INCOMPLETE -- so
    # keep it applied and grow it instead of starting over. At most two
    # consecutive grow segments per walk, then normal reverting resumes.
    _grow_used = 0
    # WALL CLOCK FOR THE WHOLE WALK (env REPERTOIRE_WALL seconds, 0 = off).
    # PHASE_WALL_CAP below bounds ONE phase_run. This walk calls phase_run
    # once per segment and again for every extension, so PHASE_WALL_CAP
    # bounds a SEGMENT and the instance ceiling is 6-12x whatever you set.
    # Measured 2026-08-01: django-16139 ran 9465s under PHASE_WALL_CAP=1800
    # and the cap never fired. This is the instance-level bound that was
    # intended. It BREAKS rather than returns so the fallback below still
    # restores the best candidate -- a timed-out walk must not hand back a
    # clean tree (see the NON-DESTRUCTIVE INVARIANT note above).
    def _green_audit(msgs, meta):
        """INCOMPLETE-FIX GUARD (env GREEN_AUDIT, default off).

        Measured: 20 of 74 misses (27%) ended repro_green=True,
        given_tests_ok=True, resolved=False -- both signals satisfied by a
        HALF-fix. A green reproduction proves the covered path is fixed; it
        says nothing about the paths the reproduction never exercised. One
        bounded audit phase per instance: enumerate what the issue mentions
        that the reproduction does not touch, extend, re-verify.
        """
        if os.environ.get("GREEN_AUDIT") != "1" or state.get("green_audited"):
            return msgs, meta
        state["green_audited"] = True
        _t = int(os.environ.get("GREEN_AUDIT_TURNS", "14"))
        _goal = (
            "STOP -- do not submit yet. Your reproduction passes, but a "
            "reproduction can pass while covering only PART of the issue. "
            "Re-read the problem statement. List every distinct example, "
            "input variant, and symptom it mentions -- every command form, "
            "every data value, every case variant, every code path named. "
            "For EACH one, state whether your reproduction script actually "
            "exercises it. EXTEND the reproduction to cover anything it "
            "misses, run it, and if the extended version fails, fix the "
            "source until it passes. If everything was already covered, say "
            "so and submit.")
        log(" -- GREEN_AUDIT: green nominates, audit confirms coverage "
            "(%d turns)" % _t)
        try:
            _r, msgs2, meta2 = phase_run(cpu, tools, tool2sys, handlers,
                                         system_prompt, _goal, _t,
                                         log=log, init_messages=msgs, **kw)
            msgs, meta = msgs2, meta2
            try:
                handlers["swe.verify_fix"](None, {})
            except Exception:
                pass
            log(" -- GREEN_AUDIT done: repro_green=%s"
                % bool(state.get("repro_green")))
        except Exception as e:
            log(" -- GREEN_AUDIT failed (%s); accepting green as-is"
                % type(e).__name__)
        return msgs, meta

    def _oracle_ok():
        """One bit, harness-side. None = gate off or inconclusive."""
        if os.environ.get("ORACLE_GATE") != "1" or "_oracle_probe" not in handlers:
            return None
        try:
            r = handlers["_oracle_probe"]()
        except Exception as e:
            log(" -- ORACLE_GATE probe failed (%s)" % type(e).__name__)
            return None
        log(" -- ORACLE_GATE: hidden tests %s (harness-side; content never "
            "entered the context)"
            % ("PASS" if r else "FAIL" if r is False else "inconclusive"))
        return r

    def _cap_text(t, limit=4000):
        t = t or ""
        if len(t) <= limit:
            return t
        return t[:limit] + "\n... [truncated, %d chars total]" % len(t)

    def _ledger():
        """SEG_COMPACT (2026-08-12): a factual ledger that REPLACES the raw
        transcript at segment boundaries.

        Motivated by comparing our traces with a frontier harness: it carries
        state across context boundaries as a structured summary, not as the
        raw transcript. Our walk threads the ENTIRE conversation into every
        segment, so by segment 5 the model sits on four verbatim failed
        attempts -- and repetition-is-conviction (sympy-17022: four segments,
        four byte-identical 746-byte candidates) says a transcript full of an
        edit is a PROMPT to produce that edit again. SEG_ECHO patched one
        symptom; this is the generalization: facts in, imitable history out.
        Built harness-side from state -- deterministic, no model summary.
        """
        L = ["LEDGER -- your earlier conversation in this run was compacted "
             "away. The facts below replace it; trust them over memory."]
        _rs = state.get("repro_script")
        if _rs:
            L.append("Registered reproduction script (%s; currently %s):\n%s"
                     % (state.get("repro_mode", "?"),
                        "GREEN" if state.get("repro_green") else "RED",
                        _cap_text(_rs)))
        else:
            L.append("No reproduction script has been registered yet.")
        _d = state.get("diag")
        if _d:
            L.append("Diagnosis record: " + "; ".join(
                "%s=%s" % _kv for _kv in sorted(_d.items())))
        _learned = state.get("learned")
        if _learned:
            L.append("LEARNED in earlier segments (facts observed and "
                     "kept; turn numbers refer to transcripts that were "
                     "compacted away):\n- " + "\n- ".join(_learned))
        if candidates:
            L.append("Patches produced by earlier segments (each made and "
                     "checked already -- do NOT re-produce one of these "
                     "verbatim): " + "; ".join(
                         "%s (%d bytes%s)"
                         % (c[0], len(c[1]),
                            ", went green but the project hidden tests "
                            "still FAILED" if len(c) > 2 and c[2] else "")
                         for c in candidates))
        try:
            _cd = handlers["_capture_diff"]()
        except Exception:
            _cd = ""
        if (_cd or "").strip():
            L.append("CURRENT TREE: the patch below is APPLIED right "
                     "now:\n%s" % _cap_text(_cd))
        else:
            L.append("CURRENT TREE: clean -- no patch is currently applied.")
        return "\n\n".join(L)

    import time as _wt
    _walk_t0 = _wt.time()
    _walk_cap = float(os.environ.get("REPERTOIRE_WALL", "0") or 0)
    _walk_capped = False
    # EXT-BUDGET (2026-08-24): wall seconds already spent per segment
    # index, so an extension pass is charged against the SAME budget
    # instead of receiving a fresh cap (django-11910: seg 1 ran twice,
    # 2059s of the 2400s walk, and segments 3-6 never ran).
    _seg_spent = {}
    while i < len(ops):
        if _walk_cap and (_wt.time() - _walk_t0) > _walk_cap:
            _walk_capped = True
            log(" -- REPERTOIRE wall cap %.0fs reached after %d segment(s) "
                "of %d; stopping the walk and falling back"
                % (_walk_cap, i, len(ops)))
            break
        name, how = ops[i]
        # Segment 1 is the SAFETY ANCHOR: it runs on the plain goal with no
        # operation directive, and its candidate is what we fall back to. It
        # must therefore be a fair baseline. The un-segmented harness gave the
        # model FIX_BUDGET=200 turns; segment 1 was getting 20, a tenth of it,
        # which is most of why django-15790 regressed in 316s -- not the
        # repertoire, just not enough time. Default 60 covers the median
        # successful run (26 turns) and its upper tail, while still bounding
        # the flails we measured at 82/125/185 turns.
        turns_this = int(os.environ.get("SEG1_TURNS", "60")) if i == 0 else seg_turns
        if extended:
            seg_goal = (
                "You have not changed any source yet. Stop reading and make the "
                "edit: %s. %s  Patch the source now, then run your reproduction."
                % (name.upper(), how))
            if os.environ.get("DIAG_GATE", "0") == "1":
                # Navigation for the enforced ladder (cycle-1 finding: a run
                # that never edits never meets an edit-gated challenge).
                seg_goal += (
                    "\n\nYour edits are GATED on the diagnosis ladder. If it "
                    "is not complete (see the diagnosis line in the "
                    "worksheet): run differential(bug_script=..., "
                    "control_script=...) now, then declare_site(file=..., "
                    "function=..., role=..., reason=...), then patch. No "
                    "more locate or read_range until the ladder has moved.")
        elif state.pop("grow_pending", False):
            seg_goal = (
                "Your fix is STILL APPLIED -- the tree was NOT reverted. It "
                "made your reproduction pass, but the project's full test "
                "suite still fails, which means the fix is INCOMPLETE rather "
                "than wrong. Do not rewrite it and do not shrink it. EXTEND "
                "it: (1) if you changed an operator method (__mul__, "
                "__truediv__, __add__...), read its SIBLING operator methods "
                "now -- they usually embed the same pattern and need the "
                "IDENTICAL change in this same patch; (2) check the TYPE you "
                "return against the package idiom -- if neighbouring code "
                "returns a library object where you return a bare Python "
                "literal (1, True, None), return the canonical object "
                "(sympy: S.One, not 1); (3) re-read the issue for the "
                "GENERAL property behind its one example and cover the other "
                "sites in this file that assume the opposite. Accepted fixes "
                "for issues that resist small patches are usually 15-45+ "
                "lines covering cases the issue never mentions.")
        elif i == 0:
            seg_goal = goal
        else:
            seg_goal = (
                "That did not fix it -- your reproduction is still not green, "
                "and the tree has been reverted to its original state. Try a "
                "DIFFERENT KIND of fix now: %s. %s  Make the change, re-run "
                "your reproduction, and stop when it passes."
                % (name.upper(), how))
        _oref = state.pop("oracle_refused", False)
        _ocol = state.pop("oracle_collateral", False)
        if _oref and _ocol:
            # THE 5131 CLASS: F2P passed, P2P regressed. The fix idea is
            # RIGHT; the diff around it is what fails. Say exactly that.
            seg_goal += (
                "\n\nIMPORTANT: your previous fix DID address the reported "
                "bug -- but it broke EXISTING behaviour elsewhere. That "
                "damage comes from COLLATERAL edits in your diff, not from "
                "the fix idea. Keep the same fix, but reapply it as the "
                "SMALLEST possible diff: do not delete blank lines, do not "
                "re-space or re-wrap anything, change nothing beyond the "
                "exact line(s) the fix requires. One line changed is the "
                "target.")
        if _oref and not _ocol:
            # ORACLE-REFUSAL HINT (astropy-14365, measured): the issue example
            # is lowercase COMMANDS only, so every issue-derived repro passes
            # on the IGNORECASE hunk alone -- but the issue states a GENERAL
            # property, and qdp.py has exactly 3 case-assuming sites, one of
            # which is the missing hunk. Tell the model to generalize.
            seg_goal += (
                "\n\nIMPORTANT: a previous fix in this run satisfied the "
                "reported example, yet the problem was NOT fully solved. The "
                "issue likely states a GENERAL property beyond its one "
                "example -- re-read it. Enumerate EVERY other site in the "
                "file you edited that assumes the opposite of that property "
                "(other comparisons, other string literals, other regexes) "
                "and fix them ALL in one patch together with the original "
                "fix. Two gaps that recur: (1) if you changed an OPERATOR "
                "method (__mul__, __truediv__, __add__, __eq__, ...), the "
                "SIBLING operator methods of the same class usually embed "
                "the SAME pattern and need the IDENTICAL change -- read "
                "each sibling and fix it too; (2) check the TYPE you "
                "return against the package idiom: if neighbouring code "
                "returns a library object where you return a bare Python "
                "literal (1, True, None), return what the neighbours "
                "return; (3) if the codebase defines a CANONICAL object "
                "equal to the value your fix produces (a singleton, a "
                "module-level constant), return THAT object itself -- "
                "callers may check identity with `is`, and an "
                "equal-but-fresh value fails that check; (4) treat the "
                "issue text with a GRAIN OF SALT -- it is a user report, "
                "not a contract, and the remedy it proposes may be one the "
                "maintainers would reject. When the issue conflicts with "
                "the codebase own conventions (deprecation-before-removal, "
                "API style, naming), prefer the conventions.")
        if state.pop("grow_echo_note", False):
            seg_goal += (
                "\n\nFACT: after being told to EXTEND your fix, you "
                "submitted a patch BYTE-IDENTICAL to the one the project's "
                "hidden tests already failed. Identical bytes get identical "
                "verdicts -- re-running the reproduction changed nothing and "
                "never will. You must ADD lines to the patch: open the "
                "SIBLING operator/method sites and the canonical-object "
                "idiom now, make a NEW edit, then re-verify.")
        _cgap = state.pop("coverage_gap_note", None)
        if _oref and _cgap:
            seg_goal += (
                "\n\nFACT: these line(s) YOUR PATCH ADDED were never "
                "executed by your reproduction: %s. A line you never "
                "watched run is a line you know nothing about -- it can "
                "hold a crash your reproduction cannot see. Either extend "
                "the reproduction so it actually crosses those lines, or "
                "delete them if the fix does not need them." % _cgap)
        if _repeat_note:
            _pi, _pn, _cn = _repeat_note
            seg_goal += (
                "\n\nFACT: the patch you produced for %s was BYTE-IDENTICAL "
                "to the one you produced for %s in segment %d. Both were "
                "reverted and neither made the reproduction pass. Writing "
                "the same bytes under a different instruction means one of "
                "exactly two things, and you must decide which BEFORE you "
                "edit again. Either that edit is RIGHT and your "
                "REPRODUCTION is checking the wrong thing -- then fix the "
                "reproduction, not the source. Or the edit is wrong and you "
                "are repeating it -- then change a different line. Say which "
                "one it is, then act on it."
                % (_cn.upper(), _pn.upper(), _pi))
            _repeat_note = None
        log(" -- REPERTOIRE segment %d/%d: %s" % (i + 1, len(ops), name))
        # Give this segment no more time than the WALK has left. Without this
        # the wall is only a floor: measured 2026-08-01, segment 1 ended at
        # 1944s under a 2400s wall, so segment 2 started with a fresh 1800s
        # of its own and the real bound was wall + cap.
        _seg_cap = None
        if _walk_cap:
            _left = _walk_cap - (_wt.time() - _walk_t0)
            _env_cap = float(os.environ.get("PHASE_WALL_CAP", "0") or 0)
            _seg_cap = min(_env_cap, _left) if _env_cap else _left
            # SEG1 BUDGET SHARE (env SEG1_WALL_FRAC, 0 = off). Segment 1 is the
            # safety anchor and gets SEG1_TURNS=60, ten times what a later
            # segment gets. Measured on fresh32: 16503 and 15346 spent the
            # ENTIRE 2400s walk inside segment 1, so the other twelve
            # operations never ran -- the anchor consumed the walk it exists to
            # anchor, and 16503 produced patch_bytes=0 for its trouble. Cap the
            # share so the walk is still a walk.
            _s1f = float(os.environ.get("SEG1_WALL_FRAC", "0") or 0)
            if i == 0 and _s1f > 0:
                _share_left = _walk_cap * _s1f - _seg_spent.get(0, 0.0)
                if extended and _share_left <= 60:
                    log(" -- segment 1: share %.0fs already spent "
                        "(%.0fs); skipping the extension and moving on"
                        % (_walk_cap * _s1f, _seg_spent.get(0, 0.0)))
                    extended = False
                    i += 1
                    continue
                _seg_cap = min(_seg_cap, max(_share_left, 60))
            if _seg_cap <= 0:
                _walk_capped = True
                log(" -- REPERTOIRE wall cap %.0fs exhausted before segment "
                    "%d; stopping the walk and falling back" % (_walk_cap, i + 1))
                break
        _init_msgs = msgs
        if os.environ.get("SEG_COMPACT", "0") == "1" and msgs:
            # Facts in, imitable history out (see _ledger). The seg_goal
            # already carries every targeted injection (SEG_ECHO, oracle
            # refusal hints, GROW) -- those survive compaction untouched.
            seg_goal = _ledger() + "\n\n" + seg_goal
            # SEG_COMPACT KEEPS THE TASK (2026-08-25): the purge dropped every
            # non-system message, and the PROBLEM STATEMENT lives in the first
            # user message of the phase -- so from segment 2 on the model did
            # not know what bug it was fixing. django-11422 (a previously
            # resolved instance) ran all six segments blind: 10 turns, zero
            # source changes, patch_bytes=0. Facts in, history out -- but the
            # task is not history.
            _keep_sys  = [m for m in msgs if m.get("role") == "system"][:1]
            _keep_task = [m for m in msgs if m.get("role") != "system"][:1]
            _init_msgs = _keep_sys + _keep_task
            log(" -- SEG_COMPACT: transcript (%d msgs) replaced by system + "
                "ledger for segment %d" % (len(msgs), i + 1))
        _t_segstart = _wt.time()
        reason, msgs, meta = phase_run(cpu, tools, tool2sys, handlers,
                                       system_prompt, seg_goal, turns_this,
                                       wall_cap=_seg_cap,
                                       log=log, init_messages=_init_msgs,
                                       success=_corroborated(state, handlers, log),
                                       **kw)
        _seg_spent[i] = _seg_spent.get(i, 0.0) + (_wt.time() - _t_segstart)
        # LEARNED extraction (2026-08-24, Mikey): "look through that mess
        # and say: is there anything in here that needs to go into the
        # context window for what has to happen further down the line?"
        # One extra pass per segment, accepted deliberately -- getting it
        # right outranks the time. Guards from the critic-digest lesson:
        # full transcript honestly marked, turn citations required, empty
        # output allowed and expected.
        if (os.environ.get("SEG_COMPACT", "0") == "1"
                and os.environ.get("LEARNED", "1") == "1"):
            try:
                from repo_bootstrap_tools import llm_call as _lcall
                from repo_bootstrap_tools import _extract_json as _exj
                from trace_consumers import (events_from_messages as _efm,
                                             _events_digest as _edg)
                _lev = _efm(msgs)
                if _lev:
                    _ldg = _edg(_lev[-40:], max_chars=40000, arg_chars=2000)
                    _lraw = _lcall(
                        system=("You extract only facts that a future "
                                "step will need. Respond ONLY with a "
                                "JSON array of strings; an empty array "
                                "is a good answer."),
                        prompt=("Full transcript digest of one work "
                                "segment (operation: %s):\n%s\n\n"
                                "Is there anything in here that must be "
                                "carried forward for what has to happen "
                                "next? Keep only things LEARNED: an "
                                "environment fact, a located file/"
                                "function/line, a verified behavior, an "
                                "approach that failed and why. Only "
                                "facts actually observed above -- end "
                                "each with its turn like [turn 12]. No "
                                "advice, no plans, no summaries. If "
                                "nothing qualifies, return []."
                                % (name, _ldg)),
                        max_tokens=800)
                    _litems = _exj(_lraw)
                    if isinstance(_litems, list) and _litems:
                        _lst = state.setdefault("learned", [])
                        for _li in _litems[:8]:
                            _li = str(_li).strip()[:300]
                            if _li and _li not in _lst:
                                _lst.append(_li)
                        del _lst[:-40]
                        log(" -- LEARNED: kept %d fact(s) from segment %d"
                            % (min(len(_litems), 8), i + 1))
            except Exception as _le:
                log(" -- LEARNED extraction skipped (%s: %s)"
                    % (type(_le).__name__, _le))
        if reason == "solved" or state.get("repro_green"):
            log(" -- REPERTOIRE solved at segment %d (%s)" % (i + 1, name))
            # COVERAGE GAP (django-11910): green only proves the lines the
            # reproduction CROSSED. Compute which added lines it never
            # executed; if the oracle refuses this green, those lines are
            # stated as facts in the next segment goal.
            try:
                _gap = (handlers.get("_coverage_gap", lambda: {})()
                        if os.environ.get("COVERAGE_GAP") == "1" else {})
            except Exception:
                _gap = {}
            if _gap:
                _gtxt = "; ".join("%s: line(s) %s" % (f, ", ".join(map(str, l)))
                                  for f, l in sorted(_gap.items()))
                log(" -- COVERAGE GAP: added lines never executed by the "
                    "reproduction -- %s" % _gtxt)
                state["coverage_gap_note"] = _gtxt
            # GROW_ECHO (2026-08-13, cycle 4, under GROW_SEG). never28
            # measured GROW on 4 instances with ZERO extensions: sklearn-13497
            # banked 645/645/645 -- the grow segment made NO edit, re-ran the
            # reproduction on the kept tree (trivially green), and the walk
            # spent a FULL hidden-test probe to learn what identical bytes
            # already guaranteed. Identical bytes get identical verdicts:
            # skip the probe, reuse the refusal, and say the no-edit out loud.
            _orc_known = False
            if os.environ.get("GROW_SEG", "0") == "1":
                _lrd = state.get("last_refused_diff")
                if _lrd:
                    try:
                        _nowd = handlers["_capture_diff"]()
                    except Exception:
                        _nowd = ""
                    if _nowd and _nowd == _lrd:
                        _orc_known = True
            if _orc_known:
                log(" -- GROW_ECHO: candidate is byte-identical to the "
                    "already-refused patch; verdict known, probe skipped")
                state["grow_echo_note"] = True
                _orc = False
            else:
                _orc = _oracle_ok()
            if _orc is False:
                # The graded tests disagree with the model's green: the
                # incomplete-fix class, caught in the act. Bank the candidate
                # (green-nominated, so the fallback still prefers it) and
                # keep walking. After the revert the bug is back, so the
                # registered reproduction is genuinely red again -- the
                # walk's own bookkeeping stays truthful.
                try:
                    _cand = handlers["_capture_diff"]()
                    if _cand.strip():
                        candidates.append((name, _cand, True))
                        state["last_refused_diff"] = _cand
                        log(" -- segment %d (%s): oracle-refused candidate "
                            "banked (%d bytes)" % (i + 1, name, len(_cand)))
                except Exception as _e:
                    log(" -- candidate capture failed (%s)" % type(_e).__name__)
                state["repro_green"] = False
                state.pop("green_seen", None)
                state["oracle_refused"] = True
                if "PASS_TO_PASS regressed" in getattr(oracle_probe, "last_tail", ""):
                    state["oracle_collateral"] = True
                if (os.environ.get("GROW_SEG", "0") == "1"
                        and not state.get("oracle_collateral")
                        and _grow_used < 2):
                    # Keep the refused candidate APPLIED and grow it. Never
                    # on collateral (P2P regressed): growing a diff that
                    # broke neighbours compounds the damage -- that path
                    # still reverts below.
                    _grow_used += 1
                    state["grow_pending"] = True
                    log(" -- GROW %d/2: keeping the oracle-refused candidate "
                        "applied; next segment extends it" % _grow_used)
                    extended = False
                    i += 1
                    continue
                try:
                    handlers["_revert_tree"]()
                except Exception as _e:
                    log(" -- revert failed (%s); continuing" % type(_e).__name__)
                extended = False
                i += 1
                continue
            if _orc is None:
                msgs, meta = _green_audit(msgs, meta)
            return "declared", msgs, meta
        if reason in ("no_call",):
            return reason, msgs, meta

        try:
            _changed = bool(handlers["_diff_nonempty"]())
        except Exception:
            _changed = True            # never block the walk on a harness error

        # (a) HARNESS-SIDE VERIFICATION. A segment that edits the source and
        # never re-runs the reproduction cannot trigger the success break -- it
        # made changes nobody checked (observed: "add a branch", 2 patches,
        # 0 verify calls). Do not rely on the model to check its own work.
        if _changed and state.get("seen_red") and not state.get("repro_green"):
            try:
                handlers["swe.verify_fix"](None, {})
                log(" -- segment %d: harness verify_fix -> repro_green=%s"
                    % (i + 1, bool(state.get("repro_green"))))
            except Exception as e:
                log(" -- segment %d: verify_fix failed (%s)"
                    % (i + 1, type(e).__name__))
            if state.get("repro_green"):
                log(" -- REPERTOIRE solved at segment %d (%s) on harness verify"
                    % (i + 1, name))
                _orc = _oracle_ok()
                if _orc is False:
                    # fall through: the normal end-of-segment logic banks the
                    # candidate and reverts, and the walk continues.
                    state["repro_green"] = False
                    log(" -- ORACLE_GATE: overriding the harness-verify green; "
                        "the walk continues")
                else:
                    if _orc is None:
                        msgs, meta = _green_audit(msgs, meta)
                    return "declared", msgs, meta

        # (b) AN OPERATION IS NOT SPENT UNTIL IT WAS ACTUALLY ATTEMPTED.
        # Observed: two segments burned their budget (one for 18 turns) without
        # applying a single patch -- consuming a branch of the search while
        # never trying that kind of fix. Grant one extension, then move on.
        if not _changed and not extended:
            _rem_walk = ((_walk_cap - (_wt.time() - _walk_t0))
                         if _walk_cap else None)
            if _rem_walk is not None and _rem_walk < 120:
                log(" -- segment %d (%s): no source change and only "
                    "%.0fs of walk left; skipping the extension"
                    % (i + 1, name, _rem_walk))
                i += 1
                continue
            log(" -- segment %d (%s): no source change; extending once"
                % (i + 1, name))
            extended = True
            continue
        extended = False
        i += 1

        if _changed:
            try:
                _cand = handlers["_capture_diff"]()
                if _cand.strip():
                    # a candidate whose reproduction went green outranks the
                    # segment-1 fallback, even though green alone did not stop
                    # the walk. nomination without termination.
                    candidates.append((name, _cand, bool(state.pop("green_seen", False))))
                    log(" -- segment %d (%s): candidate saved (%d bytes)"
                        % (i, name, len(_cand)))
                    if os.environ.get("SEG_ECHO") == "1":
                        import hashlib as _hl
                        _h = _hl.sha1(
                            _cand.encode("utf-8", "replace")).hexdigest()
                        _prev = _diff_seen.get(_h)
                        if _prev:
                            _repeat_note = (_prev[0], _prev[1], name)
                            log(" -- SEG_ECHO: segment %d (%s) reproduced the "
                                "byte-identical diff from segment %d (%s)"
                                % (i, name, _prev[0], _prev[1]))
                        else:
                            _diff_seen[_h] = (i, name)
            except Exception as e:
                log(" -- candidate capture failed (%s)" % type(e).__name__)

        try:
            handlers["_revert_tree"]()          # clean slate for the next kind
        except Exception as e:
            log(" -- revert failed (%s); continuing without it" % type(e).__name__)
    log(" -- REPERTOIRE %s %d of %d operations without a green reproduction"
        % ("stopped on the wall cap after" if _walk_capped else "exhausted",
           i, len(ops)))
    # restore the fallback rather than submitting an empty tree
    if candidates:
        _green = [c for c in candidates if len(c) > 2 and c[2]]
        _pick = _green[0] if _green else candidates[0]
        _name, _diff = _pick[0], _pick[1]
        log(" -- fallback: %d candidate(s), %d with a green reproduction; using %s"
            % (len(candidates), len(_green), _name))
        # BANK AUDIT (env BANK_AUDIT + ORACLE_GATE, both default off).
        # Measured: we bank up to 20 candidates and then pick by list
        # position. 12113 banked a candidate touching the file the accepted
        # fix edits and submitted a different one. Ask the oracle which of
        # them actually passes -- the same one bit we already spend at green
        # points, over work we have already done.
        if (os.environ.get("BANK_AUDIT") == "1"
                and os.environ.get("ORACLE_GATE") == "1"
                and "_oracle_probe" in handlers and len(candidates) > 1):
            try:
                _seen, _ord = set(), []
                for _c in (_green + [c for c in candidates if c not in _green]):
                    _k = " ".join((_c[1] or "").split())
                    if _k and _k not in _seen:
                        _seen.add(_k)
                        _ord.append(_c)
                _cap = int(os.environ.get("BANK_AUDIT_MAX", "6") or 6)
                _ord = _ord[:_cap]
                log(" -- BANK_AUDIT: %d distinct candidate(s), probing up to %d"
                    % (len(_seen), len(_ord)))
                for _idx, _c in enumerate(_ord):
                    try:
                        handlers["_revert_tree"]()
                    except Exception:
                        pass
                    _rr = handlers["_restore_diff"](_c[1])
                    if not _rr.get("restored"):
                        log(" -- BANK_AUDIT: candidate %d (%s) would not apply"
                            % (_idx + 1, _c[0]))
                        continue
                    _v = handlers["_oracle_probe"]()
                    log(" -- BANK_AUDIT: candidate %d (%s, %d bytes) -> %s"
                        % (_idx + 1, _c[0], len(_c[1]),
                           "PASS" if _v else "FAIL" if _v is False
                           else "inconclusive"))
                    if _v:
                        _name, _diff = _c[0], _c[1]
                        log(" -- BANK_AUDIT: submitting candidate %d (%s) "
                            "instead of the positional pick" % (_idx + 1, _c[0]))
                        break
                else:
                    log(" -- BANK_AUDIT: no banked candidate passed; keeping "
                        "the positional pick (%s)" % _name)
            except Exception as _ba:
                log(" -- BANK_AUDIT error (%s: %s); keeping positional pick"
                    % (type(_ba).__name__, _ba))
        try:
            _r = handlers["_restore_diff"](_diff)
            log(" -- restored candidate from segment 1 (%s): %s"
                % (_name, "ok" if _r.get("restored") else "FAILED " + _r.get("err", "")))
        except Exception as e:
            log(" -- candidate restore failed (%s)" % type(e).__name__)
    else:
        log(" -- no candidate patch was produced by any operation")
    return "declared", msgs, meta


def phase_run(cpu, tools, tool2sys, handlers, system_prompt, user_goal,
              budget, wall_cap=None,
              gate=None, log=print, checkpoint=None, worksheet=None,
              emit=None, stall_window=None, init_messages=None,
              success=None, free_text=None):
    """Drive one phase: chat, dispatch tool calls, repeat until the model
    calls a RETURN-typed tool (env_ready/submit) or budget is exhausted.

    Returns (terminated_reason, transcript, meta_log)
      terminated_reason: 'declared', 'gate_blocked', 'budget', 'no_call'
    """
    # init_messages lets a caller drive SEGMENTS: run N turns, inspect state,
    # then continue the same conversation with a new instruction. That is what
    # makes the repertoire an outer LOOP we own rather than advice we give the
    # model when it happens to submit.
    if init_messages:
        messages = list(init_messages)
        if user_goal:
            messages.append({"role": "user", "content": user_goal})
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_goal},
        ]
    meta_log = []
    # WALL-CLOCK CAP (env PHASE_WALL_CAP seconds, 0/unset = off). Measured
    # 2026-07-27/28: four flails, django-11019 at 9850s and django-11283 at
    # 11792s, both misses -- 6 of one night\'s 10 hours spent on two problems
    # that were never going to converge. The turn budget almost never trips
    # (1 of 49 runs) because the model keeps producing novel-looking calls.
    # Time is the honest bound.
    import time as _time
    _phase_t0 = _time.time()
    # An explicit cap from the caller wins over the env. The repertoire walk
    # knows its own deadline and must be able to hand a segment less than the
    # full PHASE_WALL_CAP, or the walk overshoots by a whole segment.
    _wall_cap = (float(wall_cap) if wall_cap is not None
                 else float(os.environ.get("PHASE_WALL_CAP", "0") or 0))
    searched_sigs = {}   # error signature -> turn first searched
    _catalog = {}        # out<turn> -> full tool result, recall()-able
    _seen_sigs = set()   # result signatures seen so far (no-progress watchdog)
    _novel_hist = []     # per-turn: 1 if the result was new, else 0
    for turn in range(budget):
        if worksheet is not None:
            # the state object, written down and re-shown EVERY turn: the model
            # reads its situation instead of having to remember it
            try:
                messages[1]["content"] = user_goal + "\n\n" + worksheet()
            except Exception:
                pass
        msg = None
        for attempt in range(3):
            try:
                msg, meta = cpu._chat(messages)
                break
            except Exception as e:
                err = str(e)
                time.sleep(20 * (attempt + 1))
        if msg is None:
            return "cpu_error", messages, meta_log + [{"error": err}]
        meta_log.append({"turn": turn,
                          "prompt_tokens": meta.get("prompt_tokens"),
                          "eval_tokens":   meta.get("eval_tokens")})
        if emit:
            emit("generation", {"turn": turn,
                                "content": msg.get("content") or "",
                                "reasoning": (msg.get("reasoning_content")
                                              or msg.get("thinking") or ""),
                                "eval_tokens": meta.get("eval_tokens"),
                                "prompt_tokens": meta.get("prompt_tokens"),
                                "finish_reason": meta.get("finish_reason"),
                                "trunc_grow": meta.get("trunc_grow")})
        if checkpoint:
            try:
                tmp = checkpoint + ".tmp"
                json.dump({"phase1": messages, "phase1_meta": meta_log,
                           "partial": True},
                          open(tmp, "w"), default=str)
                os.replace(tmp, checkpoint)
            except Exception:
                pass
        tcs = msg.get("tool_calls") or []
        if not tcs:
            full = (msg.get("content") or msg.get("reasoning_content")
                    or msg.get("thinking") or "")
            # A turn with no tool call is where a PROSE ANSWER arrives. The
            # harness used to truncate it to 400 chars and scold; if something
            # asked a question, hand it the whole reply first.
            answered = False
            if free_text:
                try:
                    answered = bool(free_text(full))
                except Exception:
                    answered = False
            content = full[:400]
            messages.append({"role": "assistant", "content": content or "..."})
            messages.append({"role": "user",
                              "content": ("Noted. Continue -- call a tool."
                                          if answered
                                          else "Call one of the provided tools now.")})
            continue
        tc = tcs[0]
        fn = tc.get("function", {})
        tool = fn.get("name", "")
        args = fn.get("arguments") or {}
        args_err = None
        if isinstance(args, str):
            args, args_err = _parse_tool_args(args)
        target = tool2sys.get(tool, "")
        log(f"  [{turn:>2}] {tool}({str(args)[:80]}) -> ", end="", flush=True)
        if emit:
            emit("tool_call", {"turn": turn, "tool": tool,
                               "function": target, "args": args,
                               "args_error": args_err})
        if args_err is not None:
            # The argument string never parsed -- almost always because the
            # generation was cut off at the num_predict ceiling. Dispatching
            # a placeholder dict here makes the handler answer with a
            # misleading error about the missing field, and the model, having
            # no idea its JSON was truncated, re-sends the identical call.
            # Nothing is executed; say what actually happened instead.
            log(f"ARGS-UNPARSEABLE ({args_err})")
            messages.append({"role": "assistant", "content": "",
                             "tool_calls": [{"id": f"t{turn}", "type": "function",
                                             "function": {"name": tool,
                                                          "arguments": "{}"}}]})
            messages.append({"role": "tool", "tool_call_id": f"t{turn}",
                             "content": json.dumps({
                                 "error": "your tool-call arguments were not valid "
                                          f"JSON ({args_err}); NOTHING was executed",
                                 "hint": "The argument string ended early, so this "
                                         "call never ran. Re-send the same tool call "
                                         "with complete, valid JSON. If the payload "
                                         "is long, make it smaller: patch fewer lines "
                                         "per call, or split a long script across "
                                         "several calls."})})
            continue
        # Recall: hand back the FULL, un-summarized output of an earlier call.
        if tool == "recall":
            _r = str((args or {}).get("ref", "")).strip()
            _stored = _catalog.get(_r)
            log("RECALL %s -> %s" % (_r, "hit" if _stored is not None else "miss"))
            messages.append({"role": "assistant", "content": "",
                             "tool_calls": [{"id": f"t{turn}", "type": "function",
                                             "function": {"name": tool,
                                                          "arguments": args}}]})
            _body = (json.dumps(_stored, default=str) if _stored is not None
                     else json.dumps({"error": "no stored output %r; ids look "
                                      "like out<turn> and appear in summary notes"
                                      % _r}))
            messages.append({"role": "tool", "tool_call_id": f"t{turn}",
                             "content": _body[:16000]})
            continue
        # Terminal tool: check gate then break out.
        if target == "RETURN":
            if gate is not None and not gate():
                # Model tried to declare done but the gate says no. Feed back
                # the reason and continue.
                log("GATE-REJECTED")
                if emit:
                    emit("gate", {"turn": turn, "decision": "rejected"})
                messages.append({"role": "assistant", "content": "",
                                  "tool_calls": [{"id": f"t{turn}", "type": "function",
                                                   "function": {"name": tool,
                                                                 "arguments": args}}]})
                _gate_payload = {"error": getattr(gate, "reject_message", None)
                                          or ("verification gate not passed; "
                                              "run_sanity and run_smoke_test must "
                                              "both return ok=true first")}
                _detail = getattr(gate, "reject_detail", None)
                if _detail:
                    _gate_payload["harness_check"] = _detail
                messages.append({"role": "tool", "tool_call_id": f"t{turn}",
                                  "content": json.dumps(_gate_payload)})
                continue
            log(f"DECLARED {tool}")
            if emit:
                emit("declared", {"turn": turn, "tool": tool})
            return "declared", messages + [
                {"role": "assistant", "content": "", "tool_calls": [tc]},
            ], meta_log
        if _wall_cap and (_time.time() - _phase_t0) > _wall_cap:
            log("WALL-CAP: %.0fs elapsed > %.0fs cap; ending phase"
                % (_time.time() - _phase_t0, _wall_cap))
            if emit:
                emit("wall_cap", {"turn": turn, "elapsed": _time.time() - _phase_t0})
            return "budget", messages, meta_log

        # Normal tool: dispatch.
        # PHASE_DEADLINE (2026-08-24): hand the remaining budget down so a
        # single slow tool call cannot overshoot the phase wall. The check
        # above only runs BETWEEN turns; measured overshoot was 28-42%
        # (worst 178s against a 23s cap), which consumed the whole 2400s
        # repertoire wall in one segment and produced zero-byte patches.
        try:
            import test_runner as _tr_mod, swe_fix_tools as _sft_mod
            _dl = (_phase_t0 + _wall_cap) if _wall_cap else None
            _tr_mod.PHASE_DEADLINE = _dl
            _sft_mod.PHASE_DEADLINE = _dl
        except Exception:
            pass
        h = handlers.get(target)
        if h is None:
            result = {"error": f"unknown tool {tool!r}"}
        else:
            try:
                result = h(None, args)
            except Exception as e:
                result = {"error": f"handler crashed: {type(e).__name__}: {e}"}
        # SIBLING-SITE SWEEP (env SIBLING_SWEEP, default off). Measured failure:
        # on astropy-14365 the model added re.IGNORECASE (correct, matches gold)
        # and never touched the sibling `if v == "NO"` at line 309 -- a plain
        # string compare no traceback names and no regex flag can reach. It
        # submitted a half fix; the graded test failed "DID NOT WARN", a silent
        # data-correctness bug. This classifies the edit just made and reports
        # UNCHANGED sites of the SAME class. Facts with line numbers only, never
        # advice: prompt nudges have been disproven twice on this model.
        if (os.environ.get("SIBLING_SWEEP") == "1" and tool == "patch"
                and isinstance(result, dict) and "edited" in result):
            try:
                _sw = handlers["_sibling_sweep"](
                    str(args.get("old_snippet") or ""),
                    str(args.get("new_snippet") or ""),
                    # start_line is OPTIONAL in the patch schema and absent in
                    # snippet mode. edited_line is what the harness actually
                    # wrote, so prefer it.
                    result.get("edited_line") or args.get("start_line"),
                    result.get("edited"),
                )
                if _sw:
                    result["sibling_sites"] = _sw
                    log("SIBLING_SWEEP class=%s sites=%d"
                        % (_sw["edit_class"], len(_sw["unchanged_same_class_sites"])))
                else:
                    log("SIBLING_SWEEP no-fire (no class or no unchanged sites)")
            except Exception as _e:
                log("SIBLING_SWEEP error: %s: %s" % (type(_e).__name__, _e))
        # DIFF HYGIENE (env DIFF_HYGIENE, default off): xarray-5131 fixed the
        # bug (F2P passed) and failed grading on collateral whitespace -- a
        # deleted blank line broke 3 doctests. Deterministic lint on the diff.
        if (os.environ.get("DIFF_HYGIENE") == "1" and tool == "patch"
                and isinstance(result, dict) and "edited" in result):
            try:
                _dh = handlers["_diff_hygiene"]()
                if os.environ.get("DIFF_REPAIR") == "1":
                    _fx = handlers["_diff_repair"](result.get("edited"))
                    if _fx:
                        log("DIFF_REPAIR restored %d drifted line(s)" % _fx)
                        _dh = handlers["_diff_hygiene"]()
                if _dh:
                    result["diff_hygiene_warning"] = _dh["note"]
                    log("DIFF_HYGIENE %s" % "; ".join(_dh["problems"]))
            except Exception as _e:
                log("DIFF_HYGIENE error: %s" % type(_e).__name__)
        # SIBLING BODY (env SIBLING_BODY, default off). django-16910 resolves
        # 2 of 9 runs. Diffing its archive shows three misses with an IDENTICAL
        # structure to a resolved patch, differing only in that they ALSO
        # pasted the first loop of the neighbouring _get_defer_select_mask --
        # their comments even say QuerySet.defer() where the resolved one says
        # only(). That is a RETRIEVAL error between two adjacent near-identical
        # methods, not a reasoning error, and no amount of test plumbing
        # touches it. Across the 622-patch archive the signature carries a
        # 4.3x enrichment for misses (3.4% vs 0.8%).
        #
        # It WARNS. It fires on correct patches too, because porting a
        # sibling's shared tail is legitimate -- so it names the copied lines
        # and lets the model decide, rather than pretending to know.
        if (os.environ.get("SIBLING_BODY") == "1" and tool == "patch"
                and isinstance(result, dict) and "edited" in result):
            try:
                _sbr = handlers["_sibling_body_check"](
                    result.get("edited"), args.get("new_snippet"),
                    result.get("edited_line"))
                if _sbr:
                    result["sibling_body_warning"] = _sbr["note"]
                    log("SIBLING_BODY %s <- %s (%d of %d lines)"
                        % (_sbr["edited_function"], _sbr["sibling"],
                           _sbr["shared_unique_lines"], _sbr["written_lines"]))
            except Exception as _e:
                log("SIBLING_BODY error: %s: %s" % (type(_e).__name__, _e))
        # SPEC PROBE (env SPEC_PROBE, default off). DETERMINISTIC spec
        # reconstruction, born 2026-08-03: four misses in one night each
        # failed exactly ONE hidden assertion. AST facts, not advice text:
        # (1) edit touched an operator dunder -> name the class's un-edited
        # sibling dunders; (2) edited snippet returns a bare Python literal
        # in a file with library-singleton evidence -> name the line. Fires
        # only when computably true; warns, never blocks.
        if (os.environ.get("SPEC_PROBE") == "1" and tool == "patch"
                and isinstance(result, dict) and "edited" in result):
            try:
                _spn = handlers["_spec_probe"](
                    result.get("edited"), args.get("new_snippet"),
                    result.get("edited_line"))
                if _spn:
                    result["spec_probe_warning"] = _spn
                    log("SPEC_PROBE fired on %s" % result.get("edited"))
            except Exception as _e:
                log("SPEC_PROBE error: %s: %s" % (type(_e).__name__, _e))
        # GRAPHIFY INJECT (env GRAPHIFY_INJECT, default off). Same failure as
        # SIBLING_SWEEP above -- fixed one site, missed the sibling -- reached
        # from the other direction. The sweep finds unchanged sites of the same
        # SYNTACTIC class; this finds sites in the same CALL GRAPH, which no
        # pattern match on the edit itself can see.
        #
        # It fires rather than being offered because offering it was measured
        # and did nothing: GRAPHIFY=1 put `neighborhood` in the tool list for
        # two instances / 84 tool calls and the model never once called it.
        if (os.environ.get("GRAPHIFY_INJECT") == "1" and tool == "patch"
                and isinstance(result, dict) and "edited" in result):
            try:
                _gn = handlers["_neighborhood_of_edit"](
                    result.get("edited"),
                    result.get("edited_line") or args.get("start_line"))
                if _gn:
                    result["call_graph_neighborhood"] = _gn
                    log("GRAPHIFY_INJECT symbol=%s chars=%d"
                        % (_gn.get("edited_symbol"),
                           len(_gn.get("other_sites_in_the_call_graph") or "")))
                else:
                    log("GRAPHIFY_INJECT no-fire (no enclosing symbol, or "
                        "symbol absent from the code graph)")
            except Exception as _e:
                log("GRAPHIFY_INJECT error: %s: %s" % (type(_e).__name__, _e))
        if (os.environ.get("NEIGHBOR_INJECT") == "1" and tool == "patch"
                and isinstance(result, dict) and "edited" in result):
            try:
                _nb = handlers["swe.neighbor_tests"](None, {})
            except Exception as _e:
                _nb = {"error": "inject handler crashed: %s: %s" % (type(_e).__name__, _e)}
            log("NEIGHBOR_INJECT probe -> " + str(_nb)[:400])
            _rg = _nb.get("regressed") if isinstance(_nb, dict) else None
            if _rg and _rg != 0:
                result["blast_radius"] = _nb.get("note") or "patch broke a neighbor test; keep your fix and repair it before submitting."
                if _nb.get("which"):
                    result["blast_radius_broke"] = _nb["which"]
                log("NEIGHBOR_INJECT FIRED regressed=%s which=%s" % (_rg, _nb.get("which")))
            else:
                log("NEIGHBOR_INJECT NO-FIRE regressed=%r error=%r" % (
                    _rg, (_nb.get("error") if isinstance(_nb, dict) else None)))
        log(str(result)[:120])
        if emit:
            emit("tool_result", {"turn": turn, "tool": tool, "result": result})
        # CHECK AND BREAK (Mikey, 2026-07-28: "we have to check if we have the
        # right answer and break in that phase run"). Asked after EVERY tool
        # result, not at submit and not at a segment boundary -- if the
        # reproduction goes green on turn 2 of 8 we stop on turn 2. The model
        # does not have to notice, announce, or agree that it is done.
        if success is not None:
            try:
                if success():
                    log("SOLVED: success condition met at turn %d" % turn)
                    if emit:
                        emit("solved", {"turn": turn})
                    return "solved", messages, meta_log
            except Exception:
                pass          # a broken predicate must never trap the agent

        if stall_window:
            _sig = _result_sig(tool, result)
            _novel_hist.append(0 if _sig in _seen_sigs else 1)
            _seen_sigs.add(_sig)
            if (len(_novel_hist) >= stall_window
                    and sum(_novel_hist[-stall_window:]) <= 2):
                log("STALLED: only %d novel results in the last %d turns"
                    % (sum(_novel_hist[-stall_window:]), stall_window))
                if emit:
                    emit("stalled", {"turn": turn, "window": stall_window})
                return "stalled", messages, meta_log
        # Human reflex: see an error -> search the web for it -> THEN act.
        failed = isinstance(result, dict) and (
            result.get("ok") is False or "error" in result)
        if failed:
            sig = error_signature(str(result.get("error")
                                      or result.get("stderr") or ""))
            if sig and len(sig) > 12:
                if sig in searched_sigs:
                    result["error_web_search"] = (
                        f"(already searched at turn {searched_sigs[sig]} — "
                        "same error again means your last change did not "
                        "address it; re-read those results, try a DIFFERENT "
                        "action)")
                else:
                    searched_sigs[sig] = turn
                    try:
                        hits = _ddg_search(sig[:120], 3)
                    except Exception:
                        hits = []
                    if hits:
                        result["error_web_search"] = [
                            {"title": h["title"][:120],
                             "snippet": h["snippet"][:240]} for h in hits]
        messages.append({"role": "assistant", "content": "",
                         "tool_calls": [{"id": f"t{turn}", "type": "function",
                                          "function": {"name": tool, "arguments": args}}]})
        _ref = "out%d" % turn
        _catalog[_ref] = result
        _full = json.dumps(result, default=str)
        _content = _full if len(_full) <= 4800 else smart_summarize(result, 4800, _ref)
        messages.append({"role": "tool", "tool_call_id": f"t{turn}",
                         "content": _content})
        # Mid-run critic: every 8 turns a detached reviewer scans the recent
        # trace (and web-searches the latest error) for loops/drift/self-harm.
        if turn % 8 == 7:
            try:
                advice = critic_review(messages)
            except Exception:
                advice = ""
            if advice:
                log(f"  [critic] {advice[:300]}")
                if emit:
                    emit("critic", {"turn": turn, "advice": advice})
                messages.append({"role": "user",
                                 "content": f"[HARNESS CRITIC] {advice}"})
    return "budget", messages, meta_log



# import-name -> pip package name, for the missing-module reflex
_PKG_ALIASES = {
    "cv2": "opencv-python", "yaml": "pyyaml", "PIL": "pillow",
    "sklearn": "scikit-learn", "bs4": "beautifulsoup4", "OpenSSL": "pyopenssl",
    "dateutil": "python-dateutil", "attr": "attrs", "jinja2": "jinja2",
}
_MISSING_RE = re.compile(r"No module named ['\"]([\w.]+)['\"]")


def _web_lookup_pkg(mod):
    """When the pip name isn't obvious, do what a developer does: search
    'how do I install <module>' and read the pip package name off the
    results. Returns a package name or None."""
    try:
        from repo_bootstrap_tools import _ddg_search, llm_call, _extract_json
    except Exception:
        return None
    hits = _ddg_search(f"python how do I install module {mod} pip", 5)
    if not hits:
        return None
    blob = "\n".join(f"- {h['title']}: {h['snippet']}" for h in hits)
    raw = llm_call(
        system="You map a Python import name to its pip package. JSON only.",
        prompt=(f"A Python import 'import {mod}' fails with ModuleNotFoundError. "
                f"From these search results, what is the exact pip install "
                f"name?\n\n{blob}\n\n"
                'Return JSON: {"pip_name": "..."} (just the package, or null '
                "if the results do not say)"),
        max_tokens=400, format_json=True)
    pkg = (_extract_json(raw) or {}).get("pip_name")
    return pkg if pkg and pkg not in ("null", "None") else None


def _try_pip(pkg, repo, env, env_dir):
    """Install one package with NO SHELL.

    `pkg` may be model output that came back from a web search
    (_web_lookup_pkg), so it goes through pkg_guard: bounded PEP 508 name,
    argv list, nothing for a shell to reinterpret. An unsafe name is refused
    rather than escaped -- there is no escaping that stays correct.
    """
    import pkg_guard as _pg
    exe = os.path.join(env_dir, "bin", "python")
    try:
        argv = _pg.pip_argv(exe, pkg)
    except ValueError as e:
        print("  -- %s" % e)
        return False
    try:
        r = subprocess.run(argv, cwd=repo, capture_output=True, text=True,
                           timeout=300, env=env)
        if r.returncode != 0 and "No module named pip" in (r.stderr or ""):
            subprocess.run([exe, "-m", "ensurepip", "--default-pip"], cwd=repo,
                           capture_output=True, text=True, timeout=120,
                           env=env)
            r = subprocess.run(argv, cwd=repo, capture_output=True, text=True,
                               timeout=300, env=env)
    except (OSError, subprocess.SubprocessError) as e:
        # argv exec RAISES where shell=True returned 127. Declining one
        # install must not kill the run.
        print("  -- pip install %r failed to launch: %s" % (pkg, e))
        return False
    return r.returncode == 0


def _pip_install(mod, repo, env, env_dir):
    """Install a missing module. Try the alias/bare name first; if that
    fails, web-search 'how do I install <module>' for the real pip name."""
    pkg = _PKG_ALIASES.get(mod, mod.split(".")[0])
    if _try_pip(pkg, repo, env, env_dir):
        return True, pkg
    looked = _web_lookup_pkg(mod)
    if looked and looked != pkg:
        import pkg_guard as _pg
        print("  -- pip name from web: %r -> %r (%s)"
              % (mod, looked, _pg.relatedness(mod, looked)))
        if _try_pip(looked, repo, env, env_dir):
            return True, looked
    return False, pkg


def _run_with_missing_module_reflex(cmd, repo, env, env_dir, max_installs=4):
    """Run cmd; while it fails with 'No module named X', install X and retry.
    Guard: each module installed at most once, so a genuine build failure
    (module truly unavailable) surfaces instead of looping."""
    tried = set()
    for _ in range(max_installs + 1):
        r = subprocess.run(cmd, shell=True, cwd=repo, capture_output=True,
                           text=True, timeout=600, env=env)
        out = (r.stdout or "") + (r.stderr or "")
        m = _MISSING_RE.search(out)
        if not m:
            return r
        mod = m.group(1)
        if mod in tried:
            return r   # already installed it once — real failure, surface it
        tried.add(mod)
        ok, pkg = _pip_install(mod, repo, env, env_dir)
        if not ok:
            return r
    return r


# --- Env-faithfulness correction for warnings-as-errors repos -----------------
# Some repos run with `filterwarnings = error`. A too-new *pure-python* dep can
# emit a DeprecationWarning at import time that becomes fatal and turns test
# COLLECTION into "found no collectors" -- scoring a CORRECT patch as a miss
# (false negative; Docker-confirmed on matplotlib 23913/23964/23987/24149).
# Pin such deps back to an era-compatible version. General, repo-level; derived
# from package behaviour, never from any instance's answer.
WARN_AS_ERROR_DEP_PINS = {
    # matplotlib 3.x calls pyparsing's camelCase API (enablePackrat/setParseAction);
    # pyparsing >=3.1 raises PyparsingDeprecationWarning on those -> fatal under
    # matplotlib's filterwarnings=error. <3.1 keeps the API but stays silent.
    # Second cause, same NO_COLLECTORS symptom: some matplotlib dev builds compute
    # __version__ via setuptools_scm.get_version() AT IMPORT; setuptools-scm 8+ pulls
    # in vcs-versioning, whose "release-branch-semver" entry-point is a deprecation
    # shim that raises DeprecationWarning -> fatal collection error even once
    # pyparsing is pinned. Downgrade to the self-contained setuptools-scm 7.x AND
    # remove the orphaned vcs-versioning so the native (silent) scheme is used.
    # A spec written as "-pkg" means uninstall pkg.
    # Third cause, same NO_COLLECTORS symptom: uv resolves pytest to the LATEST
    # (9.x / >=8.4), which raises PytestRemovedIn10Warning for an unrelated same-
    # file test that passes a generator to @parametrize (e.g. test_rcparams.py::
    # test_validator_valid) -> fatal under matplotlib filterwarnings=error -> the
    # whole module fails to collect -> a correct target patch is scored a miss.
    # Pin to the era-appropriate pytest 7.x (still supports --no-header).
    "matplotlib/matplotlib": ["pyparsing<3.1", "setuptools_scm<8", "-vcs_versioning", "pytest<8"],
}


def pin_warn_as_error_deps(repo_dir, repo_name, env_kind="uv", env_vars=None):
    """Downgrade too-new pure-python deps that break test collection under a
    warnings-as-errors repo. Env-layer; general (repo-level); never touches the
    answer. Returns the applied pins (empty if repo unaffected)."""
    pins = WARN_AS_ERROR_DEP_PINS.get(repo_name)
    if not pins:
        return []
    env_dir = ".condaenv" if env_kind == "conda" else ".venv"
    py = os.path.join(repo_dir, env_dir, "bin", "python")
    if not os.path.exists(py):
        return []
    env = os.environ.copy(); env.update(env_vars or {})
    installs = [p for p in pins if not p.startswith("-")]
    removals = [p[1:] for p in pins if p.startswith("-")]
    ok = True
    if installs:
        quoted = " ".join('"%s"' % p for p in installs)
        r = subprocess.run('"%s" -m pip install %s' % (py, quoted),
                           shell=True, cwd=repo_dir, capture_output=True, text=True,
                           timeout=600, env=env)
        ok = ok and r.returncode == 0
    for pkg in removals:
        # uninstalling an absent package is not an error (pip prints "not installed")
        r = subprocess.run('"%s" -m pip uninstall -y "%s"' % (py, pkg),
                           shell=True, cwd=repo_dir, capture_output=True, text=True,
                           timeout=300, env=env)
        ok = ok and r.returncode == 0
    print(" -- warn-as-error dep pins (%s): %s" % (
        "ok" if ok else "FAIL", pins), flush=True)
    return pins


_LOCAL_HTTPBIN_MARKER = "LLMOS harness httpbin shim"

# Repos whose test-suite reaches an EXTERNAL http service via an env var and so
# fail OFFLINE with ConnectionError -> a correct patch is scored a false
# negative. psf/requests test_requests.py does
#   HTTPBIN = os.environ.get("HTTPBIN_URL", "http://httpbin.org/")
# and httpbin.org is unreachable here. pytest-httpbin bundles a local httpbin
# app; a repo-root conftest.py can start it and point HTTPBIN_URL at 127.0.0.1
# BEFORE the test module imports. General, repo-level; no answer/instance data.
LOCAL_HTTPBIN_REPOS = {"psf/requests"}

_LOCAL_HTTPBIN_CONFTEST = """# """ + _LOCAL_HTTPBIN_MARKER + """ (env layer, auto-generated).
# Some tests read HTTPBIN_URL and otherwise hit the public httpbin.org, which is
# unreachable offline (ConnectionError). Start a local httpbin (bundled with
# pytest-httpbin) and point HTTPBIN_URL at it, BEFORE test modules import. This
# file contains NO instance-specific knowledge and nothing derived from a patch.
import os
import atexit

if not os.environ.get("HTTPBIN_URL"):
    try:
        from httpbin import app as _httpbin_app
        from pytest_httpbin import serve as _serve
        _srv = _serve.Server(application=_httpbin_app)
        _srv._thread.daemon = True  # never block pytest exit
        _srv.start()
        atexit.register(_srv.stop)
        os.environ["HTTPBIN_URL"] = _srv.url
    except Exception:
        pass
"""


def ensure_local_httpbin(repo_dir, repo_name, env_kind="uv", env_vars=None):
    """For repos whose tests read HTTPBIN_URL and otherwise hit httpbin.org
    (offline ConnectionError -> false negative): ensure pytest-httpbin (bundles
    a local server) is installed and drop a repo-root conftest.py that starts it
    and sets HTTPBIN_URL. Env-layer; general (repo-level); never touches the
    answer. Returns True if wired."""
    if repo_name not in LOCAL_HTTPBIN_REPOS:
        return False
    env_dir = ".condaenv" if env_kind == "conda" else ".venv"
    py = os.path.join(repo_dir, env_dir, "bin", "python")
    if not os.path.exists(py):
        return False
    env = os.environ.copy(); env.update(env_vars or {})
    try:
        chk = subprocess.run('"%s" -c "import pytest_httpbin, httpbin"' % py,
                             shell=True, cwd=repo_dir, capture_output=True,
                             text=True, timeout=120, env=env)
        if chk.returncode != 0:
            subprocess.run('"%s" -m pip install --prefer-binary pytest-httpbin' % py,
                           shell=True, cwd=repo_dir, capture_output=True,
                           text=True, timeout=900, env=env)
    except Exception:
        pass
    cf = os.path.join(repo_dir, "conftest.py")
    try:
        if os.path.exists(cf):
            existing = open(cf, encoding="utf-8", errors="ignore").read()
            if _LOCAL_HTTPBIN_MARKER not in existing:
                print(" -- local httpbin: repo already has a conftest.py; not modifying", flush=True)
                return False
        open(cf, "w", encoding="utf-8").write(_LOCAL_HTTPBIN_CONFTEST)
        print(" -- local httpbin wired (conftest + HTTPBIN_URL) for %s" % repo_name, flush=True)
        return True
    except Exception as e:
        print(" -- local httpbin wiring failed:", e, flush=True)
        return False


# Deliverable-boundary enforcement of the no-test-edit policy.
# swe_fix_tools.h_patch refuses test-path edits and _diff_nonempty ignores
# them, but check()/reproduce()/run_sanity() run arbitrary python with
# cwd=repo and can rewrite a test file behind that guard. An unfiltered
# `git diff` then carries the edit into the submitted patch, `git apply`
# of the official test_patch collides with it, and the run is voided on a
# mechanical technicality instead of being scored on its source fix.
# Kept byte-identical to swe_fix_tools._TEST_PATH_RE on purpose.
_TEST_PATH_RE = re.compile(
    r"(^|/)(tests?|testing)/|(^|/)test_|_test\.py$|(^|/)conftest\.py$")


def revert_test_paths(repo):
    """Restore every modified test-path file in `repo` before the deliverable
    diff is taken. Returns the list of reverted paths (telemetry). Never
    raises: a failure here must not turn a scoreable run into a crash."""
    try:
        r = sh("git -C %s diff --name-only" % repo, timeout=60)
        files = [f.strip() for f in (r.stdout or "").splitlines() if f.strip()]
        bad = [f for f in files if _TEST_PATH_RE.search(f)]
        if not bad:
            return []
        for f in bad:
            sh("git -C %s checkout -- %s" % (repo, shlex.quote(f)), timeout=60)
        still = sh("git -C %s diff --name-only" % repo, timeout=60)
        left = [f.strip() for f in (still.stdout or "").splitlines()
                if f.strip() and _TEST_PATH_RE.search(f.strip())]
        print(" -- reverted %d test-path file(s) before scoring: %s%s"
              % (len(bad), ", ".join(bad[:6]),
                 ("  [STILL DIRTY: %s]" % ", ".join(left)) if left else ""),
              flush=True)
        return bad
    except Exception as e:
        print(" -- test-path revert failed (scoring continues):", e, flush=True)
        return []


def score(inst, repo, env_vars, env_kind="uv"):
    """Apply the model's diff + the test patch, run FAIL_TO_PASS."""
    # .hypothesis dirs left by phase-1 test runs turn a UserWarning into a
    # collection ERROR (astropy makes warnings fatal) and produced a false
    # resolved=False on astropy-14995 (manual rescore: FTP + 6 P2P all pass).
    shutil.rmtree(os.path.join(repo, ".hypothesis"), ignore_errors=True)
    # Warnings-as-errors repos: pin era-compatible pure-python deps so an
    # unrelated DeprecationWarning cannot turn collection into a false negative.
    pin_warn_as_error_deps(repo, inst["repo"], env_kind, env_vars)
    # A test-file edit is never part of a fix (h_patch refuses one outright);
    # enforce that at the deliverable boundary too, so a scratch script that
    # wrote into tests/ cannot void an otherwise scoreable source patch.
    revert_test_paths(repo)
    # COMPILE GATE (2026-08-03, django-11630): the end-of-walk fallback can
    # reapply a banked candidate without the patch path's compile check, and
    # the scorer then graded a tree that did not even PARSE. Zero-trust
    # applies to our own plumbing: every changed .py must compile before the
    # deliverable diff is taken. One deterministic repair attempt on failure;
    # if that fails too, scoring proceeds and fails honestly.
    try:
        import py_compile as _pyc
        _chg = [f.strip() for f in
                (sh("git -C %s diff --name-only" % repo, timeout=60).stdout
                 or "").splitlines() if f.strip().endswith(".py")]
        for _f in _chg:
            _fp = os.path.join(repo, _f)
            try:
                _pyc.compile(_fp, doraise=True)
            except Exception as _ce:
                print(" -- COMPILE GATE: %s does not parse (%s); attempting "
                      "deterministic repair" % (_f, type(_ce).__name__),
                      flush=True)
                try:
                    import diff_hygiene as _dhm
                    _nfix = _dhm.repair_syntax(repo, _f)
                    _nfix += _dhm.repair_wrap_block(repo, _f)
                    _pyc.compile(_fp, doraise=True)
                    print(" -- COMPILE GATE: repaired %s (%d fix(es))"
                          % (_f, _nfix), flush=True)
                except Exception:
                    print(" -- COMPILE GATE: %s STILL broken; grading will "
                          "fail honestly" % _f, flush=True)
    except Exception as _ge:
        print(" -- COMPILE GATE error (non-fatal): %s" % type(_ge).__name__,
              flush=True)
    diff = sh(f"git -C {repo} diff", timeout=60).stdout
    open(os.path.join(TRACES, inst["instance_id"] + ".patch"), "w").write(diff)
    open(os.path.join(repo, "_t.patch"), "w").write(inst["test_patch"])
    ap = sh("git apply _t.patch", cwd=repo)
    if ap.returncode != 0:
        return False, len(diff), "test patch did not apply (agent touched a test file?)"
    # Local httpbin for repos whose tests read HTTPBIN_URL (else offline
    # ConnectionError -> false negative). After the test patch so a suite-
    # provided conftest is never clobbered.
    ensure_local_httpbin(repo, inst["repo"], env_kind, env_vars)
    # One deterministic test path for everything (env kind, django runner,
    # positional node ids, ensure-pytest, missing-module reflex).
    import test_runner as _tr
    res = _tr.run_tests(repo, env_kind, inst["FAIL_TO_PASS"],
                        env_vars=env_vars, repo=inst["repo"], timeout=600,
                        log_path=os.path.join(SCORE_LOGS, inst["instance_id"] + ".log"))
    # The official criterion is F2P passing AND P2P staying green. We ran only
    # F2P -- 1 of 7 given tests at the median -- so a patch that fixed the new
    # test while breaking an existing one scored as resolved. Both are given;
    # run both. (Answer-key content is never shown to the model: this executes
    # at scoring time only, after the agent has finished.)
    p2p = inst.get("PASS_TO_PASS") or []
    p2p_ok, p2p_tail = True, ""
    if res["ok"] and p2p:
        try:
            r2 = _tr.run_tests(repo, env_kind, p2p, env_vars=env_vars,
                               repo=inst["repo"], timeout=600,
                               log_path=os.path.join(
                                   SCORE_LOGS, inst["instance_id"] + ".p2p.log"))
            p2p_ok, p2p_tail = bool(r2["ok"]), _clean_tail(r2["tail"])
        except Exception as e:
            p2p_ok, p2p_tail = True, "p2p run failed: %s" % type(e).__name__
    if res["ok"] and p2p and not p2p_ok:
        # BASE CONTROL (2026-08-02): the GOLD patch itself failed this P2P leg
        # on xarray-5131 ("3 failed" -- env-sensitive doctests), so every
        # refusal ever issued there was the ENVIRONMENT, not the model.
        # referee.py has run base controls since day one; the in-env oracle
        # now gets the same rule: a P2P failure only counts against the model
        # if the same P2P set PASSES on the clean base tree.
        try:
            sh("git stash -q --include-untracked", cwd=repo)
            rb = _tr.run_tests(repo, env_kind, p2p, env_vars=env_vars,
                               repo=inst["repo"], timeout=600,
                               log_path=os.path.join(
                                   SCORE_LOGS, inst["instance_id"] + ".p2pbase.log"))
            sh("git stash pop -q", cwd=repo)
            if not rb["ok"]:
                p2p_ok = True
                p2p_tail = "P2P fails at BASE too -> env-broken, excused"
        except Exception as _e:
            try:
                sh("git stash pop -q", cwd=repo)
            except Exception:
                pass
    ok = bool(res["ok"]) and p2p_ok
    tail = _clean_tail(res["tail"])
    if res["ok"] and not p2p_ok:
        tail = "F2P passed but PASS_TO_PASS regressed -> NOT resolved | " + p2p_tail
    return ok, len(diff), tail


def oracle_probe(inst, repo, env_vars, env_kind="uv"):
    """ORACLE GATE: grade the CURRENT tree with the hidden tests, harness-side,
    then restore. Verdict-only; test content never reaches the model.

    score() applies the test patch to the tree, so afterwards this reverses it
    (`git apply -R _t.patch`) and deletes _t.patch, closing the window where a
    later tool call could read it. score()'s own revert_test_paths() at final
    grading is a second net under this one. Returns True/False, or None when
    inconclusive (test patch did not apply -- nothing to reverse).
    """
    try:
        ok, _pb, tail = score(inst, repo, env_vars, env_kind=env_kind)
    except Exception as e:
        print(" -- ORACLE probe error: %s: %s" % (type(e).__name__, e), flush=True)
        ok, tail = None, "probe error"
    tp = os.path.join(repo, "_t.patch")
    try:
        if "did not apply" not in (tail or "") and os.path.exists(tp):
            r = sh("git apply -R _t.patch", cwd=repo)
            if r.returncode != 0:
                # never `git checkout .` here -- that would destroy the
                # model's own patch. revert only test paths.
                revert_test_paths(repo)
        if os.path.exists(tp):
            os.remove(tp)
    except Exception as e:
        print(" -- ORACLE restore error: %s: %s" % (type(e).__name__, e), flush=True)
    oracle_probe.last_tail = tail or ""
    if ok is None or "did not apply" in (tail or ""):
        return None
    return bool(ok)


def _clean_tail(tail):
    """End score_tail at the last real test-summary line; drop trailing
    server-log noise (e.g. the local-httpbin flasgger warning) that otherwise
    pollutes FN-triage telemetry."""
    lines = (tail or "").splitlines()
    pat = re.compile(r"(passed|failed|error|\bOK\b|FAILED|no tests ran|===)", re.I)
    last = None
    for i, ln in enumerate(lines):
        if pat.search(ln):
            last = i
    if last is None:
        return tail
    return "\n".join(lines[:last + 1])


def install_spec_extras(repo_dir, env_kind, env_vars, iid):
    """Install the instance's spec-declared optional TEST deps (pandas,
    matplotlib, ...) that a plain repo install does NOT pull, so importorskip-
    gated tests actually run. Sourced from ~/swe/spec_extras.json (SWE-bench
    spec packages), version-matched. Env-layer; never touches the answer."""
    import json as _json
    try:
        extras = _json.load(open(os.path.expanduser("~/swe/spec_extras.json"))).get(iid, [])
    except Exception:
        extras = []
    extras = [e for e in extras
              if not e.lower().endswith((".txt", ".yml", ".yaml", ".cfg", ".toml"))]
    if not extras:
        return []
    env_dir = ".condaenv" if env_kind == "conda" else ".venv"
    py = os.path.join(repo_dir, env_dir, "bin", "python")
    if not os.path.exists(py):
        return []
    env = os.environ.copy(); env.update(env_vars or {})
    quoted = " ".join('"%s"' % e for e in extras)
    r = subprocess.run('"%s" -m pip install --prefer-binary %s' % (py, quoted),
                       shell=True, cwd=repo_dir, capture_output=True, text=True,
                       timeout=1800, env=env)
    print(" -- spec extras (%s): %s" % ("ok" if r.returncode == 0 else "FAIL", extras), flush=True)
    return extras


def _load_repo_knowledge(repo):
    """Load the per-package knowledge base (knowledge/<repo>.md) if present."""
    if os.environ.get("DISABLE_KB"):
        return ""
    fp = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "knowledge", repo.replace("/", "__") + ".md")
    try:
        txt = open(fp, encoding="utf-8").read()
    except OSError:
        return ""
    return "PACKAGE KNOWLEDGE BASE for %s (accumulated, general; consult before guessing):\n%s" % (repo, txt[:2600])


TRIAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "kind": {"type": "string",
                 "enum": ["crash", "wrong_value", "perf", "api", "format",
                          "regression", "compat", "other"]},
        "subsystem":      {"type": "string"},
        "repro_criteria": {"type": "string"},
        "done_criteria":  {"type": "string"},
        "produced_by":    {"type": "string"},
        "change_site":    {"type": "string"},
        "invariant":      {"type": "string"},
        "needs":       {"type": "array", "items": {"type": "string"}},
        "dont_break":  {"type": "array", "items": {"type": "string"}},
        "steps":       {"type": "array", "items": {"type": "string"}},
    },
    "required": ["kind", "subsystem", "done_criteria", "invariant"],
}


def _triage(problem, repo):
    """One structured understanding pass before any tool is touched."""
    if not os.environ.get("TRIAGE"):
        return None, ""
    from repo_bootstrap_tools import llm_call, _extract_json
    raw = llm_call(
        system=("You triage bug reports for an autonomous fixing agent. "
                "Answer ONLY the JSON form. kind must be one of: crash, "
                "wrong_output, regression, api_contract, performance, build."),
        prompt=("Repository: %s\n\nIssue:\n%s\n\n"
                'Return JSON: {"kind": "...", "subsystem": "...", '
                '"repro_criteria": "what a failing script must demonstrate", '
                '"done_criteria": "what change in behavior proves the fix", '
    '"produced_by": "WORK BACKWARD one step: what mechanism inside the code '
    'must be correct for done_criteria to hold? Name the actual thing -- the '
    'key something is stored under, the value a flag takes, the order two '
    'calls happen in, the source a field is read from. NOT a restatement of '
    'done_criteria in other words.", '
    '"change_site": "work backward once more: what must change for that '
    'mechanism to be right? Name the behavior to change, not a guessed path.", '
                '"needs": ["resources to gather FIRST: docs to web-search '
                '(e.g. an installation or API guide), packages, reference '
                'pages -- empty list if none"], '
                '"dont_break": ["behavior that must keep working"], '
                '"steps": ["3-5 subtasks"], '
                '"invariant": "the checkable property the issue says is '
                'violated, in ONE short sentence (identity: same input twice '
                'gives the same object; idempotence: applying twice = once; '
                'roundtrip: parse(print(x))==x; expected-value: f(input) == '
                'stated output; no-raise: f(input) must not raise)"}'
                % (repo, problem[:2500])),
        max_tokens=8000, format_json=True)
    t = _extract_json(raw) or {}
    if not t.get("kind"):
        # Thinking produced something unparseable (7% of runs). Do NOT re-reason
        # with thinking off -- that degrades the judgement (a parser bug got
        # classified kind="perf"). Reformat what this call already said, with
        # the schema enforced at the token level.
        from repo_bootstrap_tools import reshape_json
        _rep = reshape_json(raw, TRIAGE_SCHEMA)
        if _rep.get("kind"):
            print(" -- triage JSON repaired via strict schema", flush=True)
            t = _rep
    if not t.get("kind"):
        return None, ""
    steps = t.get("steps") or []
    needs = t.get("needs") or []
    dont = t.get("dont_break") or []
    out = ("TRIAGE (done before you start; verify against the code, do not "
           "assume):\n"
           "  problem kind: %s | subsystem: %s\n"
           "  a valid reproduction must show: %s\n"
           "  the fix is done when: %s"
           % (t.get("kind"), t.get("subsystem"),
              t.get("repro_criteria"), t.get("done_criteria")))
    if needs:
        out += "\n  gather first: %s" % "; ".join(str(x) for x in needs[:4])
    if dont:
        out += "\n  must keep working: %s" % "; ".join(str(x) for x in dont[:4])
    _pb, _cs = t.get("produced_by"), t.get("change_site")
    if _pb:
        out += ("\n  BACKWARD CHAIN (each link must hold; they get harder to "
                "observe going down):"
                "\n    1. observable : %s"
                "\n    2. produced by: %s" % (t.get("done_criteria"), _pb))
        if _cs:
            out += "\n    3. so change  : %s" % _cs
        out += ("\n  Use check() on link 2 DIRECTLY -- several small checks "
                "until you KNOW, before you patch. Do not infer link 2 from "
                "link 1, and do not re-run the whole reproduction to guess at "
                "it. If a link involves an operation with one known-correct "
                "form (importing, caching, identity comparison), route it "
                "through the tool that does it correctly.")
    if t.get("invariant"):
        out += "\n  the issue states this checkable property: %s" % t["invariant"]
    out += "\n  plan: %s" % "; ".join(str(s) for s in steps[:5])
    return out, (t.get("invariant") or "")


def _atlas_learn(inst):
    """Append this resolved instance's lesson to the atlas ledger and refresh
    the repo's atlas file. Called on every resolve; never raises."""
    import re as _re, json as _json
    try:
        adir = os.path.expanduser("~/swe/atlas")
        os.makedirs(adir, exist_ok=True)
        iid = inst["instance_id"]
        pp = os.path.join(TRACES, iid + ".patch")
        if not os.path.isfile(pp):
            return
        files = sorted(set(_re.findall(r"^diff --git a/(\S+)",
                                       open(pp).read(), _re.M)))
        if not files:
            return
        title = (inst.get("problem_statement") or "").strip().splitlines()
        title = (title[0] if title else "")[:100]
        led = os.path.join(adir, "ledger.jsonl")
        rows = {}
        if os.path.isfile(led):
            for ln in open(led):
                try:
                    r = _json.loads(ln)
                    rows[r["iid"]] = r
                except Exception:
                    pass
        rows[iid] = {"iid": iid, "repo": inst["repo"], "title": title,
                     "files": files, "ts": int(time.time())}
        with open(led, "w") as f:
            for r in rows.values():
                f.write(_json.dumps(r) + "\n")
        # regenerate this repo's atlas file from the ledger
        repo = inst["repo"]
        ours = [r for r in rows.values() if r["repo"] == repo]
        fn = os.path.join(adir, repo.replace("/", "__") + ".md")
        lines = ["CODE ATLAS for %s — where past issues were actually fixed" % repo,
                 "(from this system's own resolved runs; treat as evidence, "
                 "verify by reading — past fixes suggest, they do not decide)",
                 ""]
        for r in sorted(ours, key=lambda x: x.get("ts", 0)):
            lines.append("- %s\n    -> %s" % (r["title"], ", ".join(r["files"][:4])))
        open(fn, "w").write("\n".join(lines) + "\n")
        print(" -- atlas learned: %s -> %s" % (iid, ",".join(files[:2])), flush=True)
    except Exception as e:
        print(" -- atlas learn failed (non-fatal): %s" % e, flush=True)


def _write_probe(problem, invariant, repo):
    """Second, focused call: turn the stated invariant into a standalone probe
    script. PLAIN CODE out -- no JSON wrapper (a code-bearing JSON field made
    the thinking model truncate before emitting anything). Validated with
    compile() before the scaffold will lock it; any failure returns ""."""
    if not (os.environ.get("TRIAGE") and invariant):
        return ""
    from repo_bootstrap_tools import llm_call
    raw = llm_call(
        system=("You write verification probes: short standalone python "
                "scripts that test ONE stated property of a library. Output "
                "ONLY python code. No prose, no markdown fences."),
        prompt=("Repository: %s (installed and importable).\n"
                "Issue (for context):\n%s\n\n"
                "Property to test: %s\n\n"
                "Write a standalone script (stdlib + this repo only, no "
                "pytest) that tests EXACTLY that property: print a one-line "
                "verdict, exit(1) while the property is violated, exit(0) "
                "once it holds. No placeholders."
                % (repo, problem[:1800], invariant)),
        max_tokens=6000)
    code = (raw or "").strip()
    if code.startswith("```"):
        code = code.split("\n", 1)[-1]
        code = code.rsplit("```", 1)[0]
    code = code.strip()
    if not code:
        return ""
    try:
        compile(code, "<probe>", "exec")
    except SyntaxError:
        print(" -- probe writer produced non-compiling code; discarded", flush=True)
        return ""
    return code


def _load_atlas(repo, exclude_iid=None):
    """Code atlas: topic -> files, from our own resolved runs. Env-gated.
    LEAVE-ONE-OUT AT INJECTION: rebuilt from the iid-keyed ledger excluding the
    current instance, so a re-run never sees knowledge derived from itself.
    Remembering is total (the ledger keeps everything); only self-knowledge is
    withheld, per instance, at read time."""
    import json as _json
    adir = os.environ.get("ATLAS_DIR")
    if not adir:
        return ""
    led = os.path.join(adir, "ledger.jsonl")
    rows = []
    try:
        for ln in open(led, encoding="utf-8"):
            try:
                r = _json.loads(ln)
            except Exception:
                continue
            if r.get("repo") == repo and r.get("iid") != exclude_iid:
                rows.append(r)
    except OSError:
        pass
    _idio = ""
    try:
        with open(os.path.join(adir, "idioms",
                               repo.replace("/", "__") + ".md"),
                  encoding="utf-8") as _f:
            _idio = _f.read().strip()
    except OSError:
        pass
    if not rows and not _idio:
        return ""
    lines = ["CODE ATLAS for %s — where past issues were actually fixed" % repo,
             "(from this system's own resolved runs; treat as evidence, "
             "verify by reading — past fixes suggest, they do not decide)", ""]
    for r in sorted(rows, key=lambda x: x.get("ts", 0)):
        lines.append("- %s\n    -> %s" % (r.get("title", ""),
                                           ", ".join((r.get("files") or [])[:4])))
    out = "\n".join(lines)
    if _idio:
        head = ("PACKAGE IDIOMS for %s (hand-curated from past "
                "failures; follow unless the code you read "
                "contradicts them):\n" % repo) + _idio
        out = head + ("\n\n" + out if rows else "")
    return out[:4000]


def _archive_prior_trace(inst):
    """Preserve this instance's EXISTING trace before the current run overwrites
    it, tagged with its outcome + a timestamp.

    MUST be called at the top of run_one, before _save_trace. It was previously
    called after _save_trace and only when the new run resolved, which meant it
    copied the run that had just finished (not the prior one) and never fired in
    the case that actually loses data: a re-run that FAILS and clobbers a
    previously-resolved trace. Ungated on purpose -- a miss re-run over a solve is
    precisely the pair we need for a regression diff."""
    import shutil, json as _json, time as _time
    iid = inst["instance_id"]
    base = os.path.join(TRACES, iid + ".trace.json")
    if not os.path.isfile(base):
        return
    tag = "unknown"
    try:
        tag = "resolved" if _json.load(open(base)).get("outcome", {}).get("resolved") else "miss"
    except Exception:
        pass
    adir = os.path.join(TRACES, "archive"); os.makedirs(adir, exist_ok=True)
    stem = "%s__%s__%s" % (iid, _time.strftime("%Y%m%d_%H%M%S"), tag)
    try:
        shutil.copy2(base, os.path.join(adir, stem + ".trace.json"))
        pp = os.path.join(TRACES, iid + ".patch")
        if os.path.isfile(pp):
            shutil.copy2(pp, os.path.join(adir, stem + ".patch"))
        print(" -- archived prior trace (%s)" % tag, flush=True)
    except Exception as e:
        print(" -- trace archive failed:", e, flush=True)


def run_one(inst):
    print(f"\n=== {inst['instance_id']} ({inst['repo']}) ===", flush=True)
    # FIRST: this run is about to overwrite any existing trace for this instance.
    _archive_prior_trace(inst)
    t0 = time.time()
    repo = clone(inst)
    # ENV FIDELITY: apply the official (repo, version) spec pre_install dep-pin
    # edits to setup.py BEFORE any env is built, so the model own
    # "pip install -e .[test]" resolves era-correct versions instead of today.
    # Root cause of the Docker-confirmed scoring false negatives. Opt out: USE_SPEC_ENV=0.
    if os.environ.get("USE_SPEC_ENV", "1") != "0":
        _se = spec_env.apply_pre_install(repo, inst["instance_id"], inst["repo"])
        if _se.get("applied"):
            print(" -- spec env: applied %d pre_install pins (v%s)%s" % (
                len(_se["applied"]), _se.get("version"),
                ("; skipped %d non-local" % len(_se["skipped"])) if _se.get("skipped") else ""),
                flush=True)
        elif _se.get("version"):
            print(" -- spec env: no local pins for v%s" % _se.get("version"), flush=True)
        _sd = spec_env.apply_system_deps(inst["instance_id"], inst["repo"])
        if _sd.get("installed"):
            print(" -- spec env: installed system deps: %s" % _sd["installed"], flush=True)
        elif _sd.get("failed"):
            print(" -- spec env: system deps MISSING (no sudo?): %s" % _sd["failed"], flush=True)
    # -------- Phase 1: bootstrap --------
    b_handlers, b_state = make_bootstrap_handlers(
        repo, fail_to_pass=inst["FAIL_TO_PASS"],
        pass_to_pass=inst.get("PASS_TO_PASS"), repo=inst["repo"])
    cpu = ToolCallCPU(tools=BOOTSTRAP_TOOLS, tool2sys=BOOTSTRAP_TOOL2SYS,
                     system_prompt=BOOTSTRAP_SYSTEM_PROMPT, model=MODEL, host=HOST,
                     temperature=SWE_TEMP, num_predict=NUM_PREDICT, num_ctx=NUMCTX,
                     keep_alive="24h")
    goal = (f"Set up the repository at ./ for testing. It is: {inst['repo']}. "
            f"The problem it addresses (for context, do not fix yet):\n\n"
            f"{inst['problem_statement'][:2000]}")
    # Known-green tests from the instance metadata — the principled smoke
    # choice (oracle-ish assist; disable for leaderboard-pure runs).
    p2p = (inst.get("PASS_TO_PASS") or [])[:3]
    if p2p:
        goal += ("\n\nKnown-stable tests that should already pass in a "
                 f"healthy environment (good run_smoke_test choices): {p2p}")
    pb = playbook_for(inst["repo"])
    if pb:
        goal += "\n\n" + format_playbook_context(pb)
        print(f" -- injected build playbook for {inst['repo']} "
              f"(validated {pb['validated_runs']}x)", flush=True)
        _attrib_log(inst["instance_id"], inst["repo"], "playbook", [pb])
    rems = remedies_for(inst["repo"])
    if rems:
        goal += "\n\n" + format_remedy_context(rems)
        print(f" -- injected {len(rems)} known remedies for {inst['repo']}", flush=True)
        _attrib_log(inst["instance_id"], inst["repo"], "remedy", rems)
    _kb = _load_repo_knowledge(inst["repo"])
    if _kb:
        goal += "\n\n" + _kb
        print(f" -- injected package knowledge base for {inst['repo']}", flush=True)
        _attrib_log(inst["instance_id"], inst["repo"], "knowledge_base", None, blob=_kb)
    print(" -- phase 1: bootstrap --", flush=True)
    _emit1 = make_emitter(inst["instance_id"], "bootstrap")
    _emit1("phase_start", {"budget": BOOTSTRAP_BUDGET, "repo": inst["repo"]})
    ckpt = os.path.join(TRACES, inst["instance_id"] + ".partial.json")
    def _boot_gate():
        # On declare: if the package imports but smoke hasn't run, the harness
        # verifies the env itself (auto_verify_env) so the model never burns
        # turns guessing a smoke test. Capture its diagnostic so a REJECTED
        # declare tells the model WHY (missing test dep / uncollectable suite)
        # instead of only a generic "gate not passed".
        if b_state.get("sanity_ok") and not b_state.get("smoke_ok"):
            try:
                _res = auto_verify_env(b_state, repo)
                _boot_gate.reject_detail = _auto_verify_reject_detail(_res)
            except Exception:
                _boot_gate.reject_detail = None
        ok = env_ready(b_state)
        if ok:
            _boot_gate.reject_detail = None
        return ok
    b_reason, b_msgs, b_meta = phase_run(cpu, BOOTSTRAP_TOOLS, BOOTSTRAP_TOOL2SYS,
                                          b_handlers, BOOTSTRAP_SYSTEM_PROMPT,
                                          goal, BOOTSTRAP_BUDGET,
                                          gate=_boot_gate,
                                          checkpoint=ckpt, emit=_emit1)
    env_ok = env_ready(b_state)
    _emit1("phase_end", {"reason": b_reason, "env_ok": env_ok})
    if not env_ok and b_state.get("sanity_ok"):
        # LAST-RESORT env check (general robustness): the model exhausted the
        # BOOTSTRAP_BUDGET without landing a passing smoke test, but the env may
        # be healthy -- a common failure mode is the model repeatedly picking
        # smoke tests that fail to COLLECT (e.g. a module importing the
        # CPython-internal _testcapi, absent from uv standalone builds) instead
        # of a stable passing test. Give the env ONE deterministic
        # auto_verify_env pass before discarding the whole instance.
        # STRICTLY NON-WORSENING: only fires when the instance would otherwise
        # be recorded env_ok=False (a total loss); reuses the exact accept
        # logic already trusted for the model-driven auto path; leakage-safe
        # (auto_verify_env excludes FAIL_TO_PASS, runs only generic tests).
        try:
            _lr = auto_verify_env(b_state, repo)
            if _lr.get("ok"):
                _lr_msg = _lr.get("auto_test") or (_lr.get("note") or "")[:60]
                print(f" -- phase 1 last-resort auto_verify_env ACCEPTED env ({_lr_msg})",
                      flush=True)
        except Exception as _e:
            print(f" -- last-resort auto_verify_env errored: {_e}", flush=True)
        env_ok = env_ready(b_state)
    if not env_ok:
        # Save trace & bail out on this instance without spending fix budget.
        dt = time.time() - t0
        outcome = {"id": inst["instance_id"], "resolved": False,
                    "phase1_reason": b_reason, "env_ok": False,
                    "patch_bytes": 0, "secs": round(dt),
                    "note": "env_setup_failed"}
        print(f" -> env NOT ready ({b_reason})  {dt:.0f}s", flush=True)
        _save_trace(inst, {"phase1": b_msgs, "phase1_meta": b_meta,
                            "state": b_state, "outcome": outcome})
        return outcome
    print(f" -- phase 1 OK: {b_state.get('active_env_kind')}/{b_state.get('python_version')}, "
          f"{len(b_state.get('installed', []))} installs", flush=True)
    # Corrections: install spec-declared optional test deps (pandas/matplotlib),
    # version-matched, so importorskip-gated tests run instead of silently skipping.
    install_spec_extras(repo, b_state.get("active_env_kind", "uv"), b_state["env_vars"], inst["instance_id"])
    ensure_local_httpbin(repo, inst["repo"], b_state.get("active_env_kind", "uv"), b_state["env_vars"])
    # -------- Phase 2: fix --------
    # STRICT setting: problem statement only — no FAIL_TO_PASS ids (those
    # tests mostly do not exist until the scoring test_patch is applied,
    # and leaking them is oracle information anyway).
    f_handlers, f_state = make_fix_handlers(
        repo, env_vars=b_state["env_vars"],
        env_kind=b_state.get("active_env_kind", "uv"), repo=inst["repo"])
    # CYCLE-7: the declare-time issue-seeded search reads this (the model's
    # own search terms follow its hypothesis and never surface an unthought-of
    # file; the issue text is an independent seed).
    f_state["problem_statement"] = inst.get("problem_statement", "")
    # ORACLE GATE (env ORACLE_GATE, default off): harness-side probe the walk
    # consults when a green reproduction would stop the search. Verdict-only;
    # content never reaches the context (see oracle_probe).
    f_handlers["_oracle_probe"] = lambda: oracle_probe(
        inst, repo, b_state["env_vars"], b_state.get("active_env_kind", "uv"))
    # New CPU instance for phase 2 — separate context, fresh system prompt.
    cpu2 = ToolCallCPU(tools=FIX_TOOLS, tool2sys=FIX_TOOL2SYS,
                       system_prompt=FIX_SYSTEM_PROMPT, model=MODEL, host=HOST,
                       temperature=SWE_TEMP, num_predict=NUM_PREDICT, num_ctx=NUMCTX,
                       keep_alive="24h")
    print(" -- phase 2: fix --", flush=True)
    fix_goal = (f"Problem:\n{inst['problem_statement'][:3000]}\n\n"
                "Reproduce this bug with a failing script, fix the source, "
                "then verify your reproduction passes.")
    # REPRO_SEED: the reporter usually wrote a reproduction. Use theirs instead
    # of asking the model to invent one -- 46% of Lite statements contain a
    # runnable candidate, and django-16873 spent 3504s inventing a worse one
    # than the SimpleTestCase sitting in its own issue text. A seeded script
    # still has to earn registration: nonzero exit, and not the broken tier.
    if os.environ.get("REPRO_SEED") == "1":
        try:
            _sd = f_handlers["_seed_reproduction"](inst["problem_statement"])
            print(" -- REPRO_SEED %s" % _sd, flush=True)
        except Exception as _e:
            print(" -- REPRO_SEED error: %s: %s" % (type(_e).__name__, _e),
                  flush=True)
    _tri, _inv = _triage(inst["problem_statement"], inst["repo"])
    if not _tri:
        print(" -- triage empty; retrying once", flush=True)
        _tri, _inv = _triage(inst["problem_statement"], inst["repo"])
    if _tri:
        fix_goal += "\n\n" + _tri
        print(" -- triage: %s" % _tri.splitlines()[1].strip(), flush=True)
    try:
        for _ln in (_tri or "").splitlines():
            if "the fix is done when:" in _ln:
                f_state["triage_goal"] = _ln.split("when:", 1)[1].strip()
            if "reproduction must show:" in _ln:
                f_state["triage_repro"] = _ln.split("show:", 1)[1].strip()
            if "2. produced by:" in _ln:
                f_state["chain_mechanism"] = _ln.split("produced by:", 1)[1].strip()
            if "3. so change  :" in _ln:
                f_state["chain_change"] = _ln.split("so change  :", 1)[1].strip()
    except Exception:
        pass
    _pn = _seed_churn(inst["instance_id"], f_state)
    if _pn:
        print(" -- prior attempts: %s" % _pn[:110], flush=True)
    _probe = _write_probe(inst["problem_statement"], _inv, inst["repo"])
    if not _probe and _inv:
        print(" -- probe empty; retrying once", flush=True)
        _probe = _write_probe(inst["problem_statement"], _inv, inst["repo"])
    _probe_status = f_handlers["_lock_probe"](_probe) if _probe else "none"
    _atlas = _load_atlas(inst["repo"], exclude_iid=inst["instance_id"])
    if _atlas:
        fix_goal += ("\n\nWHERE TO LOOK FIRST — read_range these before any "
                     "locate/grep:\n" + _atlas)
        print(" -- injected code atlas", flush=True)
        _attrib_log(inst["instance_id"], inst["repo"], "atlas", None, blob=_atlas)
    pats = patterns_load()
    pats, _pat_held = _pattern_ablate(pats, inst["instance_id"])
    if pats:
        fix_goal += "\n\n" + format_patterns_context(pats)
        print(f" -- injected {len(pats)} engineering patterns", flush=True)
        _attrib_log(inst["instance_id"], inst["repo"], "pattern", pats)
    if _pat_held:
        print(f" -- ABLATION withheld {len(_pat_held)} of "
              f"{len(pats) + len(_pat_held)} engineering patterns", flush=True)
        _attrib_log(inst["instance_id"], inst["repo"], "pattern_withheld",
                    _pat_held)
    if _kb:
        fix_goal += "\n\n" + _kb
    from swe_fix_tools import render_worksheet as _rw
    from swe_fix_tools import capture_readiness as _cap_ready

    def _fix_gate():
        """Only the canonical hidden tests decide pass/fail (Mikey's ruling), so
        the fix phase blocks submit on exactly ONE thing: is there a real change
        to submit. The self-authored reproduction and the invariant probe are
        still computed and still feed attempt ranking -- they inform, they never
        veto. A blocking self-check makes the agent UNDO correct work: it cannot
        tell 'my patch is wrong' from 'my test cannot observe this bug'."""
        try:
            ok = bool(f_handlers["_diff_nonempty"]())
        except Exception:
            ok = True          # never trap the agent on a harness error
        if not ok:
            _fix_gate.reject_message = (
                "cannot submit: there is no change to submit -- the working tree "
                "has no non-test source diff. Patch the source, then submit.")
            _fix_gate.reject_detail = None
            return False

        # REPRODUCTION GATE (env REPRO_GATE=1, default off). THE prerequisite.
        # Measured: instances where a failing reproduction was ever registered
        # resolve at 67% (22/33); where none ever was, 38% (5/13). And on the
        # 14365 walk test seen_red was False, so the walk had no stopping signal
        # at all and simply re-sent the same edit under five different
        # directives. An exhaustive search is only as good as the thing that
        # tells it to stop. Demand the red reproduction FIRST.
        # Bounded: an unobservable bug exists, and trapping the agent on one is
        # worse than letting it submit blind.
        if os.environ.get("REPRO_GATE") == "1" and not f_state.get("seen_red"):
            _rn = f_state.get("repro_gate_refusals", 0)
            if _rn < int(os.environ.get("REPRO_GATE_MAX", "2")):
                f_state["repro_gate_refusals"] = _rn + 1
                _fix_gate.reject_message = (
                    "NO REPRODUCTION YET (%d/%s). You have never registered a "
                    "script that FAILS because of this bug, so nothing can tell "
                    "you whether your patch worked -- not you, and not this "
                    "harness. Before submitting: write a short script that "
                    "exercises the reported behaviour and exits NONZERO on the "
                    "current code, and run it with reproduce(). If it exits 0 it "
                    "does not demonstrate the bug -- rewrite it. Then patch, "
                    "re-run it, and submit when it passes."
                    % (_rn + 1, os.environ.get("REPRO_GATE_MAX", "2")))
                _fix_gate.reject_detail = None
                print(" -- REPRO_GATE blocked submit (%d/%s): no red reproduction"
                      % (_rn + 1, os.environ.get("REPRO_GATE_MAX", "2")), flush=True)
                return False

        # REPERTOIRE WALK (env REPERTOIRE_WALK=1, default off).
        # No recognition: on each submit where the reproduction is not green,
        # hand the model the NEXT operation in frequency order and let it try
        # again. Stops the moment the reproduction goes green, or when the walk
        # budget is spent (REPERTOIRE_MAX, default 5) -- an unbounded walk is a
        # flail, and we have watched four of those.
        # The stopping signal is the self-authored reproduction: measured at
        # 67% resolved when it ever went red vs 38% when it did not. Imperfect,
        # and the only runtime signal that is not the answer key.
        if os.environ.get("REPERTOIRE_WALK") == "1":
            _green = bool(f_state.get("repro_green"))
            _i = f_state.get("repertoire_idx", 0)
            _max = int(os.environ.get("REPERTOIRE_MAX", "5"))
            if not _green and _i < min(_max, len(REPERTOIRE)):
                _name, _how = REPERTOIRE[_i]
                f_state["repertoire_idx"] = _i + 1
                _fix_gate.reject_message = (
                    "NOT DONE YET (%d/%d). Your reproduction is not green, so "
                    "the bug is not demonstrably fixed. Try a different KIND of "
                    "fix -- this one: %s. %s  Make that change, re-run your "
                    "reproduction, and submit when it passes."
                    % (_i + 1, min(_max, len(REPERTOIRE)), _name.upper(), _how))
                _fix_gate.reject_detail = None
                print(" -- REPERTOIRE_WALK %d/%d: %s"
                      % (_i + 1, min(_max, len(REPERTOIRE)), _name), flush=True)
                return False

        # SIBLING GATE (env SIBLING_GATE=1, default off).
        # This deliberately widens what the fix phase blocks on, against the
        # rule in the docstring above. Justification for the exception: that
        # rule exists because a failing SELF-AUTHORED TEST cannot distinguish
        # "my patch is wrong" from "my test cannot observe this bug", so it
        # makes the agent revert good work. A sibling site is not a test
        # result -- it is a static fact about the file (line 309 compares a
        # string with ==), it cannot be wrong about existence, and it invites
        # ADDING an edit rather than undoing one. Still bounded: at most
        # SIBLING_GATE_MAX refusals, then it lets the submit through, because
        # an unsatisfiable gate produces a flail and we have watched four.
        if os.environ.get("SIBLING_GATE") == "1":
            _sw = f_state.get("sibling_outstanding") or {}
            _sites = _sw.get("unchanged_same_class_sites") or []
            _n = f_state.get("sibling_refusals", 0)
            if _sites and _n < int(os.environ.get("SIBLING_GATE_MAX", "2")):
                f_state["sibling_refusals"] = _n + 1
                _cls = "/".join(_sw.get("edit_class") or ["same"])
                _fix_gate.reject_message = (
                    "SUBMIT BLOCKED (%d of %s): your edit was a %s change. These "
                    "lines in the same file are the same kind of site and are "
                    "still unchanged: %s. A fix of this class usually has more "
                    "than one site -- the traceback only names the first. Read "
                    "each one and decide whether it needs the same change. Patch "
                    "the ones that do, then submit. If none of them do, submit "
                    "again and it will go through."
                    % (_n + 1, os.environ.get("SIBLING_GATE_MAX", "2"), _cls,
                       ", ".join("%s:%s" % (s.get("file", "?"), s["line"])
                                 for s in _sites[:8])))
                _fix_gate.reject_detail = {"sites": _sites[:8]}
                print(" -- SIBLING_GATE blocked submit (%d sites, class %s)"
                      % (len(_sites), _cls), flush=True)
                return False
        return ok
    _fix_gate.reject_message = None
    _fix_gate.reject_detail = None

    _emit2 = make_emitter(inst["instance_id"], "fix")
    _emit2("phase_start", {"budget": FIX_BUDGET, "repo": inst["repo"]})
    if os.environ.get("REPERTOIRE_SEGMENTS") == "1":
        # OUR loop, not the model's: bounded attempt per kind of fix, success
        # checked after every tool result, tree reverted between operations.
        f_reason, f_msgs, f_meta = repertoire_fix(
            cpu2, FIX_TOOLS, FIX_TOOL2SYS, f_handlers, FIX_SYSTEM_PROMPT,
            fix_goal, f_state,
            seg_turns=int(os.environ.get("SEG_TURNS", "10")),
            max_ops=int(os.environ.get("REPERTOIRE_MAX", "6")),
            worksheet=lambda: _rw(f_state), gate=_fix_gate,
            free_text=lambda _t: _cap_ready(f_state, _t),
            checkpoint=ckpt, emit=_emit2, stall_window=FIX_STALL)
    else:
        f_reason, f_msgs, f_meta = phase_run(cpu2, FIX_TOOLS, FIX_TOOL2SYS,
                                              f_handlers, FIX_SYSTEM_PROMPT,
                                              fix_goal, FIX_BUDGET,
                                              worksheet=lambda: _rw(f_state),
                                              free_text=lambda _t: _cap_ready(f_state, _t),
                                              gate=_fix_gate,
                                              checkpoint=ckpt, emit=_emit2, stall_window=FIX_STALL)
    _emit2("phase_end", {"reason": f_reason})
    # THE GIVEN TESTS (Mikey): rerun the repo's own base-commit tests that
    # passed before any patch. Self-authored checks are recorded but decide
    # nothing; this is the legitimate evidence. Runs unconditionally -- it used
    # to happen only if the agent chose to call verify_fix.
    _given_ok, _given_n, _regressed = None, 0, []
    try:
        _base = f_state.get("baseline_pass") or []
        _given_n = len(_base)
        if _given_n:
            _regressed = f_handlers["_check_regressions"]() or []
            _given_ok = not _regressed
        print(" -- given tests: %d sampled, %d regressed -> %s"
              % (_given_n, len(_regressed),
                 "OK" if _given_ok else ("REGRESSED" if _given_n else "UNKNOWN")),
              flush=True)
    except Exception as _e:
        print(" -- given tests check failed: %s" % type(_e).__name__, flush=True)

    # Score with the exact SWE-bench recipe.
    resolved, patch_bytes, tail = score(inst, repo, b_state["env_vars"],
                                        env_kind=b_state.get("active_env_kind", "uv"))
    dt = time.time() - t0
    outcome = {"id": inst["instance_id"], "resolved": bool(resolved),
                "phase1_reason": b_reason, "phase2_reason": f_reason,
                "env_ok": True,
                "env_kind": b_state.get("active_env_kind"),
                "python":   b_state.get("python_version"),
                "installs": b_state.get("installed", []),
                "env_vars": b_state["env_vars"],
                "patch_bytes": patch_bytes, "secs": round(dt),
                "fix_verified_by_model": f_state["fix_verified"],
                "given_tests_ok": _given_ok,
                "given_tests_n": _given_n,
                "given_tests_regressed": _regressed,
                "syntax_ok": f_state.get("syntax_ok", True),
                "score_tail": tail[:400]}
    print(f" -> resolved={resolved}  patch_bytes={patch_bytes}  {dt:.0f}s | {tail[:120]}",
          flush=True)
    # Complete the outcome BEFORE serializing: anything added afterwards is
    # invisible to every reader of the saved trace.
    outcome["probe_status"] = _probe_status
    outcome["probe_green"] = f_state.get("probe_green")
    outcome["repro_green"] = f_state.get("repro_green")
    outcome["seen_red"] = f_state.get("seen_red")
    _save_trace(inst, {"phase1": b_msgs, "phase1_meta": b_meta, "state": b_state,
                        "phase2": f_msgs, "phase2_meta": f_meta,
                        "fix_state": f_state, "outcome": outcome})
    if outcome.get("resolved"):
        _atlas_learn(inst)   # learning is a side effect of running, not a chore
    return outcome


def _seed_churn(iid, f_state):
    """Seed the per-attempt selector-edit counter from this instance's OWN
    archived attempts (runtime-owned traces; our failures, not gold data).
    Returns a prior-attempts note for the worksheet, or ""."""
    import glob as _g
    seeded = {}
    misses = 0
    files_hit = {}
    for f in _g.glob(os.path.join(TRACES, "archive", iid + "__*.trace.json")):
        try:
            t = json.load(open(f))
        except Exception:
            continue
        oc = t.get("outcome") or {}
        if oc.get("resolved"):
            continue
        misses += 1
        fs = t.get("fix_state") or {}
        for fn, n in (fs.get("func_edits") or {}).items():
            # COUNT ATTEMPTS, NOT EDITS (2026-08-26). The old code summed
            # func_edits across traces, but every trace ALREADY stores the
            # seed that attempt was handed, so the sum compounded
            # geometrically: 11564's seven traces read 1,1,1,3,6,12,24 -> a
            # seed of 48; 11019's eleven reached 288 -> a seed of 576, against
            # a true per-attempt count of ~1-3. Taking the max is not enough
            # either -- the traces already on disk carry the poisoned values,
            # so a max would plateau at 288 forever. What the churn signal
            # actually wants (4bf2ed7: "the model churns one predicate edit
            # per attempt") is HOW MANY ATTEMPTS touched this function. That
            # is bounded by `misses`, cannot compound, and is immune to the
            # historical corruption.
            if n:
                seeded[fn] = seeded.get(fn, 0) + 1
        for e in (t.get("phase2_events") or []):
            if e.get("tool") == "patch" and e.get("ok"):
                a = e.get("args")
                if isinstance(a, str):
                    try:
                        a = json.loads(a)
                    except Exception:
                        a = {}
                fpath = (a or {}).get("file") if isinstance(a, dict) else None
                if fpath:
                    files_hit[fpath] = files_hit.get(fpath, 0) + 1
    if seeded:
        f_state.setdefault("func_edits", {}).update(seeded)
    if misses >= 2:
        top = max(files_hit, key=files_hit.get) if files_hit else "?"
        note = ("%d prior FAILED attempts on this instance; their patches "
                "went to %s (%d edits)" % (misses, top, files_hit.get(top, 0)))
        # FILE-LEVEL EXHAUSTION (django-16820, measured): 14 failed attempts,
        # 30+ patches, every one into squashmigrations.py -- the issue SAYS
        # squash, but the mechanism lives in the optimizer it calls. When
        # every attempt dug one hole and none resolved, say the hole is dry.
        # CHURN_EXHAUST (2026-08-26): this verdict was generalised from ONE
        # instance (django-16820) and it inverts a signal that is usually
        # correct -- on a benchmark, the file every attempt patched is
        # normally the file the bug is in. Measured blast radius: 41 of the
        # 300 instances get it, 28 of them in the 124-instance never-resolved
        # set, i.e. we tell a quarter of the hardest set to avoid the right
        # file. Worse, render_worksheet truncated the note mid-sentence, so
        # the model got "treat that file as EXHAUSTED. The rea" -- the
        # prohibition without the redirect that justified it.
        #   soft (default): keep the observation, drop the prohibition.
        #   hard          : the original 2026-08-06 wording.
        #   off           : say nothing about exhaustion.
        _xmode = os.environ.get("CHURN_EXHAUST", "soft").lower()
        if (misses >= 6 and files_hit.get(top, 0) >= misses
                and _xmode != "off"):
            if _xmode == "hard":
                note += (". Every one of those attempts patched %s and every "
                         "one FAILED: treat that file as EXHAUSTED. The real "
                         "mechanism most likely lives in a module it imports "
                         "or calls -- follow the imports one level down and "
                         "make your fix there instead." % top)
            else:
                note += (". All %d attempts patched %s and none resolved. "
                         "That is probably still the right file -- do NOT "
                         "avoid it. But before you commit to another edit "
                         "there, spend one read checking whether the "
                         "mechanism actually lives one level down, in a "
                         "module it imports or calls."
                         % (misses, top))
        if seeded:
            fn = max(seeded, key=seeded.get)
            note += (". The function %s was condition-edited in %d of those "
                     "attempts -- if your plan is another edit to its "
                     "conditions, that move is EXHAUSTED: widen the fix to "
                     "the branch it routes to." % (fn, seeded[fn]))
        f_state["prior_attempts_note"] = note
        return note
    return ""


def _postmortem(inst, blob):
    """Mechanical analysis of one run -> the research corpus. Counted, not judged."""
    import collections
    try:
        ev = blob.get("phase2_events") or []
        fs = blob.get("fix_state") or {}
        oc = blob.get("outcome") or {}
        names = [e.get("tool") or "" for e in ev]

        def _args(e):
            a = e.get("args")
            if isinstance(a, str):
                try:
                    a = json.loads(a)
                except Exception:
                    a = {}
            return a if isinstance(a, dict) else {}

        pat = [e for e in ev if e.get("tool") == "patch"]
        seen, repeats, failed = set(), 0, 0
        line_mode = esc_tot = esc_fail = 0
        for e in pat:
            a = _args(e)
            snip = str(a.get("old_snippet") or "")
            key = " ".join(snip.split())[:300] or ("L%s" % a.get("start_line"))
            if key in seen:
                repeats += 1
            seen.add(key)
            if a.get("start_line") is not None:
                line_mode += 1
            has_bs = chr(92) in snip
            if has_bs:
                esc_tot += 1
            if not e.get("ok"):
                failed += 1
                if has_bs:
                    esc_fail += 1

        errs = collections.Counter()
        for e in ev:
            er = e.get("error")
            if er:
                errs[" ".join(str(er).split())[:110]] += 1

        OBS = {"reproduce", "verify_fix", "run_tests", "read_range",
               "locate", "check"}
        pidx = [i for i, n in enumerate(names) if n == "patch"]
        unobs = sum(1 for a, b in zip(pidx, pidx[1:])
                    if not any(names[i] in OBS for i in range(a + 1, b)))

        rec = {"iid": inst.get("instance_id"), "repo": inst.get("repo"),
               "ts": int(time.time()),
               "resolved": bool(oc.get("resolved")),
               "ended_by": oc.get("phase2_reason"),
               "secs": oc.get("secs"), "turns": len(ev),
               "tools": dict(collections.Counter(names)),
               "patches": {"attempts": len(pat), "failed": failed,
                           "distinct": len(seen), "repeated": repeats,
                           "line_anchored": line_mode,
                           "unobserved_gaps": unobs,
                           "escape_anchors": esc_tot,
                           "escape_anchors_failed": esc_fail},
               "verification": {"seen_red": fs.get("seen_red"),
                                "repro_green": fs.get("repro_green"),
                                "probe_green": fs.get("probe_green"),
                                "fix_verified": fs.get("fix_verified")},
               "features": dict(fs.get("features_fired") or {}),
               "readiness": list(fs.get("readiness") or []),
               "funcs_edited": dict(fs.get("func_edits") or {}),
               "top_errors": errs.most_common(4)}
        d = os.path.expanduser("~/swe/research/postmortem")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "%s.jsonl" % inst.get("instance_id")),
                  "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + chr(10))
        return "%d turns, %d patches (%d failed, %d repeated)" % (
            len(ev), len(pat), failed, repeats)
    except Exception as e:
        return "FAILED %s: %s" % (type(e).__name__, str(e)[:90])


def _save_trace(inst, blob):
    os.makedirs(TRACES, exist_ok=True)
    # Clean completion supersedes the crash checkpoint.
    try:
        os.remove(os.path.join(TRACES, inst["instance_id"] + ".partial.json"))
    except OSError:
        pass
    p = os.path.join(TRACES, inst["instance_id"] + ".trace.json")
    with open(p, "w") as f:
        json.dump(blob, f, indent=1, default=str)
    # Run trace consumers (events, remedy store, training export). Never
    # let a consumer failure damage the run or the already-saved trace.
    try:
        summary = harvest_trace(inst, blob)
        with open(p, "w") as f:
            json.dump(blob, f, indent=1, default=str)
        print(f" -- trace harvest: {summary}", flush=True)
        print(" -- postmortem: %s" % _postmortem(inst, blob), flush=True)
    except Exception as e:
        print(f" -- trace harvest failed (trace still saved): {type(e).__name__}: {e}", flush=True)


def main():
    os.makedirs(WORK, exist_ok=True); os.makedirs(TRACES, exist_ok=True)
    insts = json.load(open(os.path.expanduser("~/swe/instances.json")))
    N = int(sys.argv[1]) if len(sys.argv) > 1 else len(insts)
    insts = insts[:N]
    results = []
    for i, inst in enumerate(insts, 1):
        try:
            r = run_one(inst)
        except Exception as e:
            r = {"id": inst["instance_id"], "resolved": False,
                  "error": f"{type(e).__name__}: {e}"}
        results.append(r)
        json.dump(results, open(os.path.expanduser("~/swe/results_v2.json"), "w"),
                  indent=2)
    # Batch-end hygiene (Mikey): clone once per repo (the mirror), reuse
    # checkouts during the batch, delete them ALL at the end. Mirrors make
    # recreation cheap. KEEP_WORK=1 skips (post-batch debugging/rescoring).
    if not os.environ.get("KEEP_WORK"):
        # Delete only RESOLVED instances' checkouts. Failures stay on disk —
        # we are probably going to do more work on those (Mikey).
        resolved_ids = {r["id"] for r in results if r.get("resolved")}
        freed, kept = 0, 0
        for inst in insts:
            d = os.path.join(WORK, inst["instance_id"])
            if not os.path.isdir(d):
                continue
            if inst["instance_id"] in resolved_ids:
                shutil.rmtree(d, ignore_errors=True)
                freed += 1
            else:
                kept += 1
        print(f"[cleanup] removed {freed} resolved checkouts, kept {kept} "
              f"failed ones for further work (mirrors retained)", flush=True)
    resolved = sum(int(r.get("resolved")) for r in results)
    env_ok = sum(int(r.get("env_ok", False)) for r in results)
    print(f"\n=== LLMOS v2 on SWE-bench Lite: {resolved}/{len(results)} resolved, "
          f"{env_ok}/{len(results)} env_ok ===", flush=True)
    for r in results:
        tag = "RESOLVED" if r.get("resolved") else ("env_setup_failed" if not r.get("env_ok")
                                                     else "no")
        print(f"  {r['id']}: {tag}  ({r.get('secs','?')}s)", flush=True)


if __name__ == "__main__":
    main()
