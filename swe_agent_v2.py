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
NUMCTX = 131072
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
    while i < len(ops):
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
        elif i == 0:
            seg_goal = goal
        else:
            seg_goal = (
                "That did not fix it -- your reproduction is still not green, "
                "and the tree has been reverted to its original state. Try a "
                "DIFFERENT KIND of fix now: %s. %s  Make the change, re-run "
                "your reproduction, and stop when it passes."
                % (name.upper(), how))
        log(" -- REPERTOIRE segment %d/%d: %s" % (i + 1, len(ops), name))
        reason, msgs, meta = phase_run(cpu, tools, tool2sys, handlers,
                                       system_prompt, seg_goal, turns_this,
                                       log=log, init_messages=msgs,
                                       success=_corroborated(state, handlers, log),
                                       **kw)
        if reason == "solved" or state.get("repro_green"):
            log(" -- REPERTOIRE solved at segment %d (%s)" % (i + 1, name))
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
                return "declared", msgs, meta

        # (b) AN OPERATION IS NOT SPENT UNTIL IT WAS ACTUALLY ATTEMPTED.
        # Observed: two segments burned their budget (one for 18 turns) without
        # applying a single patch -- consuming a branch of the search while
        # never trying that kind of fix. Grant one extension, then move on.
        if not _changed and not extended:
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
            except Exception as e:
                log(" -- candidate capture failed (%s)" % type(e).__name__)

        try:
            handlers["_revert_tree"]()          # clean slate for the next kind
        except Exception as e:
            log(" -- revert failed (%s); continuing without it" % type(e).__name__)
    log(" -- REPERTOIRE exhausted %d operations without a green reproduction"
        % len(ops))
    # restore the fallback rather than submitting an empty tree
    if candidates:
        _green = [c for c in candidates if len(c) > 2 and c[2]]
        _pick = _green[0] if _green else candidates[0]
        _name, _diff = _pick[0], _pick[1]
        log(" -- fallback: %d candidate(s), %d with a green reproduction; using %s"
            % (len(candidates), len(_green), _name))
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
              budget, gate=None, log=print, checkpoint=None, worksheet=None,
              emit=None, stall_window=None, init_messages=None,
              success=None):
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
    _wall_cap = float(os.environ.get("PHASE_WALL_CAP", "0") or 0)
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
            content = (msg.get("content") or msg.get("reasoning_content")
                       or msg.get("thinking") or "")[:400]
            messages.append({"role": "assistant", "content": content or "..."})
            messages.append({"role": "user",
                              "content": "Call one of the provided tools now."})
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
                    args.get("start_line"),
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
                log(f"  [critic] {advice[:100]}")
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
    ok = bool(res["ok"]) and p2p_ok
    tail = _clean_tail(res["tail"])
    if res["ok"] and not p2p_ok:
        tail = "F2P passed but PASS_TO_PASS regressed -> NOT resolved | " + p2p_tail
    return ok, len(diff), tail


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
    if not rows:
        return ""
    lines = ["CODE ATLAS for %s — where past issues were actually fixed" % repo,
             "(from this system's own resolved runs; treat as evidence, "
             "verify by reading — past fixes suggest, they do not decide)", ""]
    for r in sorted(rows, key=lambda x: x.get("ts", 0)):
        lines.append("- %s\n    -> %s" % (r.get("title", ""),
                                           ", ".join((r.get("files") or [])[:4])))
    return "\n".join(lines)[:3500]


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
    rems = remedies_for(inst["repo"])
    if rems:
        goal += "\n\n" + format_remedy_context(rems)
        print(f" -- injected {len(rems)} known remedies for {inst['repo']}", flush=True)
    _kb = _load_repo_knowledge(inst["repo"])
    if _kb:
        goal += "\n\n" + _kb
        print(f" -- injected package knowledge base for {inst['repo']}", flush=True)
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
    # New CPU instance for phase 2 — separate context, fresh system prompt.
    cpu2 = ToolCallCPU(tools=FIX_TOOLS, tool2sys=FIX_TOOL2SYS,
                       system_prompt=FIX_SYSTEM_PROMPT, model=MODEL, host=HOST,
                       temperature=SWE_TEMP, num_predict=NUM_PREDICT, num_ctx=NUMCTX,
                       keep_alive="24h")
    print(" -- phase 2: fix --", flush=True)
    fix_goal = (f"Problem:\n{inst['problem_statement'][:3000]}\n\n"
                "Reproduce this bug with a failing script, fix the source, "
                "then verify your reproduction passes.")
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
    pats = patterns_load()
    if pats:
        fix_goal += "\n\n" + format_patterns_context(pats)
        print(f" -- injected {len(pats)} engineering patterns", flush=True)
    if _kb:
        fix_goal += "\n\n" + _kb
    from swe_fix_tools import render_worksheet as _rw

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
            checkpoint=ckpt, emit=_emit2, stall_window=FIX_STALL)
    else:
        f_reason, f_msgs, f_meta = phase_run(cpu2, FIX_TOOLS, FIX_TOOL2SYS,
                                              f_handlers, FIX_SYSTEM_PROMPT,
                                              fix_goal, FIX_BUDGET,
                                              worksheet=lambda: _rw(f_state),
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
            seeded[fn] = seeded.get(fn, 0) + n
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
        if seeded:
            fn = max(seeded, key=seeded.get)
            note += (". The function %s has been condition-edited %d time(s) "
                     "across attempts -- if your plan is another edit to its "
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
