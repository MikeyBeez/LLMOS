"""Repertoire-walk unit tests (2026-08-08).

Covers the parts of repertoire_fix that are pure control flow, with phase_run
stubbed out -- no LLM, no network, no GPU, no repo.

  - SEG_ECHO is OFF by default and adds nothing to any segment goal.
  - SEG_ECHO on: a segment whose captured diff is byte-identical to an earlier
    segment's makes the NEXT goal state that as a fact, naming the earlier
    segment number.  This is the sympy-17022 failure: segments 2, 3, 4 and 5
    each produced a 746-byte candidate while being told to try a different
    kind of fix.
  - SEG_ECHO on with genuinely different diffs says nothing.
  - SEGMENT 1 IS THE SAFETY ANCHOR: it runs on the caller's plain goal with no
    operation directive.  Easy to break by "tidying" the i == 0 branch, and
    breaking it silently removes the fair baseline the fallback depends on.
  - REPERTOIRE structural invariants.

Run: cd ~/Code/LLMOS && python3 -m pytest tests/test_repertoire_seg.py -q
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import swe_agent_v2 as SA


class _Env:
    """Set env vars for the duration of a block, restoring what was there."""

    def __init__(self, **kw):
        self.kw = kw
        self.old = {}

    def __enter__(self):
        for k, v in self.kw.items():
            self.old[k] = os.environ.get(k)
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return self

    def __exit__(self, *a):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return False


def walk(diffs, segments=4, **env):
    """Run repertoire_fix over `segments` ops, returning (goals, log_lines).

    `diffs` is the sequence of diffs _capture_diff hands back, one per segment.
    phase_run is stubbed: it records the goal it was given and reports that the
    budget ran out, which is what a segment that fails to fix anything does.
    """
    goals, logs = [], []
    del CAPS[:]
    seq = {"n": 0}

    def fake_phase_run(cpu, tools, tool2sys, handlers, system_prompt, goal,
                       turns, *a, **kw):
        goals.append(goal)
        CAPS.append(kw.get("wall_cap"))
        return "budget", [], {}

    def capture():
        d = diffs[min(seq["n"], len(diffs) - 1)]
        seq["n"] += 1
        return d

    handlers = {
        "_diff_nonempty": lambda: True,
        "_capture_diff": capture,
        "_revert_tree": lambda: None,
        "swe.verify_fix": lambda pcb, args: {"ok": False},
    }
    state = {"seen_red": False}          # keep the harness-verify branch quiet

    real = SA.phase_run
    SA.phase_run = fake_phase_run
    try:
        base = dict(SEG_ECHO=None, REPERTOIRE_WALL="0", PHASE_WALL_CAP="0",
                    COVERAGE_GAP=None, ORACLE_GATE=None, SEG1_TURNS="1")
        base.update(env)
        with _Env(**base):
            SA.repertoire_fix(cpu=None, tools=[], tool2sys={},
                              handlers=handlers, system_prompt="s",
                              goal="PLAIN GOAL", state=state, seg_turns=1,
                              max_ops=segments, log=logs.append)
    finally:
        SA.phase_run = real
    return goals, logs


CAPS = []          # wall_cap handed to each segment, filled by walk()

SAME = "--- a/x.py\n+++ b/x.py\n@@\n-a\n+b\n"
GROWN = SAME + "@@\n-c\n+d\n"          # SAME plus new hunks: a real extension
OTHER = "--- a/y.py\n+++ b/y.py\n@@\n-c\n+d\n"


class SegEchoTest(unittest.TestCase):

    def test_off_by_default(self):
        """Every behaviour change ships behind a flag, default off."""
        goals, logs = walk([SAME] * 4)
        self.assertEqual(len(goals), 4)
        for g in goals:
            self.assertNotIn("BYTE-IDENTICAL", g)
        self.assertFalse([l for l in logs if "SEG_ECHO" in l])

    def test_repeat_is_stated_as_a_fact(self):
        goals, logs = walk([SAME] * 4, SEG_ECHO="1")
        # Segments 1 and 2 cannot know about a repeat yet.
        self.assertNotIn("BYTE-IDENTICAL", goals[0])
        self.assertNotIn("BYTE-IDENTICAL", goals[1])
        # Segment 2 reproduced segment 1's diff, so segment 3 is told.
        self.assertIn("BYTE-IDENTICAL", goals[2])
        self.assertIn("in segment 1", goals[2])
        self.assertIn("CHANGE AN ARGUMENT VALUE", goals[2])
        self.assertTrue([l for l in logs if "SEG_ECHO" in l])

    def test_repeat_note_is_consumed_not_sticky(self):
        """The note names the repeat that just happened, not an old one."""
        goals, _ = walk([SAME, SAME, OTHER, OTHER], segments=4, SEG_ECHO="1")
        self.assertIn("BYTE-IDENTICAL", goals[2])   # segment 2 repeated
        self.assertNotIn("BYTE-IDENTICAL", goals[3])  # segment 3 was novel

    def test_distinct_diffs_say_nothing(self):
        goals, logs = walk([SAME, OTHER, SAME + "x", OTHER + "y"],
                           SEG_ECHO="1")
        for g in goals:
            self.assertNotIn("BYTE-IDENTICAL", g)
        self.assertFalse([l for l in logs if "SEG_ECHO" in l])


class SegmentOneAnchorTest(unittest.TestCase):

    def test_segment_one_runs_on_the_plain_goal(self):
        """No operation directive in segment 1 -- it is the fair baseline."""
        goals, _ = walk([SAME, OTHER], segments=2)
        self.assertEqual(goals[0], "PLAIN GOAL")
        self.assertNotIn("DIFFERENT KIND", goals[0])
        self.assertNotIn(SA.REPERTOIRE[0][0].upper(), goals[0])

    def test_later_segments_carry_their_directive(self):
        goals, _ = walk([SAME, OTHER], segments=2)
        self.assertIn("DIFFERENT KIND", goals[1])
        self.assertIn(SA.REPERTOIRE[1][0].upper(), goals[1])
        self.assertIn(SA.REPERTOIRE[1][1], goals[1])


class RepertoireShapeTest(unittest.TestCase):

    def test_shape(self):
        self.assertEqual(len(SA.REPERTOIRE), 13)
        names = []
        for entry in SA.REPERTOIRE:
            self.assertEqual(len(entry), 2)
            name, how = entry
            self.assertTrue(name and name.islower())
            self.assertTrue(len(how) > 30, name)
            names.append(name)
        self.assertEqual(len(set(names)), len(names))

    def test_max_ops_truncates_from_the_front(self):
        """REPERTOIRE_MAX must keep segment 1 -- it is the fallback."""
        goals, _ = walk([SAME, OTHER, SAME], segments=3)
        self.assertEqual(len(goals), 3)
        self.assertEqual(goals[0], "PLAIN GOAL")
        self.assertIn(SA.REPERTOIRE[2][0].upper(), goals[2])


class Seg1BudgetShareTest(unittest.TestCase):
    """SEG1_WALL_FRAC bounds the safety anchor.

    Segment 1 gets SEG1_TURNS=60 against 20 for the rest, and on fresh32 two
    instances spent the whole 2400s walk inside it -- twelve operations never
    ran, and one of the two finished with no patch at all. The anchor has to
    be a fair baseline, not the entire budget.
    """

    def test_off_by_default(self):
        walk([SAME, OTHER, SAME], segments=3, REPERTOIRE_WALL="1000")
        self.assertGreater(CAPS[0], 900)          # segment 1 gets the lot

    def test_fraction_caps_segment_one_only(self):
        walk([SAME, OTHER, SAME], segments=3, REPERTOIRE_WALL="1000",
             SEG1_WALL_FRAC="0.4")
        self.assertLessEqual(CAPS[0], 400)
        self.assertGreater(CAPS[1], 400)          # later segments untouched

    def test_no_wall_means_no_cap(self):
        walk([SAME, OTHER], segments=2, SEG1_WALL_FRAC="0.4")
        self.assertIsNone(CAPS[0])


class GrowSegTest(unittest.TestCase):
    """GROW_SEG (2026-08-12, cycle 3 of the single-example loop).

    sympy-24909: THREE segments went green on the model's own reproduction
    and ORACLE_GATE refused every one -- then the walk reverted each refused
    candidate and asked for a DIFFERENT kind of fix, so every candidate
    restarted from zero and stayed small (589/589/943 bytes vs gold's
    sibling-covering patch).  A green-but-refused candidate is INCOMPLETE,
    not wrong.  With GROW_SEG=1 the walk keeps it applied (max 2 consecutive
    grows) and the next goal says EXTEND; collateral refusals (PASS_TO_PASS
    regressed) always revert.  Default stays off.
    """

    def tearDown(self):
        if hasattr(SA.oracle_probe, "last_tail"):
            del SA.oracle_probe.last_tail

    def grow_walk(self, segments=4, accept_after=None, last_tail="",
                  grow_bytes=False, **env):
        """Every segment goes green; the oracle refuses until `accept_after`
        greens have been probed (None = refuse forever).  Returns
        (goals, logs, reverts, result_reason)."""
        goals, logs, reverts = [], [], []
        st = {"seen_red": False}
        probes = {"n": 0}

        def fake_phase_run(cpu, tools, tool2sys, handlers, system_prompt,
                           goal, turns, *a, **kw):
            goals.append(goal)
            st["repro_green"] = True
            return "solved", [], {}

        def probe():
            probes["n"] += 1
            if accept_after is not None and probes["n"] > accept_after:
                return True
            return False

        handlers = {
            "_diff_nonempty": lambda: True,
            # GROW_ECHO made byte-identity meaningful: a walk whose capture
            # never changes models a model that never extends. grow=True in
            # the capture models a REAL extension after the first refusal.
            "_capture_diff": (lambda: (GROWN if len(goals) > 1 else SAME))
                             if grow_bytes else (lambda: SAME),
            "_revert_tree": lambda: reverts.append(1),
            "_restore_diff": lambda d: {"restored": True},
            "_oracle_probe": probe,
            "swe.verify_fix": lambda pcb, args: {"ok": False},
        }
        self._probes = probes
        SA.oracle_probe.last_tail = last_tail
        real = SA.phase_run
        SA.phase_run = fake_phase_run
        try:
            base = dict(SEG_ECHO=None, REPERTOIRE_WALL="0", PHASE_WALL_CAP="0",
                        COVERAGE_GAP=None, ORACLE_GATE="1", SEG1_TURNS="1",
                        GROW_SEG=None, BANK_AUDIT=None)
            base.update(env)
            with _Env(**base):
                res = SA.repertoire_fix(cpu=None, tools=[], tool2sys={},
                                        handlers=handlers, system_prompt="s",
                                        goal="PLAIN GOAL", state=st,
                                        seg_turns=1, max_ops=segments,
                                        log=logs.append)
        finally:
            SA.phase_run = real
        return goals, logs, reverts, res[0]

    def test_off_by_default_every_refusal_reverts(self):
        """Every behaviour change ships behind a flag, default off."""
        goals, logs, reverts, _ = self.grow_walk()
        self.assertEqual(len(reverts), 4)
        for g in goals:
            self.assertNotIn("STILL APPLIED", g)
        self.assertFalse([l for l in logs if "GROW" in l])

    def test_grow_keeps_tree_and_emits_extend_goal(self):
        goals, logs, reverts, _ = self.grow_walk(GROW_SEG="1")
        # Refusals 1 and 2 grow (tree kept); 3 and 4 hit the cap and revert.
        self.assertEqual(len(reverts), 2)
        self.assertIn("STILL APPLIED", goals[1])
        self.assertIn("EXTEND", goals[1])
        self.assertIn("STILL APPLIED", goals[2])
        # After the cap the goal must tell the truth again: tree reverted.
        self.assertNotIn("STILL APPLIED", goals[3])
        self.assertIn("reverted", goals[3])
        self.assertTrue([l for l in logs if "GROW 1/2" in l])
        self.assertTrue([l for l in logs if "GROW 2/2" in l])

    def test_oracle_accept_after_grow_ends_the_walk(self):
        goals, logs, reverts, reason = self.grow_walk(GROW_SEG="1",
                                                      accept_after=1,
                                                      grow_bytes=True)
        self.assertEqual(reason, "declared")
        self.assertEqual(len(goals), 2)   # grow segment ran, then accepted
        self.assertEqual(len(reverts), 0)

    def test_collateral_refusal_never_grows(self):
        """P2P regressed: growing a diff that broke neighbours compounds
        the damage -- that path must keep reverting even with GROW_SEG=1."""
        goals, logs, reverts, _ = self.grow_walk(
            GROW_SEG="1", last_tail="PASS_TO_PASS regressed: 2 test(s)")
        self.assertEqual(len(reverts), 4)
        for g in goals:
            self.assertNotIn("STILL APPLIED", g)
        self.assertFalse([l for l in logs if "GROW 1/2" in l])

    def test_echo_skips_probe_and_states_the_no_edit(self):
        """GROW_ECHO (cycle 4): sklearn-13497 banked 645/645/645 -- the grow
        segments made NO edit and each identical candidate cost a full
        hidden-test probe. Identical bytes get identical verdicts: one probe
        total, the fact stated in the next goal."""
        goals, logs, reverts, _ = self.grow_walk(GROW_SEG="1")
        self.assertEqual(self._probes["n"], 1)     # first green only
        self.assertTrue([l for l in logs if "GROW_ECHO" in l])
        self.assertIn("BYTE-IDENTICAL", goals[2])  # the fact, next goal
        self.assertIn("NEW edit", goals[2])

    def test_echo_off_when_grow_off(self):
        goals, logs, reverts, _ = self.grow_walk()   # GROW_SEG unset
        self.assertEqual(self._probes["n"], 4)       # every green probed
        self.assertFalse([l for l in logs if "GROW_ECHO" in l])


class CompactTest(unittest.TestCase):
    """SEG_COMPACT (2026-08-12): facts in, imitable history out.

    The walk threads the WHOLE transcript into every segment, so by segment 5
    the model sits on four verbatim failed attempts -- and
    repetition-is-conviction says a transcript full of an edit is a prompt to
    produce that edit again.  With SEG_COMPACT=1 each segment after the first
    starts from [system prompt] + a harness-built factual LEDGER (repro
    script + status, diagnosis record, banked candidates, current tree)
    prepended to the segment goal.  Targeted injections (SEG_ECHO, GROW,
    oracle hints) live in the goal already, so they survive.  Default off.
    """

    def compact_walk(self, segments=3, state=None, oracle=False, **env):
        """Segments return a NONEMPTY 3-message transcript (the real
        phase_run never returns []); with oracle=True every segment goes
        green and the probe refuses.  Returns (goals, logs, inits)."""
        goals, logs, inits = [], [], []
        st = state if state is not None else {"seen_red": False}

        def fake_phase_run(cpu, tools, tool2sys, handlers, system_prompt,
                           goal, turns, *a, **kw):
            goals.append(goal)
            inits.append(kw.get("init_messages"))
            transcript = [{"role": "system", "content": "SYS"},
                          {"role": "user", "content": goal},
                          {"role": "assistant", "content": "attempt text"}]
            if oracle:
                st["repro_green"] = True
                return "solved", transcript, {}
            return "budget", transcript, {}

        handlers = {
            "_diff_nonempty": lambda: True,
            "_capture_diff": lambda: SAME,
            "_revert_tree": lambda: None,
            "_restore_diff": lambda d: {"restored": True},
            "swe.verify_fix": lambda pcb, args: {"ok": False},
        }
        if oracle:
            handlers["_oracle_probe"] = lambda: False
            SA.oracle_probe.last_tail = ""
        real = SA.phase_run
        SA.phase_run = fake_phase_run
        try:
            base = dict(SEG_ECHO=None, REPERTOIRE_WALL="0", PHASE_WALL_CAP="0",
                        COVERAGE_GAP=None, SEG1_TURNS="1", GROW_SEG=None,
                        SEG_COMPACT=None, BANK_AUDIT=None,
                        ORACLE_GATE="1" if oracle else None)
            base.update(env)
            with _Env(**base):
                SA.repertoire_fix(cpu=None, tools=[], tool2sys={},
                                  handlers=handlers, system_prompt="s",
                                  goal="PLAIN GOAL", state=st, seg_turns=1,
                                  max_ops=segments, log=logs.append)
        finally:
            SA.phase_run = real
            if hasattr(SA.oracle_probe, "last_tail"):
                del SA.oracle_probe.last_tail
        return goals, logs, inits

    def test_off_by_default_threads_full_transcript(self):
        """Every behaviour change ships behind a flag, default off."""
        goals, logs, inits = self.compact_walk()
        self.assertIsNone(inits[0])              # segment 1: fresh build
        self.assertEqual(len(inits[1]), 3)       # segment 2: whole transcript
        for g in goals:
            self.assertNotIn("LEDGER", g)
        self.assertFalse([l for l in logs if "SEG_COMPACT" in l])

    def test_compact_replaces_transcript_with_system_plus_ledger(self):
        goals, logs, inits = self.compact_walk(SEG_COMPACT="1")
        self.assertIsNone(inits[0])              # segment 1 untouched
        self.assertEqual(len(inits[1]), 1)       # system message only
        self.assertEqual(inits[1][0]["role"], "system")
        self.assertTrue(goals[1].startswith("LEDGER"))
        self.assertTrue([l for l in logs if "SEG_COMPACT" in l])

    def test_ledger_carries_the_facts(self):
        st = {"seen_red": False, "repro_script": "import boom_marker",
              "repro_mode": "pytest",
              "diag": {"S1_reproduce": "done (red exit 1)"}}
        goals, logs, inits = self.compact_walk(state=st, SEG_COMPACT="1")
        led = goals[1]
        self.assertIn("import boom_marker", led)     # the registered repro
        self.assertIn("RED", led)                    # and its status
        self.assertIn("Diagnosis record", led)
        self.assertIn("bytes", led)                  # banked candidate line
        self.assertIn("APPLIED right now", led)      # _capture_diff nonempty

    def test_ledger_admits_missing_repro(self):
        goals, logs, inits = self.compact_walk(SEG_COMPACT="1")
        self.assertIn("No reproduction script", goals[1])

    def test_compact_and_grow_compose(self):
        """The grow goal is appended to seg_goal, so it must survive
        compaction: segment 2 states BOTH the ledger and STILL APPLIED."""
        goals, logs, inits = self.compact_walk(oracle=True, SEG_COMPACT="1",
                                               GROW_SEG="1")
        self.assertEqual(len(inits[1]), 1)
        self.assertTrue(goals[1].startswith("LEDGER"))
        self.assertIn("STILL APPLIED", goals[1])
        self.assertIn("APPLIED right now", goals[1])


if __name__ == "__main__":
    unittest.main()
