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
    seq = {"n": 0}

    def fake_phase_run(cpu, tools, tool2sys, handlers, system_prompt, goal,
                       turns, *a, **kw):
        goals.append(goal)
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


SAME = "--- a/x.py\n+++ b/x.py\n@@\n-a\n+b\n"
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


if __name__ == "__main__":
    unittest.main()
