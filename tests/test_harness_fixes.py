"""Unit tests for the trace-driven harness fixes (2026-07-25).

Covers:
  - _result_sig: same failing outcome -> same signature; distinct edits differ.
  - the no-progress watchdog: phase_run stops with 'stalled' when circling,
    runs to budget when producing novel results, and is off by default.
  - _missing_file_hint: reproduction/scratch paths get redirected, guessed
    paths point at locate(), an existing basename elsewhere is suggested.

No LLM, no network, no GPU. Budgets stay < 8 so the turn-7 critic never fires.
Run: cd ~/Code/LLMOS && python3 tests/test_harness_fixes.py
"""
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import swe_agent_v2 as SA
import swe_fix_tools as F


class Fake:
    """Yields one tool call to 'probe' every turn."""
    def _chat(self, messages):
        return ({"content": "", "tool_calls":
                 [{"function": {"name": "probe", "arguments": {}}}]},
                {"prompt_tokens": 1, "eval_tokens": 1})


def run_phase(result_fn, budget, window):
    calls = {"n": 0}

    def h(pcb, args):
        calls["n"] += 1
        return result_fn(calls["n"])

    return SA.phase_run(Fake(), tools=[], tool2sys={"probe": "probe"},
                        handlers={"probe": h}, system_prompt="s",
                        user_goal="g", budget=budget, stall_window=window)


class TestResultSig(unittest.TestCase):
    def test_same_failure_same_sig(self):
        r = {"ok": False, "error": "AssertionError: 1 != 2"}
        self.assertEqual(SA._result_sig("run_tests", dict(r)),
                         SA._result_sig("run_tests", dict(r)))

    def test_distinct_edits_differ(self):
        a = SA._result_sig("patch", {"edited": "x.py", "new_bytes": 10, "delta_bytes": 3})
        b = SA._result_sig("patch", {"edited": "x.py", "new_bytes": 20, "delta_bytes": 9})
        self.assertNotEqual(a, b)


class TestStallWatchdog(unittest.TestCase):
    def test_stops_when_circling(self):
        reason, msgs, meta = run_phase(
            lambda n: {"ok": False, "error": "same failing test"},
            budget=30, window=5)
        self.assertEqual(reason, "stalled")
        self.assertLess(len(meta), 30)   # stopped well before budget

    def test_runs_when_making_progress(self):
        reason, msgs, meta = run_phase(
            lambda n: {"ok": False, "exit": n, "stdout": "unique output %d" % n},
            budget=6, window=5)
        self.assertEqual(reason, "budget")   # novel every turn -> never stalls

    def test_off_by_default(self):
        reason, msgs, meta = run_phase(
            lambda n: {"ok": False, "error": "x"}, budget=6, window=None)
        self.assertEqual(reason, "budget")   # no window -> watchdog disabled


class TestMissingFileHint(unittest.TestCase):
    def setUp(self):
        self.repo = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.repo, "django", "template"))
        with open(os.path.join(self.repo, "django", "template",
                               "defaultfilters.py"), "w") as f:
            f.write("def join(): pass\n")

    def tearDown(self):
        shutil.rmtree(self.repo, ignore_errors=True)

    def test_reproduction_path_redirected(self):
        h = F._missing_file_hint(
            "/home/bard/swe/work/django__django-16910/_reproduction.py", self.repo)
        self.assertIn("run inline", h)
        self.assertIn("locate", h)
        self.assertNotIn("exists at", h)   # not treated as a findable repo file

    def test_guessed_path_points_at_locate(self):
        h = F._missing_file_hint("tests/model_fields/test_one_to_one.py", self.repo)
        self.assertIn("locate", h)

    def test_suggests_existing_basename(self):
        h = F._missing_file_hint("wrong/dir/defaultfilters.py", self.repo)
        self.assertIn("defaultfilters.py", h)
        self.assertIn("django/template", h)


if __name__ == "__main__":
    unittest.main(verbosity=2)
