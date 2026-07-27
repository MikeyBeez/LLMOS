"""Unit tests for the never-truncate / summarize / catalog context handling
(2026-07-25). The old phase_run hard-cut every result with [:4800] (a head cut),
throwing away the diagnosis that pytest/unittest put at the END. These lock in
the extractive summarizer, the recall tool, and that small fields survive.

Run: cd ~/Code/LLMOS && python3 tests/test_context_handling.py
No LLM, no network, no GPU.
"""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import swe_agent_v2 as SA
import repo_bootstrap_tools as R
import swe_fix_tools as F


def big_pytest():
    """A realistic test run: lots of PASSED noise, one FAILED with the
    assertion diff and summary line at the very end (the 'wrong end')."""
    lines = ["============ test session starts ============"]
    lines += ["test_mod.py::test_%d PASSED" % i for i in range(60)]
    lines += ["test_mod.py::test_bug FAILED",
              "    def test_bug():",
              ">       assert compute() == 42",
              "E       AssertionError: assert 7 == 42",
              "test_mod.py:123: AssertionError"]
    lines += ["test_mod.py::test_%d PASSED" % i for i in range(60, 110)]
    lines += ["==== 1 failed, 110 passed in 3.2s ===="]
    return "\n".join(lines)


class TestExtract(unittest.TestCase):
    def test_keeps_buried_error(self):
        s = SA._extract_important(big_pytest(), 600)
        self.assertLess(len(s), len(big_pytest()))
        self.assertIn("AssertionError: assert 7 == 42", s)
        self.assertIn("FAILED", s)

    def test_keeps_summary_line_at_the_end(self):
        # the whole point: the reason lives at the END, must survive
        s = SA._extract_important(big_pytest(), 600)
        self.assertIn("1 failed, 110 passed", s)

    def test_drops_middle_noise(self):
        s = SA._extract_important(big_pytest(), 600)
        self.assertNotIn("test_30 PASSED", s)   # a middle passing line is dropped

    def test_short_passthrough(self):
        self.assertEqual(SA._extract_important("short text", 600), "short text")


class TestSmartSummarize(unittest.TestCase):
    def test_small_result_passes_through(self):
        r = {"ok": True, "exit": 0}
        self.assertEqual(SA.smart_summarize(r, 4800, "out1"),
                         json.dumps(r, default=str))

    def test_big_result_keeps_error_and_names_recall(self):
        r = {"ok": False, "exit": 1, "test_tail": big_pytest(),
             "hint": "look at the assertion"}
        out = SA.smart_summarize(r, 1200, "out7")
        self.assertLess(len(out), len(json.dumps(r, default=str)))
        self.assertIn("AssertionError", out)       # diagnosis survives
        self.assertIn("out7", out)                 # recall id is discoverable
        self.assertIn("recall", out)
        self.assertIn("look at the assertion", out)  # small field untouched

    def test_small_fields_never_dropped(self):
        r = {"ok": False, "exit": 1, "given_tests_ok": False,
             "stdout": "boilerplate\n" * 800}
        out = SA.smart_summarize(r, 1000, "out2")
        d = json.loads(out)
        self.assertEqual(d.get("exit"), 1)                 # scalar preserved
        self.assertEqual(d.get("given_tests_ok"), False)   # scalar preserved
        self.assertEqual(d.get("_recall"), "out2")


class TestRecallTool(unittest.TestCase):
    def test_recall_present_in_both_phases(self):
        self.assertTrue(any(t["function"]["name"] == "recall"
                            for t in R.BOOTSTRAP_TOOLS))
        self.assertTrue(any(t["function"]["name"] == "recall"
                            for t in F.FIX_TOOLS))

    def test_recall_requires_ref(self):
        tool = next(t["function"] for t in F.FIX_TOOLS
                    if t["function"]["name"] == "recall")
        self.assertIn("ref", tool["parameters"].get("required", []))


if __name__ == "__main__":
    unittest.main(verbosity=2)
