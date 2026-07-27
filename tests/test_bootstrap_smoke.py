"""Unit tests for the phase-1 bootstrap harness fixes (2026-07-25).

First unit tests the SWE harness has ever had. The existing tests/ suite
covers the LLMOS kernel (MockCPU); nothing covered repo_bootstrap_tools,
which is where three models burned their phase-1 budgets this week.

Covers:
  - _clip keeps head AND tail (tail-only truncation hid tracebacks)
  - _bad_test_id_hint fires on hallucinated ids, stays silent on real failures
  - run_smoke_test schema: test_id optional -> auto_verify_env reachable
  - new phase-1 locate/read_range tools (registration + behavior + traversal guard)
  - _llm_content never returns the think stream as the answer
  - critic_review no longer uses the 600-token budget that leaked reasoning

Run: cd ~/Code/LLMOS && python3 tests/test_bootstrap_smoke.py
No LLM, no network, no GPU.
"""
import inspect
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import repo_bootstrap_tools as R

# Verbatim capture from the real ornith run on django__django-15814,
# 2026-07-25 — the failure mode that started all of this.
DJANGO_FAIL = """E
======================================================================
ERROR: test_proxy_model_fields (unittest.loader._FailedTest)
----------------------------------------------------------------------
AttributeError: type object 'ProxyModelTests' has no attribute 'test_proxy_model_fields'

Ran 1 test in 0.000s
FAILED (errors=1)
System check identified no issues (0 silenced)."""


class TestClip(unittest.TestCase):
    def test_keeps_head_and_tail(self):
        s = "TRACEBACK AT TOP\n" + "x" * 9000 + "\nFOOTER AT BOTTOM"
        c = R._clip(s)
        self.assertIn("TRACEBACK AT TOP", c)
        self.assertIn("FOOTER AT BOTTOM", c)
        self.assertLess(len(c), len(s))

    def test_short_passthrough(self):
        self.assertEqual(R._clip("abc"), "abc")

    def test_none_is_empty(self):
        self.assertEqual(R._clip(None), "")


class TestBadTestIdHint(unittest.TestCase):
    def setUp(self):
        self.repo = tempfile.mkdtemp()
        d = os.path.join(self.repo, "tests", "proxy_models")
        os.makedirs(d)
        with open(os.path.join(d, "tests.py"), "w") as f:
            f.write("class ProxyModelTests(TestCase):\n"
                    "    def test_basic_proxy(self):\n        pass\n"
                    "    def test_no_proxy(self):\n        pass\n")

    def tearDown(self):
        shutil.rmtree(self.repo, ignore_errors=True)

    def test_fires_and_lists_real_methods(self):
        h = R._bad_test_id_hint(DJANGO_FAIL, "", "whatever", self.repo)
        self.assertIsNotNone(h)
        self.assertIn("TEST DOES NOT EXIST", h)
        self.assertIn("test_basic_proxy", h)
        self.assertIn("test_no_proxy", h)

    def test_silent_on_genuine_test_failure(self):
        out = "FAILED tests/test_x.py::test_y - AssertionError: 1 != 2"
        self.assertIsNone(
            R._bad_test_id_hint(out, "", "tests/test_x.py::test_y", self.repo))


class TestSmokeSchema(unittest.TestCase):
    def _smoke(self):
        return next(t["function"] for t in R.BOOTSTRAP_TOOLS
                    if t["function"]["name"] == "run_smoke_test")

    def test_test_id_not_required(self):
        # required=[test_id] made the auto_verify_env no-arg path unreachable;
        # three models hallucinated test names to satisfy this schema.
        self.assertNotIn("test_id",
                         self._smoke()["parameters"].get("required", []))

    def test_description_advertises_auto_mode(self):
        self.assertIn("NO", self._smoke()["description"])


class TestPhase1FileTools(unittest.TestCase):
    def setUp(self):
        self.repo = tempfile.mkdtemp()
        with open(os.path.join(self.repo, "mod.py"), "w") as f:
            f.write("line one\ndef target_function():\n    return 42\n")
        self.handlers, self.state = R.make_bootstrap_handlers(self.repo)

    def tearDown(self):
        shutil.rmtree(self.repo, ignore_errors=True)

    def test_registered_in_schema_and_dispatch(self):
        names = [t["function"]["name"] for t in R.BOOTSTRAP_TOOLS]
        self.assertIn("locate", names)
        self.assertIn("read_range", names)
        self.assertEqual(R.BOOTSTRAP_TOOL2SYS["locate"], "repo.locate")
        self.assertEqual(R.BOOTSTRAP_TOOL2SYS["read_range"], "repo.read_range")
        self.assertIn("repo.locate", self.handlers)
        self.assertIn("repo.read_range", self.handlers)

    def test_locate_finds_pattern(self):
        r = self.handlers["repo.locate"](None, {"pattern": "target_function"})
        self.assertEqual(r["match_count"], 1)
        self.assertTrue(r["matches"][0].startswith("mod.py:2:"))

    def test_locate_bad_regex_is_soft_error(self):
        r = self.handlers["repo.locate"](None, {"pattern": "(["})
        self.assertIn("error", r)
        self.assertEqual(r["match_count"], 0)

    def test_read_range(self):
        r = self.handlers["repo.read_range"](
            None, {"file": "mod.py", "start": 2, "end": 3})
        self.assertIn("target_function", r["content"])
        self.assertEqual(r["total_lines"], 3)

    def test_read_range_blocks_traversal(self):
        r = self.handlers["repo.read_range"](None, {"file": "../etc/passwd"})
        self.assertIn("error", r)


class TestLlmContent(unittest.TestCase):
    def test_normal_content(self):
        self.assertEqual(R._llm_content({"content": "the answer"}),
                         "the answer")

    def test_never_returns_think_stream(self):
        # The critic's "advice" was the model's own half-finished reasoning
        # because llm_call fell back to reasoning_content.
        m = {"content": "",
             "reasoning_content":
                 "The user wants me to act as a detached reviewer."}
        self.assertEqual(R._llm_content(m), "")

    def test_critic_budget_regression_guard(self):
        import trace_consumers
        src = inspect.getsource(trace_consumers.critic_review)
        self.assertNotIn(
            "max_tokens=600", src,
            "critic_review's 600-token budget reintroduces the think-stream "
            "leak that llm_call's 1600 default exists to prevent")


if __name__ == "__main__":
    unittest.main(verbosity=2)
