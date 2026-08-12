"""DIAG_GATE tests (2026-08-11).

Mikey: "can we make these steps required in the program? ... a general
algorithm that will either try each step or say it's unnecessary. And the
results should be kept as state for the example."

The ladder: S1 reproduce red -> S2 differential -> S3 declare the site ->
edit.  Every step ends in exactly one recorded outcome in state["diag"]
(done / waived-with-reason / skipped-after-warning), the worksheet renders
the record every turn, and each refusal is one-shot so the ladder cannot
deadlock.

Run: cd ~/Code/LLMOS && python3 -m pytest tests/test_diag_gate.py -q
"""
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import swe_fix_tools as F

SRC = "def writer(x):\n    return x + 1\n\ndef reader(x):\n    return writer(x)\n"


class Base(unittest.TestCase):

    def setUp(self):
        self.repo = tempfile.mkdtemp()
        with open(os.path.join(self.repo, "mod.py"), "w") as fh:
            fh.write(SRC)
        with open(os.path.join(self.repo, "other.py"), "w") as fh:
            fh.write("y = 1\n")
        self._old = os.environ.get("DIAG_GATE")
        os.environ["DIAG_GATE"] = "1"
        self.handlers, self.state = F.make_fix_handlers(self.repo)

    def tearDown(self):
        if self._old is None:
            os.environ.pop("DIAG_GATE", None)
        else:
            os.environ["DIAG_GATE"] = self._old
        shutil.rmtree(self.repo, ignore_errors=True)

    def patch(self, path="mod.py"):
        return self.handlers["swe.patch"](None, {
            "file": path, "old_snippet": "return x + 1",
            "new_snippet": "return x + 2"})


class LadderTest(Base):

    def test_off_by_default_no_gate(self):
        os.environ.pop("DIAG_GATE", None)
        res = self.patch()
        self.assertNotIn("DIAGNOSIS", str(res.get("error", "")))

    def test_every_step_ends_recorded(self):
        # S1 challenge
        r1 = self.patch()
        self.assertIn("DIAGNOSIS 1/3", r1["error"])
        # proceed anyway -> S1 recorded skipped, S2 challenge
        r2 = self.patch()
        self.assertIn("DIAGNOSIS 2/3", r2["error"])
        self.assertEqual(self.state["diag"]["S1_reproduce"],
                         "skipped after warning")
        # proceed anyway -> S2 recorded skipped, S3 challenge
        r3 = self.patch()
        self.assertIn("DIAGNOSIS 3/3", r3["error"])
        self.assertEqual(self.state["diag"]["S2_differential"],
                         "skipped after warning")
        # declare the site -> S3 done, patch flows through to real edit
        d = self.handlers["swe.declare_site"](None, {
            "file": "mod.py", "function": "writer", "role": "writer",
            "reason": "writer of the state"})
        self.assertIn("recorded", d)
        r4 = self.patch()
        self.assertNotIn("DIAGNOSIS", str(r4.get("error", "")))
        self.assertTrue(r4.get("edited"))
        # the whole ladder is state, one recorded outcome per step
        self.assertEqual(sorted(self.state["diag"]),
                         ["S1_reproduce", "S2_differential", "S3_site"])

    def test_waivers_take_precedence_over_challenges(self):
        self.state["repro_attempts"] = 2          # tried twice, no red
        self.state["fault_seen"] = True           # crash frames seen
        r = self.patch()
        self.assertIn("DIAGNOSIS 3/3", r["error"])   # straight to S3
        self.assertIn("waived", self.state["diag"]["S1_reproduce"])
        self.assertIn("waived", self.state["diag"]["S2_differential"])

    def test_red_satisfies_s1_silently(self):
        self.state["seen_red"] = True
        self.state["fault_seen"] = True
        r = self.patch()
        self.assertIn("DIAGNOSIS 3/3", r["error"])
        self.assertIn("done", self.state["diag"]["S1_reproduce"])

    def test_site_mismatch_challenged_once(self):
        self.state["seen_red"] = True
        self.state["fault_seen"] = True
        self.patch()                               # S3 challenge consumed
        self.handlers["swe.declare_site"](None, {
            "file": "mod.py", "reason": "the writer lives here"})
        r1 = self.patch("other.py")                # off-site: challenged
        self.assertIn("declared fix site", r1["error"])
        r2 = self.patch("other.py")                # insists: allowed
        self.assertNotIn("declared fix site", str(r2.get("error", "")))

    def test_worksheet_renders_the_ladder(self):
        self.state["repro_attempts"] = 2
        self.state["fault_seen"] = True
        self.patch()
        ws = F.render_worksheet(self.state)
        self.assertIn("diagnosis", ws)
        self.assertIn("waived", ws)


class DeclareSiteTest(Base):

    def test_unknown_function_is_loud_noop_with_defs(self):
        r = self.handlers["swe.declare_site"](None, {
            "file": "mod.py", "function": "nope", "reason": "x"})
        self.assertIn("error", r)
        self.assertIn("writer", r["definitions_in_file"])
        self.assertNotIn("S3_site", self.state.get("diag", {}))

    def test_reader_role_draws_the_caution(self):
        r = self.handlers["swe.declare_site"](None, {
            "file": "mod.py", "function": "reader", "role": "reader",
            "reason": "symptom appears here"})
        self.assertIn("caution", r)


class DifferentialValidationTest(Base):

    def test_both_scripts_required(self):
        r = self.handlers["swe.differential"](None, {"bug_script": "x"})
        self.assertIn("BOTH", r["error"])

    def test_identical_scripts_rejected(self):
        r = self.handlers["swe.differential"](None, {
            "bug_script": "print(1)", "control_script": "print(1)"})
        self.assertIn("identical", r["error"])


if __name__ == "__main__":
    unittest.main()
