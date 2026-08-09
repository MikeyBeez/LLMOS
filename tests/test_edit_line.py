"""edit_line and SWEBENCH_MODE sibling-scan tests (2026-08-08).

These cover the two mechanisms added on 2026-08-08 that had only ad-hoc
checks:

  edit_line -- the LOUD NO-OP contract.  A fragment that is missing, or that
  appears twice on the named line, or that would break the parse, must write
  NOTHING and say so.  The whole point of the tool is that wrong input cannot
  corrupt the file, so the no-op path is the part worth protecting.  It also
  must not touch the line's indentation or its untouched remainder, which is
  what xarray-5131 died of.

  _sibling_sites -- must not hand back the line it was just told about.  The
  first version excluded only the OLD fragment, so the line the model had just
  edited came back top-ranked as its own sibling.  Caught end-to-end; locked
  here.

  _file_defs -- the different-names half.  A token scan cannot connect
  _print_sinc to _print_sinh, so the harness lists every definition and the
  model picks the family.  Both are gated on SWEBENCH_MODE.

No LLM, no network, no GPU.  Uses a throwaway directory as the repo.

Run: cd ~/Code/LLMOS && python3 -m pytest tests/test_edit_line.py -q
"""
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import swe_fix_tools as F

SRC = (
    "class Printer(object):\n"
    "    def _print_sinc(self, expr):\n"
    "        if expr.args[0] > 0:\n"
    "            return 1\n"
    "        return 0\n"
    "\n"
    "    def _print_sinh(self, expr):\n"
    "        if expr.args[0] > 0:\n"
    "            return 1\n"
    "        return 0\n"
)


class Base(unittest.TestCase):

    def setUp(self):
        self.repo = tempfile.mkdtemp()
        self.path = "printer.py"
        with open(os.path.join(self.repo, self.path), "w") as fh:
            fh.write(SRC)
        self._old_env = {k: os.environ.get(k)
                         for k in ("SWEBENCH_MODE", "EDIT_LINE")}
        self.handlers, self.state = F.make_fix_handlers(self.repo)
        self.edit = self.handlers["swe.edit_line"]

    def tearDown(self):
        for k, v in self._old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(self.repo, ignore_errors=True)

    def disk(self):
        with open(os.path.join(self.repo, self.path)) as fh:
            return fh.read()


class EditLineNoOpTest(Base):

    def test_missing_fragment_writes_nothing_and_shows_the_line(self):
        before = self.disk()
        res = self.edit(None, {"file": self.path, "line": 3,
                               "old": "not-in-this-line", "new": "x"})
        self.assertIn("error", res)
        self.assertIn("fragment not found", res["error"])
        self.assertIn("line_is", res)
        self.assertIn("expr.args[0]", res["line_is"])
        self.assertEqual(self.disk(), before)

    def test_repeated_fragment_writes_nothing(self):
        before = self.disk()
        res = self.edit(None, {"file": self.path, "line": 1,
                               "old": "r", "new": "R"})
        self.assertIn("error", res)
        self.assertIn("occurs", res["error"])
        self.assertEqual(self.disk(), before)

    def test_line_out_of_range_writes_nothing(self):
        before = self.disk()
        res = self.edit(None, {"file": self.path, "line": 999,
                               "old": "return", "new": "pass"})
        self.assertIn("error", res)
        self.assertEqual(self.disk(), before)

    def test_syntax_breaking_edit_is_refused_before_the_write(self):
        """Compiled in memory first: refused, not applied-then-reverted."""
        before = self.disk()
        res = self.edit(None, {"file": self.path, "line": 3,
                               "old": "if ", "new": "if if "})
        self.assertIn("error", res)
        self.assertIn("parsing", res["error"])
        self.assertIn("would_have_been", res)
        self.assertEqual(self.disk(), before)

    def test_test_files_are_refused(self):
        tp = "tests/test_printer.py"
        os.makedirs(os.path.join(self.repo, "tests"))
        with open(os.path.join(self.repo, tp), "w") as fh:
            fh.write("x = 1\n")
        res = self.edit(None, {"file": tp, "line": 1, "old": "1", "new": "2"})
        self.assertIn("error", res)
        self.assertIn("test file", res["error"])


class EditLineWriteTest(Base):

    def test_indentation_and_remainder_survive(self):
        res = self.edit(None, {"file": self.path, "line": 3,
                               "old": "> 0", "new": ">= 0"})
        self.assertNotIn("error", res)
        self.assertEqual(res["edited_line"], 3)
        line = self.disk().splitlines()[2]
        self.assertEqual(line, "        if expr.args[0] >= 0:")
        self.assertTrue(self.disk().endswith("return 0\n"))
        self.assertEqual(len(self.disk().splitlines()),
                         len(SRC.splitlines()))

    def test_before_and_after_are_reported(self):
        res = self.edit(None, {"file": self.path, "line": 4,
                               "old": "return 1", "new": "return 2"})
        self.assertNotIn("error", res)
        self.assertIn("return 1", res["before"])
        self.assertIn("return 2", res["after"])

    def test_empty_new_deletes_the_fragment_only(self):
        res = self.edit(None, {"file": self.path, "line": 3,
                               "old": " > 0", "new": ""})
        self.assertNotIn("error", res)
        self.assertEqual(self.disk().splitlines()[2],
                         "        if expr.args[0]:")


class SiblingScanTest(Base):

    def sib(self):
        return self.state["_sibling_fn"](self.state.get("last_edit_file") or "",
                                         self.state.get("last_edit_frag") or "")

    def test_gated_off_by_default(self):
        os.environ.pop("SWEBENCH_MODE", None)
        self.edit(None, {"file": self.path, "line": 3,
                         "old": "> 0", "new": ">= 0"})
        self.assertEqual(self.sib(), [])
        self.assertEqual(self.state["_defs_fn"](self.path), [])

    def test_edited_line_is_not_its_own_sibling(self):
        """The NEW text is excluded, so the edited site is not its own kin."""
        os.environ["SWEBENCH_MODE"] = "1"
        self.edit(None, {"file": self.path, "line": 3,
                         "old": "> 0", "new": ">= 0"})
        lines = [n for n, _t in self.sib()]
        self.assertNotIn(3, lines)

    def test_the_real_sibling_line_is_offered(self):
        os.environ["SWEBENCH_MODE"] = "1"
        self.edit(None, {"file": self.path, "line": 3,
                         "old": "> 0", "new": ">= 0"})
        lines = [n for n, _t in self.sib()]
        self.assertIn(8, lines)          # the same guard inside _print_sinh

    def test_defs_list_names_the_family(self):
        """A token scan cannot link _print_sinc to _print_sinh; the list can."""
        os.environ["SWEBENCH_MODE"] = "1"
        names = [nm for _n, nm in self.state["_defs_fn"](self.path)]
        self.assertIn("_print_sinc", names)
        self.assertIn("_print_sinh", names)
        self.assertIn("Printer", names)


class SnippetModeSiblingTest(Base):
    """The suppression bug found while writing these tests (2026-08-08).

    h_patch put BOTH sides of the edit into the exclusion set.  Excluding the
    NEW text is what keeps the just-edited line out of its own sibling list.
    Excluding the OLD text as well looks symmetric and is actively harmful: a
    genuine sibling is usually a line that reads EXACTLY like the one just
    fixed -- the same guard in the next operator dunder, the same comparison
    in the next _print_* method -- so it suppressed precisely what the scan
    exists to surface.  edit_line goes through line mode with no old_snippet
    and never hit it; snippet mode did.
    """

    def sib(self):
        return self.state["_sibling_fn"](self.state.get("last_edit_file") or "",
                                         self.state.get("last_edit_frag") or "")

    def test_identical_sibling_survives_a_snippet_edit(self):
        os.environ["SWEBENCH_MODE"] = "1"
        res = self.handlers["swe.patch"](None, {
            "file": self.path,
            "old_snippet": "    def _print_sinc(self, expr):\n"
                           "        if expr.args[0] > 0:",
            "new_snippet": "    def _print_sinc(self, expr):\n"
                           "        if expr.args[0] >= 0:"})
        self.assertNotIn("error", res)
        lines = [n for n, _t in self.sib()]
        self.assertIn(8, lines)       # _print_sinh still carries the old guard
        self.assertNotIn(3, lines)    # and the edited line is not its own kin


if __name__ == "__main__":
    unittest.main()
