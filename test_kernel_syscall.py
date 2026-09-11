"""Tests for the capability gate. Run: python3 test_kernel_syscall.py

The adversarial cases are the point. A gate that only passes the cases its
author thought of is the gate DeepSeek shipped: it governs the filesystem and
not the way out of it.
"""

import json
import os
import shutil
import tempfile
import unittest

import kernel_syscall as K
from kernel_syscall import Audit, Capabilities, Gate, check


class Allow(unittest.TestCase):

    def test_tool_not_granted_is_denied(self):
        caps = Capabilities(allow={"swe.read_file"}, label="reader")
        v = check("swe.write_file", {}, caps)
        self.assertFalse(v.allowed)
        self.assertEqual(v.rule, "not-in-allow-list")

    def test_granted_tool_is_allowed(self):
        caps = Capabilities(allow={"swe.read_file"}, label="reader")
        self.assertTrue(check("swe.read_file", {}, caps).allowed)

    def test_prefix_pattern(self):
        caps = Capabilities(allow={"swe.*"}, label="swe")
        self.assertTrue(check("swe.anything", {}, caps).allowed)
        self.assertFalse(check("shell.exec", {}, caps).allowed)

    def test_empty_set_grants_nothing(self):
        self.assertFalse(check("swe.read_file", {}, K.NOTHING).allowed)

    def test_bare_star_is_refused_at_construction(self):
        # The set that would have allowed the wiki.
        with self.assertRaises(ValueError):
            Capabilities(allow={"*"})

    def test_relative_write_root_refused(self):
        with self.assertRaises(ValueError):
            Capabilities(allow={"x"}, write_roots=("./checkout",))

    def test_needs_human_beats_allow(self):
        caps = Capabilities(allow={"deploy"}, needs_human={"deploy"}, label="ci")
        v = check("deploy", {}, caps)
        self.assertFalse(v.allowed)
        self.assertEqual(v.rule, "needs-human")

    def test_empty_tool_name(self):
        self.assertFalse(check("", {}, Capabilities(allow={"x"})).allowed)


class WriteRoots(unittest.TestCase):
    """The mutator list is derived from the handlers; forced here so the
    path rules are tested without importing the 4,856-line tool module."""

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.outside = tempfile.mkdtemp()
        self._saved = K._MUTATORS
        K._MUTATORS = frozenset({"swe.patch"})
        self.caps = Capabilities(allow={"swe.patch", "swe.read_file"},
                                 write_roots=(self.root,), label="task")

    def tearDown(self):
        K._MUTATORS = self._saved
        shutil.rmtree(self.root, ignore_errors=True)
        shutil.rmtree(self.outside, ignore_errors=True)

    def test_write_inside_root_allowed(self):
        p = os.path.join(self.root, "pkg", "mod.py")
        self.assertTrue(check("swe.patch", {"path": p}, self.caps).allowed)

    def test_write_outside_root_denied(self):
        p = os.path.join(self.outside, "mod.py")
        v = check("swe.patch", {"path": p}, self.caps)
        self.assertFalse(v.allowed)
        self.assertEqual(v.rule, "outside-write-root")

    def test_dotdot_traversal_denied(self):
        p = os.path.join(self.root, "..", "..", "etc", "passwd")
        v = check("swe.patch", {"path": p}, self.caps)
        self.assertFalse(v.allowed)
        self.assertEqual(v.rule, "outside-write-root")

    def test_symlink_escape_denied(self):
        # The classic. A link inside the checkout pointing out of it; the
        # string is under the root and the file is not.
        link = os.path.join(self.root, "escape")
        os.symlink(self.outside, link)
        v = check("swe.patch", {"path": os.path.join(link, "mod.py")}, self.caps)
        self.assertFalse(v.allowed)
        self.assertEqual(v.rule, "outside-write-root")

    def test_home_expansion_is_not_a_hole(self):
        target = os.path.expanduser("~/.ssh/authorized_keys")
        self.assertFalse(check("swe.patch", {"path": target}, self.caps).allowed)

    def test_list_of_paths_all_checked(self):
        good = os.path.join(self.root, "a.py")
        bad = os.path.join(self.outside, "b.py")
        self.assertTrue(check("swe.patch", {"files": [good]}, self.caps).allowed)
        self.assertFalse(check("swe.patch", {"files": [good, bad]}, self.caps).allowed)

    def test_non_mutating_tool_not_path_checked(self):
        # read_file is not in the mutator set, so a read outside the root is
        # allowed by THIS gate. Read confinement is a separate capability and
        # pretending otherwise here would be security theatre.
        p = os.path.join(self.outside, "mod.py")
        self.assertTrue(check("swe.read_file", {"path": p}, self.caps).allowed)

    def test_mutator_with_no_write_root_denied(self):
        caps = Capabilities(allow={"swe.patch"}, label="rootless")
        v = check("swe.patch", {"path": "/tmp/x"}, caps)
        self.assertFalse(v.allowed)
        self.assertEqual(v.rule, "no-write-root")

    def test_with_write_root_confines_a_shared_set(self):
        base = Capabilities(allow={"swe.patch"}, label="base")
        caps = base.with_write_root(self.root)
        p = os.path.join(self.root, "a")
        self.assertTrue(check("swe.patch", {"path": p}, caps).allowed)


class Modes(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.log = os.path.join(self.tmp, "syscalls.jsonl")
        self.caps = Capabilities(allow={"ok"}, label="t")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def lines(self):
        with open(self.log) as fh:
            return [json.loads(l) for l in fh if l.strip()]

    def test_warn_records_but_allows(self):
        g = Gate(self.caps, Audit(self.log), mode_="warn")
        v = g.admit("forbidden", {}, turn=3)
        self.assertTrue(v.allowed)               # not blocked
        self.assertEqual(v.rule, "warn-only")
        self.assertEqual(g.denied, [("forbidden", "not-in-allow-list")])
        rec = self.lines()[0]
        self.assertFalse(rec["allowed"])         # but recorded as a denial
        self.assertFalse(rec["enforced"])
        self.assertEqual(rec["turn"], 3)

    def test_enforce_blocks(self):
        g = Gate(self.caps, Audit(self.log), mode_="enforce")
        v = g.admit("forbidden", {})
        self.assertFalse(v.allowed)
        self.assertTrue(self.lines()[0]["enforced"])

    def test_off_writes_nothing(self):
        g = Gate(self.caps, Audit(self.log), mode_="off")
        self.assertTrue(g.admit("forbidden", {}).allowed)
        self.assertFalse(os.path.exists(self.log))

    def test_allowed_calls_are_taped_too(self):
        g = Gate(self.caps, Audit(self.log), mode_="enforce")
        g.admit("ok", {"a": 1})
        rec = self.lines()[0]
        self.assertTrue(rec["allowed"])
        self.assertEqual(rec["tool"], "ok")
        self.assertEqual(rec["args"], {"a": 1})

    def test_tape_failure_does_not_break_the_agent(self):
        g = Gate(self.caps, Audit("/proc/nonexistent/nope.jsonl"), mode_="enforce")
        self.assertTrue(g.admit("ok", {}).allowed)   # no raise

    def test_denial_body_is_actionable_json(self):
        g = Gate(self.caps, Audit(self.log), mode_="enforce")
        body = json.loads(g.admit("forbidden", {}).as_tool_error())
        self.assertIn("NOTHING was executed", body["error"])
        self.assertIn("retrying the same call", body["hint"])


class BadArgs(unittest.TestCase):
    """The gate runs before argument validation, so it sees whatever the
    model emitted, including things that are not dicts."""

    def test_string_args_do_not_raise(self):
        caps = Capabilities(allow={"x"}, label="t")
        self.assertTrue(check("x", "not-a-dict", caps).allowed)

    def test_none_args_do_not_raise(self):
        caps = Capabilities(allow={"x"}, label="t")
        self.assertTrue(check("x", None, caps).allowed)


class FromTools(unittest.TestCase):
    """The allow-list is derived from the phase's own tool schemas."""

    SCHEMA = [
        {"type": "function", "function": {"name": "patch", "parameters": {}}},
        {"type": "function", "function": {"name": "read_file"}},
    ]

    def test_names_extracted(self):
        caps = Capabilities.from_tools(self.SCHEMA, label="fix")
        self.assertEqual(set(caps.allow), {"patch", "read_file"})

    def test_tool_from_another_phase_denied(self):
        caps = Capabilities.from_tools(self.SCHEMA, label="fix")
        v = check("bootstrap.install", {}, caps)
        self.assertFalse(v.allowed)
        self.assertEqual(v.rule, "not-in-allow-list")

    def test_hallucinated_name_denied(self):
        caps = Capabilities.from_tools(self.SCHEMA, label="fix")
        self.assertFalse(check("write_file", {}, caps).allowed)

    def test_empty_schema_refused(self):
        with self.assertRaises(ValueError):
            Capabilities.from_tools([], label="empty")

    def test_write_root_carried(self):
        caps = Capabilities.from_tools(self.SCHEMA, write_roots=("/tmp/co",))
        self.assertEqual(caps.write_roots, (os.path.realpath("/tmp/co"),))


if __name__ == "__main__":
    unittest.main(verbosity=2)
