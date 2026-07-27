"""Unit tests for the live event bus (2026-07-25).

Locks in Phase B: make_emitter writes valid JSONL, fields are capped, emission
is best-effort (a bad path never raises), and phase_run actually emits a
generation + tool_call + tool_result for a turn. No LLM, no network, no GPU.

Run: cd ~/Code/LLMOS && python3 tests/test_events.py
"""
import inspect
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import swe_agent_v2 as SA


class FakeCPU:
    """Minimal stand-in for ToolCallCPU: one turn, one tool call, then done."""
    def __init__(self, tool="noop"):
        self.tool = tool

    def _chat(self, messages):
        msg = {"content": "I will inspect the file.",
               "reasoning_content": "thinking about it",
               "tool_calls": [{"function": {"name": self.tool, "arguments": {}}}]}
        meta = {"prompt_tokens": 11, "eval_tokens": 7}
        return msg, meta


class EventsBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.path = os.path.join(self.tmp, "sub", "events.jsonl")  # sub/ must be created
        self._old = SA.EVENTS_PATH
        SA.EVENTS_PATH = self.path
        SA._events_seq[0] = 0

    def tearDown(self):
        SA.EVENTS_PATH = self._old
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def lines(self):
        if not os.path.exists(self.path):
            return []
        return [json.loads(l) for l in open(self.path) if l.strip()]


class TestEmitter(EventsBase):
    def test_writes_valid_jsonl_with_context(self):
        emit = SA.make_emitter("django__django-1", "fix")
        emit("generation", {"turn": 0, "content": "hello"})
        emit("tool_result", {"turn": 0, "result": {"ok": True}})
        rows = self.lines()
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["type"], "generation")
        self.assertEqual(rows[0]["instance_id"], "django__django-1")
        self.assertEqual(rows[0]["phase"], "fix")
        self.assertIn("ts", rows[0])
        self.assertEqual(rows[0]["content"], "hello")

    def test_seq_increments_monotonically(self):
        emit = SA.make_emitter("i", "bootstrap")
        for i in range(5):
            emit("tool_call", {"turn": i})
        seqs = [r["seq"] for r in self.lines()]
        self.assertEqual(seqs, [1, 2, 3, 4, 5])

    def test_big_field_is_capped(self):
        emit = SA.make_emitter("i", "fix")
        emit("generation", {"turn": 0, "content": "x" * 100000})
        row = self.lines()[0]
        self.assertLess(len(row["content"]), 30000)     # bounded
        self.assertIn("[+", row["content"])             # marked as clipped
        self.assertTrue(row["content"].startswith("x"))

    def test_emit_never_raises_on_bad_path(self):
        SA.EVENTS_PATH = "/proc/cannot/make/this/events.jsonl"  # makedirs will fail
        emit = SA.make_emitter("i", "fix")
        try:
            emit("generation", {"turn": 0, "content": "hi"})  # must swallow
        except Exception as e:
            self.fail("emit raised on unwritable path: %r" % e)


class TestPhaseRunEmits(EventsBase):
    def test_signature_has_emit(self):
        self.assertIn("emit", inspect.signature(SA.phase_run).parameters)

    def test_phase_run_emits_generation_call_and_result(self):
        emit = SA.make_emitter("inst-x", "fix")
        reason, msgs, meta = SA.phase_run(
            FakeCPU(), tools=[], tool2sys={"noop": "noop"},
            handlers={"noop": lambda pcb, args: {"ok": True, "echo": args}},
            system_prompt="sys", user_goal="goal", budget=1, emit=emit)
        types = [r["type"] for r in self.lines()]
        self.assertIn("generation", types)
        self.assertIn("tool_call", types)
        self.assertIn("tool_result", types)
        # the tool_call names the dispatched handler ("function")
        tc = next(r for r in self.lines() if r["type"] == "tool_call")
        self.assertEqual(tc["tool"], "noop")
        self.assertEqual(tc["function"], "noop")

    def test_no_emit_arg_is_silent(self):
        # default emit=None -> no file, no error (backward compatible)
        reason, msgs, meta = SA.phase_run(
            FakeCPU(), tools=[], tool2sys={"noop": "noop"},
            handlers={"noop": lambda pcb, args: {"ok": True}},
            system_prompt="s", user_goal="g", budget=1)
        self.assertEqual(self.lines(), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
