"""Tests for the memory write gate. Run: python3 test_memory_surprise.py

The probe talks to a model, so every test here injects a scripted `ask`
instead. What is being tested is the DECISION, not the server.
"""

import unittest

import memory_surprise as M
from memory_surprise import normalise, probe, same_answer


def scripted(answers):
    """An `ask` that walks a fixed list, repeating the last one forever."""
    box = {"i": 0}

    def ask(_q):
        i = min(box["i"], len(answers) - 1)
        box["i"] += 1
        a = answers[i]
        if isinstance(a, Exception):
            raise a
        return a
    return ask


class Normalise(unittest.TestCase):

    def test_strips_markdown_and_case(self):
        # normalise does NOT try to pull a word answer out of a sentence --
        # that is the ambiguity rabbit hole. It lowercases, drops markdown
        # and trailing punctuation; same_answer's substring rule is what
        # matches a decorated reply to the bare value.
        self.assertEqual(normalise("The capital is **Paris**."),
                         "the capital is paris")
        self.assertTrue(same_answer("The capital is **Paris**.", "Paris"))

    def test_pulls_the_number_out(self):
        self.assertEqual(normalise("about 360 degrees"), "360")

    def test_thousands_separators(self):
        self.assertEqual(normalise("299,792,458"), "299792458")
        self.assertEqual(normalise("299792458"), "299792458")

    def test_none_and_empty(self):
        self.assertEqual(normalise(None), "")
        self.assertEqual(normalise("   "), "")


class SameAnswer(unittest.TestCase):

    def test_exact(self):
        self.assertTrue(same_answer("42", "42"))

    def test_near_number_is_the_same_answer(self):
        # 117 vs 117.3 is rounding, not a knowledge gap.
        self.assertTrue(same_answer("117", "117.3"))

    def test_far_number_is_not(self):
        self.assertFalse(same_answer("42", "28"))

    def test_decorated_number(self):
        self.assertTrue(same_answer("The answer is **42**.", "42"))

    def test_substring_words(self):
        self.assertTrue(same_answer("Paris, France", "Paris"))

    def test_short_substring_does_not_count(self):
        # "a" inside "apple" must not read as agreement.
        self.assertFalse(same_answer("a", "apple"))

    def test_zero_scale_does_not_divide_by_zero(self):
        self.assertTrue(same_answer("0", "0"))
        self.assertFalse(same_answer("0", "5"))


class Verdicts(unittest.TestCase):

    def test_known_fact_is_skipped(self):
        r = probe("What is the capital of France?", "Paris",
                  n=8, ask=scripted(["Paris"]))
        self.assertEqual(r["verdict"], "skip")
        self.assertEqual(r["reason"], "already-in-the-weights")
        self.assertEqual(r["hit_rate"], 1.0)

    def test_confidently_wrong_is_stored(self):
        # The seagull case: the model reliably says 28, the truth is 42.
        r = probe("What is the optimal number of seagulls?", "42",
                  n=8, ask=scripted(["28"]))
        self.assertEqual(r["verdict"], "store")
        self.assertEqual(r["reason"], "model-is-confidently-wrong")
        self.assertEqual(r["mode"], "28")

    def test_scattered_is_stored(self):
        r = probe("What is the best programming language?", "Rust",
                  n=6, ask=scripted(["Python", "Depends", "C", "Lisp",
                                     "Go", "Haskell"]))
        self.assertEqual(r["verdict"], "store")
        self.assertEqual(r["reason"], "model-has-no-stable-view")
        self.assertEqual(r["distinct"], 6)

    def test_only_two_verdicts_exist(self):
        # A live run showed a well-posed private question scattering exactly
        # like a vague one, so any third verdict would be claiming knowledge
        # this test does not have.
        seen = set()
        for answers in (["42"], ["28"], ["a", "b", "c", "d"],
                        [ConnectionError("down")]):
            seen.add(probe("q", "42", n=4, ask=scripted(answers))["verdict"])
        self.assertEqual(seen, {"skip", "store"})

    def test_mostly_known_still_skips(self):
        # 6 of 8 is above the 0.625 line; two stray samples do not flip it.
        r = probe("q", "42", n=8,
                  ask=scripted(["42", "42", "42", "42", "42", "42", "7", "9"]))
        self.assertEqual(r["verdict"], "skip")

    def test_a_bare_majority_of_hits_is_not_enough(self):
        r = probe("q", "42", n=8,
                  ask=scripted(["42", "42", "42", "42", "7", "9", "11", "13"]))
        self.assertNotEqual(r["verdict"], "skip")

    def test_near_miss_counts_as_known(self):
        # The generator has the fact; it just rounds differently.
        r = probe("tokens per second?", "117", n=8, ask=scripted(["117.4"]))
        self.assertEqual(r["verdict"], "skip")


class Failure(unittest.TestCase):

    def test_total_failure_stores(self):
        r = probe("q", "42", n=4,
                  ask=scripted([ConnectionError("server down")]))
        self.assertEqual(r["verdict"], "store")
        self.assertEqual(r["reason"], "probe-failed")
        self.assertTrue(r["errors"])

    def test_partial_failure_still_decides(self):
        # Three good samples and one exception: use what came back.
        r = probe("q", "42", n=4,
                  ask=scripted([TimeoutError("slow"), "42", "42", "42"]))
        self.assertEqual(r["verdict"], "skip")
        self.assertEqual(r["n"], 3)
        self.assertTrue(r["errors"])

    def test_empty_replies_are_not_a_crash(self):
        r = probe("q", "42", n=4, ask=scripted([""]))
        self.assertEqual(r["verdict"], "store")


class Thresholds(unittest.TestCase):
    """The lines are configurable and the numbers behind them are returned,
    because a threshold nobody can audit is one nobody should trust."""

    def test_numbers_are_reported(self):
        r = probe("q", "42", n=4, ask=scripted(["28"]))
        for k in ("hit_rate", "self_agreement", "distinct", "mode", "n",
                  "samples", "seconds", "detail"):
            self.assertIn(k, r)

    def test_should_store_one_liner(self):
        self.assertFalse(M.should_store("q", "42", n=4, ask=scripted(["42"])))
        self.assertTrue(M.should_store("q", "42", n=4, ask=scripted(["28"])))


if __name__ == "__main__":
    unittest.main(verbosity=2)
