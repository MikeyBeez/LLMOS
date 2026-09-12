"""LLMOS - the memory write gate: store only what the generator cannot recall.

ARCHITECTURE.md 4.6 (memory management). One piece of it: the decision of
whether a candidate fact is worth writing down at all.

THE RULE, in Mikey's words: the agent finds that the optimal X is 42. Ask a
FRESH context "what is the optimal X?". If it comes back 42, do not store it
-- you can always retrieve it from the generator. If it comes back 28, store
it, because now you know something the weights do not.

WHY A FRESH CONTEXT. Asking the working model mid-task "what did you expect"
is worthless: the answer is sitting in the window. A clean context cannot
cheat. That is the whole measurement.

WHY SAMPLING AND NOT ONE GREEDY ANSWER. One sample is a coin flip -- the same
question can answer 42 once and 37 the next time, and the decision would
depend on which run you got. N samples give two numbers instead of one, and
the second is the useful one:

    hit rate       how often the clean model produces OUR answer
                   -> does the generator already have this?
    self-agreement how often the clean model agrees with ITSELF
                   -> is this question well posed at all?

Measured on pop (Qwen3.8-27B, thinking off, temp 0.8, n=8, 2026-09-11):

    capital of France      8/8 self-agreement, 1 distinct answer
    degrees in a circle    8/8, 1
    speed of light         8/8, 1
    "optimal seagulls"     3/8, 6
    a private benchmark    2/8, 6   (answers 1, 37, 240 -- truth was 117)

Known facts pin at 8/8. Anything the model has no stable view of scatters.
The gap is wide enough that the thresholds below are not delicate.

TWO VERDICTS, AND WHY NOT THREE. skip if a clean context reliably produces
our answer; store otherwise. An earlier draft had a third verdict for the
scattered case, calling it a weak retrieval key -- and the first live run
killed it: a well-posed question about a private benchmark scattered exactly
like the deliberately vague seagull one. This test CANNOT tell "the question
is vague" from "the model has never heard of this", and a verdict that claims
to is lying. The scatter is reported in the numbers; it is not a judgement.

The `reason` field says which kind of store it was:
    model-is-confidently-wrong  the model reliably says something ELSE. The
                                best kind of memory -- the generator is wrong
                                and now we know it.
    model-has-no-stable-view    the model scattered. Not retrievable either
                                way, so store it.

FAIL-SAFE. If the probe cannot run -- server down, timeout, bad reply -- the
verdict is "store". Losing a real memory costs more than keeping a redundant
one, so the failure direction is deliberate.

THE SAME PROBE DELETES. Re-run it later against a newer model: a memory whose
verdict has become "skip" is now redundant with the weights and can be
dropped. Memory that shrinks as the model grows.

CLI:  python3 memory_surprise.py "What is the optimal number of X?" "42"
"""

from __future__ import annotations

import collections
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

URL = os.environ.get("LLMOS_PROBE_URL",
                     "http://127.0.0.1:8080/v1/chat/completions")
N = int(os.environ.get("LLMOS_PROBE_N", "8"))
TEMP = float(os.environ.get("LLMOS_PROBE_TEMP", "0.8"))
TIMEOUT = float(os.environ.get("LLMOS_PROBE_TIMEOUT", "120"))

# A mode this common means the model has a stable view. 5 of 8. Known facts
# measured 8/8 and scattered ones 2-4/8, so the line sits in open space.
CONSISTENT = float(os.environ.get("LLMOS_PROBE_CONSISTENT", "0.625"))
# This much agreement with OUR answer means the generator already has it.
KNOWN = float(os.environ.get("LLMOS_PROBE_KNOWN", "0.625"))
# Numbers within this relative distance count as the same answer: 117 and
# 117.3 are not a knowledge gap, they are rounding.
REL_TOL = float(os.environ.get("LLMOS_PROBE_REL_TOL", "0.02"))

SYSTEM = ("Answer with the value only -- a number or a few words. "
          "No sentence, no units, no markdown, no explanation.")


# ------------------------------------------------------------- normalisation

_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def normalise(s: str) -> str:
    """Strip the model's decoration down to something comparable.

    Qwen answers 'The capital of France is **Paris**' even when told not to,
    and writes 299,792,458 half the time and 299792458 the other half. Both
    of those are the same answer and must not read as disagreement.
    """
    s = re.sub(r"[*_`]", "", s or "").strip().lower()
    s = s.rstrip(".!").strip()
    m = _NUM.search(s)
    if m:
        return m.group(0).replace(",", "")
    return " ".join(s.split())


def as_number(s: str):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def same_answer(a: str, b: str, rel_tol: float = REL_TOL) -> bool:
    """Equality that understands numbers, so near-misses do not lie."""
    na, nb = normalise(a), normalise(b)
    if na == nb:
        return True
    xa, xb = as_number(na), as_number(nb)
    if xa is not None and xb is not None:
        scale = max(abs(xa), abs(xb), 1e-9)
        return abs(xa - xb) / scale <= rel_tol
    # A short answer swallowed by a longer one ("paris" in "paris france").
    if na and nb and (na in nb or nb in na) and min(len(na), len(nb)) >= 3:
        return True
    return False


# ------------------------------------------------------------------ the ask

def ask_once(question: str, url=URL, temp=TEMP, timeout=TIMEOUT) -> str:
    """One clean-context answer. No history, no prompt cache, no thinking.

    cache_prompt False so one sample cannot prime the next. enable_thinking
    False because a reasoning pass is the model working the answer out, and
    what we want to measure is what it already holds.
    """
    body = json.dumps({
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": question}],
        "temperature": temp, "top_p": 0.95, "max_tokens": 32,
        "cache_prompt": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    return (d["choices"][0]["message"].get("content") or "").strip()


# ---------------------------------------------------------------- the probe

def probe(question: str, answer: str, n=N, url=URL, temp=TEMP,
          timeout=TIMEOUT, ask=None) -> dict:
    """Decide whether `answer` to `question` is worth remembering.

    Returns a dict carrying the verdict AND the numbers behind it, because a
    threshold nobody can audit is a threshold nobody should trust.
    """
    ask = ask or (lambda q: ask_once(q, url=url, temp=temp, timeout=timeout))
    t0 = time.time()
    samples, errors = [], []
    for _ in range(max(1, n)):
        try:
            samples.append(ask(question))
        except Exception as e:                      # network, timeout, shape
            errors.append(f"{type(e).__name__}: {e}")

    if not samples:
        return {
            "verdict": "store", "reason": "probe-failed",
            "detail": "the clean-context probe could not run, so this could "
                      "not be shown redundant; storing is the safe direction",
            "errors": errors[:3], "question": question, "answer": answer,
            "seconds": round(time.time() - t0, 2),
        }

    normed = [normalise(s) for s in samples]
    counts = collections.Counter(normed)
    mode, mode_n = counts.most_common(1)[0]
    self_agreement = mode_n / len(normed)
    hits = sum(1 for s in samples if same_answer(s, answer))
    hit_rate = hits / len(samples)

    if hit_rate >= KNOWN:
        verdict, reason = "skip", "already-in-the-weights"
        detail = (f"a clean context produced this answer {hits}/{len(samples)} "
                  f"times; it can be retrieved from the generator")
    elif self_agreement >= CONSISTENT:
        verdict, reason = "store", "model-is-confidently-wrong"
        detail = (f"a clean context reliably says {mode!r} "
                  f"({mode_n}/{len(normed)}), not {normalise(answer)!r}")
    else:
        verdict, reason = "store", "model-has-no-stable-view"
        detail = (f"a clean context scattered across {len(counts)} answers in "
                  f"{len(normed)} tries (mode {mode!r}, {mode_n}); not "
                  f"retrievable. NOTE this does not distinguish 'the model "
                  f"has never heard of it' from 'the question is vague' -- "
                  f"both scatter. If the fact is one the model plausibly "
                  f"should know, the question is the suspect.")

    return {
        "verdict": verdict, "reason": reason, "detail": detail,
        "question": question, "answer": answer,
        "hit_rate": round(hit_rate, 3),
        "self_agreement": round(self_agreement, 3),
        "distinct": len(counts), "mode": mode, "n": len(samples),
        "samples": samples, "errors": errors[:3],
        "seconds": round(time.time() - t0, 2),
    }


def should_store(question: str, answer: str, **kw) -> bool:
    """The one-line form, for a caller that wants a yes or no."""
    return probe(question, answer, **kw)["verdict"] != "skip"


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__.strip().splitlines()[-1]); sys.exit(2)
    r = probe(sys.argv[1], sys.argv[2])
    print(json.dumps({k: v for k, v in r.items() if k != "samples"}, indent=2))
    print("samples:", r.get("samples"))
