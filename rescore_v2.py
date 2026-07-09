#!/usr/bin/env python3
"""Rescore v11 (and any run) with:
  1. Stronger MATH normalization (\\text{}, redundant parens, unicode).
  2. Sympy equivalence check as a fallback.
  3. Trace-tail extraction for SCHEMA VALIDATION FAILED / empty RETURN cases.
  4. MMLU letter extraction from PLAN tails.

Reads ~/{mmlu,math}/{results.vN.json,store.db,instances.json}, writes
results.vN.rescored.json.

Usage:
    ~/swebench-venv/bin/python rescore_v2.py mmlu v11
    ~/swebench-venv/bin/python rescore_v2.py math v11
"""
import json, os, re, sqlite3, sys

# ---------- MATH normalization (superset of rescore.py) ------------
_BOX  = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
_TEXT = re.compile(r"\\text\{([^{}]*)\}")
_FRAC = re.compile(r"\\d?frac\{([^{}]+)\}\{([^{}]+)\}")
_SQRT = re.compile(r"\\sqrt\{([^{}]+)\}")
_STRIP = re.compile(r"[\s\$,]|\\left|\\right|\\!|\\,|\\;|\\:|\\ ")
_PAREN_ATOM = re.compile(r"\(\s*(-?\d+(?:\.\d+)?(?:/\d+)?)\s*\)")   # (5) -> 5, (1/2) -> 1/2

def norm(s):
    if s is None: return None
    s = str(s)
    m = _BOX.search(s)
    if m: s = m.group(1)
    s = _TEXT.sub(r"\1", s)   # \text{saturday} -> saturday
    s = _FRAC.sub(lambda m: f"({m.group(1)})/({m.group(2)})", s)
    s = _SQRT.sub(lambda m: f"sqrt({m.group(1)})", s)
    s = _STRIP.sub("", s)
    s = s.replace("^\\circ", "").replace("\\%", "").replace("^{\\circ}", "").replace("\\pi", "pi")
    s = s.replace("\\mathbf{", "").replace("\\mathrm{", "").replace("\\left", "").replace("\\right", "")
    s = s.replace("}", "")
    for _ in range(4):
        s = _PAREN_ATOM.sub(r"\1", s)
    try:
        return f"{float(s):.6g}"
    except Exception:
        pass
    return s.strip().lower()


def eq_math(a, b):
    na, nb = norm(a), norm(b)
    if na is None or nb is None: return False
    if na == nb: return True
    try:
        return abs(float(na) - float(nb)) < 1e-6
    except Exception:
        pass
    try:
        import sympy as sp
        ea = sp.sympify(str(a).replace("^", "**"), evaluate=True)
        eb = sp.sympify(str(b).replace("^", "**"), evaluate=True)
        return sp.simplify(ea - eb) == 0
    except Exception:
        return False


# ---------- trace-tail extraction ----------
_LETTER = re.compile(r"\b([A-D])\b")
_NUM = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+)?")

def extract_from_trace_math(steps):
    """Look for the boxed answer, the last calc value, or a number in PLAN text."""
    # 1. \boxed{...} in any PLAN text
    for step in reversed(steps):
        if step["op"] == "PLAN":
            txt = str((step.get("args") or {}).get("text", ""))
            m = _BOX.search(txt)
            if m: return m.group(1)
    # 2. last calc value
    for step in reversed(steps):
        if step["op"] == "CALL":
            args = step.get("args") or {}
            if isinstance(args, dict) and args.get("name") == "calc":
                r = step.get("result")
                if isinstance(r, dict) and "value" in r:
                    return r["value"]
    # 3. RETURN error text (which contains a snippet of raw prose)
    for step in reversed(steps):
        if step["op"] == "RETURN":
            args = step.get("args") or {}
            err = str(args.get("error", "") if isinstance(args, dict) else "")
            m = _BOX.search(err)
            if m: return m.group(1)
            nums = _NUM.findall(err[-400:])
            if nums: return nums[-1]
    return None


def _mmlu_from_text(t):
    """Extract letter from any text blob. Order: 'answer is X', 'final ... X',
    boxed{X}, then last bare A-D in the tail."""
    if not t: return None
    # highest confidence: "the answer is X" or similar
    for pat in [
        r"(?:the\s+)?(?:correct\s+)?answer\s+is\s*[\W_]*([A-D])\b",
        r"(?:final|correct|right)\s+(?:answer|choice|option)\s*(?:is)?\s*[\W_]*([A-D])\b",
        r"choose\s+(?:option\s+)?([A-D])\b",
        r"\\boxed\{([A-D])\}",
    ]:
        m = re.search(pat, t, re.IGNORECASE)
        if m: return m.group(1).upper()
    # lowest confidence: last bare A-D in the tail
    letters = _LETTER.findall(t[-800:])
    if letters: return letters[-1]
    return None


def extract_from_trace_mmlu(steps):
    for step in reversed(steps):
        if step["op"] == "RETURN":
            args = step.get("args") or {}
            # RETURN args of a SCHEMA-FAIL row carry `raw` (full model output).
            # Check that FIRST — err is truncated to 160 chars and often useless.
            raw = str(args.get("raw", "") if isinstance(args, dict) else "")
            letter = _mmlu_from_text(raw)
            if letter: return letter
            err = str(args.get("error", "") if isinstance(args, dict) else "")
            letter = _mmlu_from_text(err)
            if letter: return letter
        if step["op"] == "PLAN":
            txt = str((step.get("args") or {}).get("text", ""))
            letter = _mmlu_from_text(txt)
            if letter: return letter
    return None


def load_steps(db, pid):
    rows = db.execute("SELECT op, args, result FROM trace WHERE pid=? ORDER BY pc", (pid,)).fetchall()
    steps = []
    for op, a, r in rows:
        try: a = json.loads(a) if isinstance(a, str) else a
        except Exception: pass
        try: r = json.loads(r) if isinstance(r, str) else r
        except Exception: pass
        steps.append({"op": op, "args": a, "result": r})
    return steps


def rescore(kind, tag):
    home = os.path.expanduser(f"~/{kind}")
    res_path = os.path.join(home, f"results.{tag}.json")
    db = sqlite3.connect(os.path.join(home, "store.db"))
    results = json.load(open(res_path))["results"]

    orig = sum(int(r["correct"]) for r in results)
    new_correct = 0
    recovered = []
    for i, r in enumerate(results, 1):
        ok = r["correct"]
        why = ""
        if not ok:
            steps = load_steps(db, i)
            if kind == "mmlu":
                gold = r["gold"]
                pred = r.get("letter")
                if not pred:
                    ex = extract_from_trace_mmlu(steps)
                    if ex and ex == gold:
                        ok = True; why = f"mmlu-trace-tail={ex}"
            else:
                # math: try eq_math on the existing raw pred first
                gold = r["gold"]
                if eq_math(r.get("pred"), gold):
                    ok = True; why = f"math-renorm(pred={r.get('pred')})"
                else:
                    ex = extract_from_trace_math(steps)
                    if ex is not None and eq_math(ex, gold):
                        ok = True; why = f"math-trace={ex}"
        new_correct += int(ok)
        if why: recovered.append((r["id"], why))
    out = {
        "kind": kind, "tag": tag, "n": len(results),
        "orig": orig, "orig_pct": orig / len(results),
        "new":  new_correct, "new_pct": new_correct / len(results),
        "recovered": recovered,
    }
    print(f"{kind.upper()} {tag}: {orig}/{len(results)} -> {new_correct}/{len(results)}")
    for rid, why in recovered:
        print(f"  {rid:<40} via {why}")
    with open(os.path.join(home, f"results.{tag}.rescored.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    kind = sys.argv[1] if len(sys.argv) > 1 else "math"
    tag  = sys.argv[2] if len(sys.argv) > 2 else "v11"
    rescore(kind, tag)
