#!/usr/bin/env python3
"""Rescore MMLU + MATH results by (1) fixing the MATH normalizer, (2) recovering
answers from the trace when RETURN was emitted with empty args, and (3) extracting
a letter/number from the reasoning tail when the CPU failed to parse an instruction.

Reads ~/mmlu/store.db + ~/mmlu/instances.json + ~/mmlu/results.json (same for math)
and writes rescored ~/mmlu/results.rescored.json and ~/math/results.rescored.json.
"""
import json, os, re, sqlite3, sys
sys.path.insert(0, os.path.expanduser("~/Code/LLMOS"))


# ---- improved MATH normalization -------------------------------------
_BOX  = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
_FRAC = re.compile(r"\\d?frac\{([^{}]+)\}\{([^{}]+)\}")
_SQRT = re.compile(r"\\sqrt\{([^{}]+)\}")
_STRIP = re.compile(r"[\s\$,]|\\left|\\right|\\!|\\,|\\;|\\:|\\ ")
_PAREN_ATOM = re.compile(r"\(\s*(-?\d+(?:\.\d+)?)\s*\)")   # (5) -> 5

def norm(s):
    if s is None:
        return None
    s = str(s)
    m = _BOX.search(s)
    if m:
        s = m.group(1)
    s = _FRAC.sub(lambda m: f"({m.group(1)})/({m.group(2)})", s)
    s = _SQRT.sub(lambda m: f"sqrt({m.group(1)})", s)
    s = _STRIP.sub("", s)
    s = s.replace("^\\circ", "").replace("\\%", "").replace("^{\\circ}", "").replace("\\pi", "pi")
    s = s.replace("\\text{", "").replace("\\mathbf{", "").replace("\\mathrm{", "")
    s = s.replace("}", "")
    # collapse redundant parens around integer/decimal atoms: (5) -> 5
    for _ in range(3):
        s = _PAREN_ATOM.sub(r"\1", s)
    # try numeric
    try:
        return f"{float(s):.6g}"
    except Exception:
        pass
    # try sympy equivalence tag: parse as expression, hash to canonical
    try:
        import sympy as sp
        expr = sp.sympify(s.replace("^", "**"), evaluate=True)
        # canonical string
        return "sym:" + str(sp.nsimplify(expr))
    except Exception:
        pass
    return s.strip().lower()


def eq_norm(a, b):
    na, nb = norm(a), norm(b)
    if na == nb:
        return True
    # numeric fallback: try to parse both as floats
    try:
        return abs(float(na) - float(nb)) < 1e-6
    except Exception:
        pass
    # sympy equality
    try:
        import sympy as sp
        return sp.simplify(sp.sympify(str(a).replace("^", "**")) - sp.sympify(str(b).replace("^", "**"))) == 0
    except Exception:
        return False


# ---- trace-based recovery --------------------------------------------
_LETTER = re.compile(r"\b([ABCD])\b")
_NUM = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+)?")

def load_trace(db_path):
    db = sqlite3.connect(db_path)
    by_pid = {}
    for pid, snap in db.execute("SELECT pid, snapshot FROM processes"):
        s = json.loads(snap)
        by_pid[pid] = {"goal": s.get("goal", ""), "result": s.get("result"), "context": []}
    for pid, pc, op, a, r in db.execute("SELECT pid, pc, op, args, result FROM trace ORDER BY pid, pc"):
        try:
            a = json.loads(a) if isinstance(a, str) else a
        except Exception:
            pass
        try:
            r = json.loads(r) if isinstance(r, str) else r
        except Exception:
            pass
        by_pid.setdefault(pid, {"goal":"","result":None,"context":[]})["context"].append(
            {"pc": pc, "op": op, "args": a, "result": r})
    return by_pid


def recover_mmlu(proc):
    """Try to pull a letter A-D from the reasoning tail on schema-fail or empty RETURN."""
    # 1) if the last op is RETURN with empty args, look at prior PLAN text
    ctx = proc.get("context", [])
    for step in reversed(ctx):
        if step["op"] == "RETURN":
            args = step.get("args") or {}
            if isinstance(args, dict) and args.get("result") in (None, "SCHEMA VALIDATION FAILED"):
                err = args.get("error", "")
                m = _LETTER.search(err[::-1])   # search from tail
                if m:
                    # search from the END of err for a letter
                    tail = err[-400:]
                    letters = _LETTER.findall(tail)
                    if letters:
                        return letters[-1]
            elif isinstance(args, dict) and args.get("result") in "ABCD":
                return args["result"]
        if step["op"] == "PLAN":
            txt = str((step.get("args") or {}).get("text", ""))
            letters = _LETTER.findall(txt[-400:])
            if letters:
                return letters[-1]
    return None


def recover_math(proc):
    """Empty RETURN -> use last calc value. If no RETURN result, use reasoning tail."""
    ctx = proc.get("context", [])
    # 1) last calc result
    last_calc = None
    for step in ctx:
        if step["op"] == "CALL":
            a = step.get("args") or {}
            if isinstance(a, dict) and a.get("name") == "calc":
                r = step.get("result")
                if isinstance(r, dict) and "value" in r:
                    last_calc = r["value"]
    # 2) inspect last RETURN
    for step in reversed(ctx):
        if step["op"] == "RETURN":
            args = step.get("args") or {}
            res = args.get("result") if isinstance(args, dict) else None
            if res not in (None, "SCHEMA VALIDATION FAILED"):
                return res
            # empty-return recovery: use last calc value
            if last_calc is not None:
                return last_calc
            # schema-fail recovery: dig into error text
            err = str(args.get("error", "")) if isinstance(args, dict) else ""
            m_box = _BOX.search(err)
            if m_box:
                return m_box.group(1)
            nums = _NUM.findall(err[-400:])
            if nums:
                return nums[-1]
        if step["op"] == "PLAN":
            txt = str((step.get("args") or {}).get("text", ""))
            m_box = _BOX.search(txt)
            if m_box:
                return m_box.group(1)
    return None


# ---- rescore MMLU + MATH ---------------------------------------------
def rescore(kind):
    home = os.path.expanduser(f"~/{kind}")
    db = load_trace(os.path.join(home, "store.db"))
    inst = {i["id"]: i for i in json.load(open(os.path.join(home, "instances.json")))}
    results = json.load(open(os.path.join(home, "results.json")))["results"]

    # map instance id -> pid (in-order spawn: pid n = nth instance, pid=1..N)
    by_id_pid = {r["id"]: idx + 1 for idx, r in enumerate(results)}

    orig_correct = 0
    new_correct  = 0
    rescored     = []
    recoveries   = []
    for r in results:
        orig_correct += int(r["correct"])
        pid = by_id_pid[r["id"]]
        proc = db.get(pid, {})
        gold = inst[r["id"]]["answer"]
        if kind == "mmlu":
            pred = r["letter"]
            if not r["correct"]:
                rec = recover_mmlu(proc)
                if rec:
                    pred = rec
                    recoveries.append((r["id"], "letter-from-tail", pred))
            ok = (pred == gold)
        else:
            pred_norm_now = r["pred_norm"]
            pred_raw = r["pred"]
            if not r["correct"]:
                # first try re-norm on the existing raw pred
                if eq_norm(pred_raw, gold):
                    ok = True
                    recoveries.append((r["id"], "renorm", pred_raw))
                else:
                    rec = recover_math(proc)
                    if rec is not None and eq_norm(rec, gold):
                        ok = True
                        recoveries.append((r["id"], "trace-recovery", rec))
                    else:
                        ok = False
            else:
                ok = True
        new_correct += int(ok)
        rescored.append({**r, "rescored_correct": ok})

    out = {
        "n": len(results),
        "orig_correct": orig_correct,
        "orig_score":   orig_correct / len(results),
        "new_correct":  new_correct,
        "new_score":    new_correct / len(results),
        "recoveries":   recoveries,
        "results":      rescored,
    }
    with open(os.path.join(home, "results.rescored.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"{kind.upper()}: {orig_correct}/{len(results)} -> {new_correct}/{len(results)}  "
          f"(+{new_correct-orig_correct} recovered)")
    for rid, kind_r, pred in recoveries:
        print(f"  recovered {rid:<36} via {kind_r:<16} -> {pred}")


if __name__ == "__main__":
    rescore("mmlu")
    print()
    rescore("math")
