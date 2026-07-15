#!/usr/bin/env python3
"""verification_signal_audit.py — measure the fix-loop's ADVISORY acceptance
signals against ground truth, to decide whether any is clean enough to promote
from an advisory note to a (soft) submit gate.

Issue #2 (verification frontier): the fix-loop submit gate is
    seen_red AND repro_green AND non-test-diff
all computed on the MODEL'S OWN reproduction. A repro that goes RED via an
uncaught exception and GREEN once the exception stops (e.g. ends `assert True`)
satisfies the gate WITHOUT checking the produced VALUE, so the model declares
while the held-out FAIL_TO_PASS (which check the value) fail = a "declared-wrong"
miss. Two advisory signals already exist but were never promoted to a gate
because their discrimination was never quantified on labeled data:
  (a) repro_strength   = swe_fix_tools._reproduction_strength(repro_script)
                         {value_check | vacuous_constant | weak}
  (b) regressions      = fix_state.regressions (neighbor pre-existing tests that
                         passed pre-patch and now fail)
This tool quantifies both, per repo and overall, and simulates the cost/benefit
of a soft gate.

ANSWER-LEAKAGE SAFE: reads ONLY (1) the model's OWN reproduction script and its
own recorded fix_state (repro_script / regressions / fix_verified), and (2) the
PUBLIC per-instance `resolved` flag + the Docker-confirmed false-negative
manifest. It never reads gold_patch / test_patch / FAIL_TO_PASS, and it feeds
nothing back into any model. Pure offline measurement (no test execution, no
network, no Docker).

Correctness labels (leakage-safe):
  CORRECT = home-resolved OR in confirmed_false_negatives (patch is right; the
            home miss was an env false-negative, Docker-confirmed).
  WRONG   = home-miss AND not a known/likely env false-negative. To keep WRONG =
            genuine "declared-wrong" (healthy env, tests ran, failed on VALUE),
            we exclude any miss whose score_tail carries an env-FN signature
            (NO_COLLECTORS / collection-error / all-skipped / network / bare
            rootdir), which are env misses, not value misses.
"""
import ast, json, os, re, sys, collections

LLMOS = os.path.expanduser("~/Code/LLMOS")
SWE = os.path.expanduser("~/swe")
RESULTS = os.path.join(SWE, "results_full300.json")
TRACES = os.path.join(SWE, "traces_v2")
MANIFEST = os.path.join(LLMOS, "swe_false_negatives.json")


def load_shipped_reproduction_strength():
    """Extract and exec the EXACT shipped _reproduction_strength source from
    swe_fix_tools.py (byte-for-byte fidelity, no heavy imports)."""
    src = open(os.path.join(LLMOS, "swe_fix_tools.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_reproduction_strength":
            ns = {"re": re}
            exec(compile(ast.Module([node], []), "<shipped>", "exec"), ns)
            return ns["_reproduction_strength"]
    raise RuntimeError("could not locate _reproduction_strength in swe_fix_tools.py")


ENV_FN_SIG = re.compile(
    r"found no collectors|no collectors"        # matplotlib warnings-as-errors
    r"|error in \d|errors during collection"    # collection error
    r"|connectionerror|httpbin"                  # network-coupled
    , re.I)


def strip_ansi(t):
    return re.sub(r"\x1b\[[0-9;]*m", "", t or "")


def is_env_fn_signature(tail):
    t = strip_ansi(tail).lower()
    if ENV_FN_SIG.search(t):
        return True
    # all-skipped (importorskip) with no pass/fail
    if "skipped" in t and "passed" not in t and "failed" not in t:
        return True
    # bare rootdir with no result summary (old-pytest --no-header auto-miss era)
    if "rootdir" in t and "passed" not in t and "failed" not in t and "error" not in t:
        return True
    return False


def main():
    strength_fn = load_shipped_reproduction_strength()
    results = json.load(open(RESULTS))
    manifest = json.load(open(MANIFEST))

    def ids(key):
        out = set()
        for e in manifest.get(key, []) or []:
            if isinstance(e, dict):
                v = e.get("instance_id") or e.get("id")
                if v:
                    out.add(v)
        return out
    confirmed_fn = ids("confirmed_false_negatives")

    rows = []
    for r in results:
        iid = r["id"]
        tpath = os.path.join(TRACES, iid + ".trace.json")
        fs = {}
        if os.path.isfile(tpath):
            try:
                tr = json.load(open(tpath))
                fs = tr.get("fix_state") or {}
            except Exception:
                fs = {}
        repro = fs.get("repro_script") or ""
        strength = strength_fn(repro)
        rows.append({
            "id": iid,
            "repo": iid.split("__")[0],
            "resolved": bool(r.get("resolved")),
            "p2": r.get("phase2_reason"),
            "self_verified": bool(r.get("fix_verified_by_model")),
            "gate_fix_verified": bool(fs.get("fix_verified")),
            "repro_green": bool(fs.get("repro_green")),
            "seen_red": bool(fs.get("seen_red")),
            "n_regressions": len(fs.get("regressions") or []),
            "has_repro": bool(repro),
            "strength": strength,
            "env_fn_sig": is_env_fn_signature(r.get("score_tail")),
            "in_confirmed_fn": iid in confirmed_fn,
        })

    # ---- correctness labels ----
    def correctness(x):
        if x["resolved"] or x["in_confirmed_fn"]:
            return "CORRECT"
        if x["env_fn_sig"] or x["in_confirmed_fn"]:
            return "ENV"  # env-miss, not a value-miss; excluded from WRONG
        return "WRONG"
    for x in rows:
        x["label"] = correctness(x)

    n = len(rows)
    print(f"# verification-signal audit  ({n} completed instances)")
    print(f"# traces present: {sum(1 for x in rows if x['has_repro'])}"
          f" with a registered repro_script")

    # ---- 0. global self-verified weak-repro cross-check (vs the 09:39 figure) ----
    sv = [x for x in rows if x["self_verified"]]
    sv_res = [x for x in sv if x["resolved"]]
    sv_miss = [x for x in sv if not x["resolved"]]
    def weakrate(xs):
        if not xs:
            return 0.0, 0
        w = sum(1 for x in xs if x["strength"] != "value_check")
        return w / len(xs), len(xs)
    wr_m, nm = weakrate(sv_miss)
    wr_r, nr = weakrate(sv_res)
    print("\n## 0. cross-check (self-verified population; 'weak' = strength != value_check)")
    print(f"   self-verified MISSES:   weak-repro {wr_m:.0%}  (n={nm})   [09:39 logged ~68%]")
    print(f"   self-verified RESOLVES: weak-repro {wr_r:.0%}  (n={nr})   [09:39 logged ~55%]")

    # ---- population for a SOFT GATE simulation ----
    # A soft gate only ever fires where the model would SUBMIT: gate satisfied
    # (fix_verified True) and repro GREEN. Restrict to CORRECT vs WRONG (drop ENV
    # + any not-yet-declared).
    pop = [x for x in rows if x["gate_fix_verified"] and x["repro_green"]
           and x["label"] in ("CORRECT", "WRONG")]
    print(f"\n## 1. declared population (gate satisfied + GREEN, value-graded): "
          f"{len(pop)}  ({sum(1 for x in pop if x['label']=='CORRECT')} CORRECT / "
          f"{sum(1 for x in pop if x['label']=='WRONG')} WRONG)")
    excl_env = [x for x in rows if x["gate_fix_verified"] and x["repro_green"]
                and x["label"] == "ENV"]
    print(f"   (excluded {len(excl_env)} env-miss declares from WRONG: "
          f"{[x['id'] for x in excl_env][:8]})")

    # ---- confusion matrix: strength x label ----
    print("\n## 2. repro_strength x correctness (declared population)")
    order = ["value_check", "vacuous_constant", "weak"]
    hdr = f"   {'strength':17s} {'CORRECT':>8s} {'WRONG':>7s}"
    print(hdr)
    cm = collections.defaultdict(lambda: collections.Counter())
    for x in pop:
        cm[x["strength"]][x["label"]] += 1
    for s in order:
        print(f"   {s:17s} {cm[s]['CORRECT']:8d} {cm[s]['WRONG']:7d}")

    # ---- soft-gate simulation: flag if GREEN and strength != value_check ----
    flagged_wrong = [x for x in pop if x["strength"] != "value_check" and x["label"] == "WRONG"]
    flagged_corr = [x for x in pop if x["strength"] != "value_check" and x["label"] == "CORRECT"]
    n_wrong = sum(1 for x in pop if x["label"] == "WRONG")
    n_corr = sum(1 for x in pop if x["label"] == "CORRECT")
    print("\n## 3. soft-gate simulation A: flag when GREEN and strength != value_check")
    print(f"   catches {len(flagged_wrong)}/{n_wrong} WRONG "
          f"({(len(flagged_wrong)/n_wrong if n_wrong else 0):.0%} of declared-wrong)")
    print(f"   false-flags {len(flagged_corr)}/{n_corr} CORRECT "
          f"({(len(flagged_corr)/n_corr if n_corr else 0):.0%} of correct declares)")
    print(f"   WRONG caught: {[x['id'] for x in flagged_wrong]}")
    print(f"   CORRECT false-flagged: {[x['id'] for x in flagged_corr]}")

    # ---- soft-gate simulation B: only vacuous_constant (strictest, lowest FP) ----
    fw = [x for x in pop if x["strength"] == "vacuous_constant" and x["label"] == "WRONG"]
    fc = [x for x in pop if x["strength"] == "vacuous_constant" and x["label"] == "CORRECT"]
    print("\n## 4. soft-gate simulation B: flag only vacuous_constant (assert True/1)")
    print(f"   catches {len(fw)}/{n_wrong} WRONG; false-flags {len(fc)}/{n_corr} CORRECT")
    print(f"   WRONG caught: {[x['id'] for x in fw]}")
    print(f"   CORRECT false-flagged: {[x['id'] for x in fc]}")

    # ---- regression signal ----
    print("\n## 5. neighbor-regression signal (declared population)")
    reg_wrong = [x for x in pop if x["n_regressions"] > 0 and x["label"] == "WRONG"]
    reg_corr = [x for x in pop if x["n_regressions"] > 0 and x["label"] == "CORRECT"]
    print(f"   nonempty regressions on WRONG:   {len(reg_wrong)}/{n_wrong}")
    print(f"   nonempty regressions on CORRECT: {len(reg_corr)}/{n_corr}  (false alarms)")

    # ---- per-repo declared-wrong breakdown ----
    print("\n## 6. declared-wrong by repo (the Issue #2 frontier)")
    byrepo = collections.Counter(x["repo"] for x in pop if x["label"] == "WRONG")
    for k, v in byrepo.most_common():
        print(f"   {k:16s} {v}")

    return rows, pop


if __name__ == "__main__":
    main()
