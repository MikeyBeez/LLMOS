#!/usr/bin/env python3
"""P2P regression audit (false-POSITIVE finder), symmetric to
reclaim_false_negatives.py. The home scorer (swe_agent_v2.score) runs ONLY
FAIL_TO_PASS, never PASS_TO_PASS, so a model patch that fixes F2P but breaks a
previously-passing test is scored resolved=True at home while the authoritative
Docker scorer (F2P pass AND P2P stay green) would mark it UNRESOLVED.

This tool re-runs each RESOLVED instance's PASS_TO_PASS via the SAME
test_runner path the scorer uses, in the instance's surviving work-dir (which
already holds model_patch + test_patch). It NEVER modifies results_full300.json;
it writes a separate report. P2P is public scoring metadata and is executed at
scoring time only -- never shown to the model -> no answer leakage.

A P2P failure here is a SUSPECTED false positive, to be confirmed by the
authoritative Docker eval before any reclassification (same PENDING discipline
as the false-negative manifest).
"""
import json, os, sys, argparse, time
sys.path.insert(0, os.path.expanduser("~/Code/LLMOS"))
import test_runner as tr

RESULTS = os.path.expanduser("~/swe/results_full300.json")
INSTS   = os.path.expanduser("~/swe/instances_full300.json")
WORK    = os.path.expanduser("~/swe/work")
OUT     = os.path.expanduser("~/swe/p2p_audit_report.json")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*", default=None, help="explicit instance ids")
    ap.add_argument("--max-p2p", type=int, default=40, help="skip instances with more P2P ids than this (keep bounded/fast)")
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--limit", type=int, default=0, help="cap number of instances audited (0=all)")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    results = json.load(open(RESULTS))
    insts = {i["instance_id"]: i for i in json.load(open(INSTS))}
    resolved_ids = [r["id"] for r in results if r.get("resolved")]
    env_kind = {r["id"]: r.get("env_kind") or "uv" for r in results}

    if args.ids:
        targets = args.ids
    else:
        targets = resolved_ids

    report = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"),
              "scorer_gap": "score() runs only FAIL_TO_PASS; PASS_TO_PASS not enforced",
              "results_file": RESULTS, "audited": [], "skipped": [],
              "suspected_false_positive": [], "p2p_green": []}
    n = 0
    for iid in targets:
        inst = insts.get(iid)
        wd = os.path.join(WORK, iid)
        if inst is None:
            report["skipped"].append({"id": iid, "why": "no metadata"}); continue
        p2p = inst.get("PASS_TO_PASS") or []
        if not os.path.isdir(wd):
            report["skipped"].append({"id": iid, "why": "no work-dir (cleaned)"}); continue
        if not p2p:
            report["skipped"].append({"id": iid, "why": "empty P2P"}); continue
        if len(p2p) > args.max_p2p:
            report["skipped"].append({"id": iid, "why": "P2P too large (%d>%d)" % (len(p2p), args.max_p2p)}); continue
        if args.limit and n >= args.limit:
            break
        n += 1
        t0 = time.time()
        try:
            res = tr.run_tests(wd, env_kind[iid], p2p, env_vars={},
                               repo=inst["repo"], timeout=args.timeout)
            rec = {"id": iid, "repo": inst["repo"], "n_p2p": len(p2p),
                   "ok": res["ok"], "exit": res["exit"], "tail": res["tail"],
                   "secs": round(time.time()-t0, 1)}
        except Exception as e:
            rec = {"id": iid, "repo": inst["repo"], "n_p2p": len(p2p),
                   "ok": None, "error": repr(e)[:200], "secs": round(time.time()-t0,1)}
        report["audited"].append(rec)
        bucket = "p2p_green" if rec.get("ok") else "suspected_false_positive"
        report[bucket].append(iid)
        print("%-42s p2p=%-3d %-7s %s" % (
            iid, len(p2p),
            "GREEN" if rec.get("ok") else ("FAIL" if rec.get("ok") is False else "ERR"),
            str(rec.get("tail") or rec.get("error"))[:70]), flush=True)

    json.dump(report, open(args.out, "w"), indent=2)
    print("\n=== summary ===")
    print("audited:", len(report["audited"]),
          "| P2P green:", len(report["p2p_green"]),
          "| SUSPECTED false positive:", len(report["suspected_false_positive"]),
          "| skipped:", len(report["skipped"]))
    if report["suspected_false_positive"]:
        print("SUSPECTED FALSE POSITIVES (need Docker confirm):", report["suspected_false_positive"])
    print("report ->", args.out)

if __name__ == "__main__":
    main()
