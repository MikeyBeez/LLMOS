#!/usr/bin/env python3
"""
reclaim_false_negatives.py  --  SWE-bench home-harness scoring correction.

WHY: Some repos (matplotlib, astropy, ...) run pytest with filterwarnings=error.
When the home harness installs a too-new pure-python dep (e.g. pyparsing 3.3),
a DeprecationWarning becomes a FATAL pytest COLLECTION error ("found no
collectors"). The model's patch may be perfectly correct, yet the FAIL_TO_PASS
tests never collect, so the instance is scored UNRESOLVED -> a FALSE NEGATIVE.

The AUTHORITATIVE scorer is the SWE-bench Docker eval
(swebench.harness.run_evaluation). This tool applies a *scoring-layer*
correction: given a manifest of instances that the Docker eval CONFIRMED the
model already resolves, it flips those records from unresolved->resolved and
emits a corrected results file with a provenance trail.

SAFETY / ANSWER-LEAKAGE:
  * Correction happens AFTER scoring, keyed only by public instance id and an
    external Docker audit artifact. No gold_patch / test_patch / FAIL_TO_PASS
    content is fed to the model. No instance-derived knowledge is injected back
    into the same instance. General/scoring-layer only.
  * Reclaims ONLY manifest entries with docker_confirmed == true.
  * NEVER overwrites the input results file. Refuses if --out == --results.
  * If the live benchmark runner is active, --out must differ from the live
    results (enforced), because the runner rewrites the whole results file on
    every instance and would clobber an in-place edit.

USAGE:
  reclaim_false_negatives.py [--results PATH] [--manifest PATH] [--out PATH]
                             [--dry-run]
Defaults: results=~/swe/results_full300.json
          manifest=~/Code/LLMOS/swe_false_negatives.json
          out=<results>.corrected.json
"""
import argparse, json, os, sys
from collections import defaultdict

def repo_of(iid): return iid.rsplit("-", 1)[0] if iid else "?"

def load(p): return json.load(open(os.path.expanduser(p)))

def metrics(recs):
    total = len(recs)
    resolved = sum(1 for r in recs if r.get("resolved"))
    return resolved, total

def by_repo(recs):
    d = defaultdict(lambda: [0, 0])
    for r in recs:
        rp = repo_of(r.get("id", ""))
        d[rp][0] += 1
        d[rp][1] += 1 if r.get("resolved") else 0
    return d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="~/swe/results_full300.json")
    ap.add_argument("--manifest", default="~/Code/LLMOS/swe_false_negatives.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    results_path = os.path.expanduser(a.results)
    out_path = os.path.expanduser(a.out) if a.out else results_path + ".corrected.json"
    if os.path.abspath(out_path) == os.path.abspath(results_path):
        sys.exit("REFUSING: --out must differ from --results (never overwrite the live results file).")

    recs = load(results_path)
    if not isinstance(recs, list):
        sys.exit("Unexpected results shape: expected a JSON list of records.")
    man = load(a.manifest)
    fns = man.get("confirmed_false_negatives", [])
    by_id = {r.get("id"): r for r in recs}

    before = metrics(recs)
    actions, pending, missing = [], [], []
    reclaimed = 0
    for e in fns:
        iid = e["id"]
        if not e.get("docker_confirmed"):
            if iid in by_id and not by_id[iid].get("resolved"):
                pending.append(iid)
            continue
        rec = by_id.get(iid)
        if rec is None:
            missing.append(iid); continue
        if rec.get("resolved"):
            actions.append(f"  already-resolved  {iid}")
            continue
        rec["resolved"] = True
        rec["_reclaimed"] = {
            "from": "unresolved",
            "reason": "docker_confirmed_false_negative",
            "evidence": e.get("evidence"),
            "root_cause": e.get("root_cause"),
        }
        reclaimed += 1
        actions.append(f"  RECLAIMED         {iid}")

    after = metrics(recs)
    print(f"results : {results_path}")
    print(f"manifest: {os.path.expanduser(a.manifest)}")
    print(f"score   : {before[0]}/{before[1]}  ->  {after[0]}/{after[1]}   (+{reclaimed} reclaimed)")
    print("actions:")
    for x in actions: print(x)
    if pending:
        print("PENDING (docker_confirmed=false, NOT reclaimed; needs Docker eval):")
        for x in pending: print(f"  pending           {x}")
    if missing:
        print("NOT-YET-SCORED (in manifest but absent from results; will apply on a later run):")
        for x in missing: print(f"  not-scored        {x}")
    # show affected repos
    br = by_repo(recs)
    aff = sorted({repo_of(e["id"]) for e in fns})
    print("affected repos (corrected resolved/total):")
    for rp in aff:
        t, s = br[rp]
        print(f"  {rp:28s} {s}/{t}")

    if a.dry_run:
        print("\n[dry-run] no file written.")
        return
    json.dump(recs, open(out_path, "w"), indent=2)
    print(f"\nwrote corrected results -> {out_path}")

if __name__ == "__main__":
    main()
