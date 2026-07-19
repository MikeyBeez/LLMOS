#!/usr/bin/env python3
"""research_ledger: the analysis corpus for the workshop's research zone.

Doctrine (Mikey, 2026-07-19): this is RESEARCH, not answer-key optimization.
Two zones, one fence:

  RESEARCH ZONE (~/swe/research/): may see everything -- gold patch, hidden
    test assertions, all our attempts across generations, verdicts. Purpose:
    find the RULES that define how we build representations and prompts.
    "Even if you don't put the information directly into the tool, you should
    be keeping all that stuff for analysis."
  RUNTIME ZONE (the harness): carries only the general rules distilled from
    research. NOTHING under ~/swe/research/ is ever read by swe_agent_v2,
    the tools, the KBs it injects, or the atlas. The fence is one line:
    runtime never opens this directory. (Enforced by convention + audit:
    grep -r "swe/research" ~/Code/LLMOS/ must hit only this file.)

Each record binds, for one worked instance: what the issue asked, what gold
did, what the hidden tests literally check, what we produced in each
generation, WHICH OF THE THREE QUESTIONS failed (goal / progress / format),
and the candidate general rule the failure suggests.

usage:
  research_ledger.py add <instance_id> [note about which question failed / rule candidate]
  research_ledger.py list
"""
import json
import os
import re
import sys
import glob
import time

RDIR = os.path.expanduser("~/swe/research")
os.makedirs(RDIR, exist_ok=True)
INSTANCES = os.path.expanduser("~/swe/instances_full300.json")


def gather(iid, note=""):
    insts = {i["instance_id"]: i for i in json.load(open(INSTANCES))}
    inst = insts[iid]
    rec = {"instance_id": iid, "repo": inst["repo"], "ts": int(time.time()),
           "analyst_note": note,
           "issue_head": (inst["problem_statement"] or "")[:1500],
           "gold_patch": inst["gold_patch"],
           "hidden_test_assertions": re.findall(
               r"^\+.*assert.*$", inst["test_patch"], re.M)[:40],
           "attempts": []}
    # every surviving version of our work on this instance, all generations
    cands = ([os.path.expanduser("~/swe/traces_v2/%s.patch" % iid)]
             + sorted(glob.glob(os.path.expanduser(
                 "~/swe/traces_v2/archive/%s__*.patch" % iid)))
             + sorted(glob.glob(os.path.expanduser(
                 "~/swe/traces_snapshots/*/%s.patch" % iid))))
    seen = set()
    for p in cands:
        if not os.path.isfile(p):
            continue
        try:
            txt = open(p, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        key = hash(txt)
        if key in seen or not txt.strip():
            continue
        seen.add(key)
        rec["attempts"].append({"source": os.path.basename(p),
                                "mtime": int(os.path.getmtime(p)),
                                "patch": txt[:4000]})
    out = os.path.join(RDIR, iid + ".json")
    json.dump(rec, open(out, "w"), indent=2)
    print("research record: %s (%d distinct attempt patches, %d hidden asserts)"
          % (out, len(rec["attempts"]), len(rec["hidden_test_assertions"])))


def list_records():
    for p in sorted(glob.glob(os.path.join(RDIR, "*.json"))):
        r = json.load(open(p))
        print("%-34s attempts=%d  note: %s"
              % (r["instance_id"], len(r.get("attempts", [])),
                 (r.get("analyst_note") or "")[:80]))


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "add":
        gather(sys.argv[2], " ".join(sys.argv[3:]))
    else:
        list_records()
