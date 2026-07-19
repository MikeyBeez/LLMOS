#!/usr/bin/env python3
"""shape_tally: measure the distribution of failure SHAPES across the research
zone (Mikey's taxonomy hypothesis -- gather/transform/format is a big slice).

Each research record may carry a "shape" tag (set via research_ledger add, or
appended to analyst_note as "shape=<x>"). Shapes:
  gather_format : pull values from >=2 places and cat into a required output
                  shape (table, string, json, message). flask-5063 is this.
  logic         : the algorithm/branch is wrong; a real reasoning bug.
  env           : environment/build/collection, not the code.
  theory        : model diagnosed a coherent WRONG cause (pytest-11148).
  format_word   : right substance, wrong vocabulary/label.
  unknown       : not yet classified.

Reports the tally so we can decide -- with data, not assertion -- how much
machinery the gather_format class earns. Asserts nothing; just counts.
"""
import glob
import json
import os
import re
import collections

RDIR = os.path.expanduser("~/swe/research")
SHAPES = ("gather_format", "logic", "env", "theory", "format_word", "unknown")


def shape_of(rec):
    s = rec.get("shape")
    if s:
        return s
    note = (rec.get("analyst_note") or "").lower()
    m = re.search(r"shape=(\w+)", note)
    if m:
        return m.group(1)
    # light keyword fallback so old records still count
    if any(k in note for k in ("format", "label", "column", "vocab", "word")):
        return "format_word"
    if "theory" in note or "wrong theory" in note:
        return "theory"
    if any(k in note for k in ("env", "install", "collect", "import")):
        return "env"
    return "unknown"


def main():
    tally = collections.Counter()
    rows = []
    for p in sorted(glob.glob(os.path.join(RDIR, "*.json"))):
        try:
            rec = json.load(open(p))
        except Exception:
            continue
        if "probe_bench" in os.path.basename(p):
            continue
        sh = shape_of(rec)
        tally[sh] += 1
        rows.append((os.path.basename(p)[:-5], sh))
    print("=== failure-shape distribution (research zone, n=%d) ===" % sum(tally.values()))
    for s in SHAPES:
        if tally[s]:
            print("  %-14s %d" % (s, tally[s]))
    other = sum(v for k, v in tally.items() if k not in SHAPES)
    if other:
        print("  (other tags: %s)" % {k: v for k, v in tally.items() if k not in SHAPES})
    print()
    for name, sh in rows:
        print("  %-34s %s" % (name, sh))
    print("\nNOTE: hypothesis (Mikey) is that gather_format dominates. Decide the "
          "report-spec build ONLY when n is large enough to see the share.")


if __name__ == "__main__":
    main()
