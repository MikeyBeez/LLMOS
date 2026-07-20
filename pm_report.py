"""Cross-run report over the post-mortem corpus.

The point of collecting is to see what one run cannot show you. This is the
query that would have surfaced the escape bug days ago, run against the
backfilled corpus.
"""
import json, glob, os, collections

recs = []
for f in glob.glob(os.path.expanduser("~/swe/research/postmortem/*.jsonl")):
    for line in open(f):
        line = line.strip()
        if line:
            try:
                recs.append(json.loads(line))
            except Exception:
                pass

R = [r for r in recs if r.get("resolved")]
M = [r for r in recs if not r.get("resolved")]
print("post-mortem corpus: %d runs (%d resolved, %d missed)" % (len(recs), len(R), len(M)))
print()

def s(rs, path):
    t = 0
    for r in rs:
        d = r
        for k in path.split("."):
            d = (d or {}).get(k) if isinstance(d, dict) else None
        t += d or 0
    return t

print("%-26s %10s %10s" % ("", "RESOLVED", "MISSED"))
for lbl, p in [("patch attempts", "patches.attempts"),
               ("  of which failed", "patches.failed"),
               ("  repeated anchors", "patches.repeated"),
               ("  unobserved gaps", "patches.unobserved_gaps"),
               ("  escape anchors", "patches.escape_anchors"),
               ("  escape anchors FAILED", "patches.escape_anchors_failed"),
               ("turns", "turns")]:
    a = s(R, p) / max(1, len(R))
    b = s(M, p) / max(1, len(M))
    print("%-26s %10.1f %10.1f" % (lbl, a, b))

et, ef = s(recs, "patches.escape_anchors"), s(recs, "patches.escape_anchors_failed")
at, af = s(recs, "patches.attempts"), s(recs, "patches.failed")
print()
print("THE SIGNAL NOBODY WAS LOOKING AT:")
print("  anchors containing a backslash : %d, failed %d (%.0f%%)"
      % (et, ef, 100.0 * ef / max(1, et)))
print("  all other anchors              : %d, failed %d (%.0f%%)"
      % (at - et, af - ef, 100.0 * (af - ef) / max(1, at - et)))
print()

print("MOST REPEATED ERRORS ACROSS ALL RUNS:")
errs = collections.Counter()
for r in recs:
    for msg, n in (r.get("top_errors") or []):
        errs[msg[:88]] += n
for msg, n in errs.most_common(6):
    print("  %4d  %s" % (n, msg))
print()

print("HOW RUNS ENDED:")
ends = collections.Counter((("RESOLVED" if r.get("resolved") else "MISSED"),
                            r.get("ended_by")) for r in recs)
for (o, e), n in ends.most_common(6):
    print("  %-9s %-10s %4d" % (o, e, n))
