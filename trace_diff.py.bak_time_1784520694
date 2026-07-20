#!/usr/bin/env python3
"""Compare two runs of the same instance.

Saving traces is half of it; the point is the diff. When a re-run flips a solve
into a miss, the question is always "what did it do differently, and where did
the two runs stop agreeing" -- and until now that was unanswerable, because the
re-run had already overwritten the solve.

Sources every surviving version of an instance's trace:
    traces_v2/<iid>.trace.json                     the live one
    traces_v2/archive/<iid>__<ts>__<tag>.trace.json  preserved before overwrite
    traces_snapshots/<ts>/<iid>.trace.json           bulk snapshots

usage:
    trace_diff.py <instance_id>              list every surviving version
    trace_diff.py <instance_id> 0 2          diff version 0 against version 2
    trace_diff.py --regressions              scan for solves that became misses
"""
import json, os, sys, glob, collections

TR = os.path.expanduser("~/swe/traces_v2")
SNAP = os.path.expanduser("~/swe/traces_snapshots")


def versions(iid):
    """Every surviving trace for this instance, oldest first."""
    out = []
    for p in sorted(glob.glob(os.path.join(TR, "archive", iid + "__*.trace.json"))):
        out.append(("archive", p))
    for p in sorted(glob.glob(os.path.join(SNAP, "*", iid + ".trace.json"))):
        out.append(("snapshot", p))
    live = os.path.join(TR, iid + ".trace.json")
    if os.path.isfile(live):
        out.append(("live", live))
    out.sort(key=lambda x: os.path.getmtime(x[1]))
    # drop byte-identical duplicates (a snapshot of an unchanged live file)
    seen, uniq = set(), []
    for kind, p in out:
        sz = os.path.getsize(p)
        key = (sz, open(p, "rb").read(4096)[:200])
        if key in seen:
            continue
        seen.add(key)
        uniq.append((kind, p))
    return uniq


def summarize(p):
    t = json.load(open(p))
    tools = [e.get("tool") for e in (t.get("phase2_events") or [])]
    meta = t.get("phase2_meta") or []
    return {
        "resolved": bool((t.get("outcome") or {}).get("resolved")),
        "turns": len(meta),
        "tools": tools,
        "hist": collections.Counter(tools),
        "prompt_max": max([m.get("prompt_tokens") or 0 for m in meta] or [0]),
        "clipped": sum(m.get("clipped_chars") or 0 for m in meta),
        "mtime": os.path.getmtime(p),
        "blob": t,
    }


def show(iid):
    vs = versions(iid)
    if not vs:
        print("no surviving traces for %s" % iid); return
    import time
    print("%s -- %d surviving version(s)\n" % (iid, len(vs)))
    for i, (kind, p) in enumerate(vs):
        try:
            s = summarize(p)
        except Exception as e:
            print("  [%d] %-8s UNREADABLE (%s)" % (i, kind, e)); continue
        print("  [%d] %-8s %s  %-8s turns=%-3d prompt_max=%-6d clipped=%d"
              % (i, kind, time.strftime("%m-%d %H:%M", time.localtime(s["mtime"])),
                 "RESOLVED" if s["resolved"] else "miss", s["turns"],
                 s["prompt_max"], s["clipped"]))
    print("\n  diff two with: trace_diff.py %s <i> <j>" % iid)


def diff(iid, i, j):
    vs = versions(iid)
    a, b = summarize(vs[i][1]), summarize(vs[j][1])
    print("=" * 70)
    print("%s   [%d] %s   vs   [%d] %s" % (iid, i, vs[i][0], j, vs[j][0]))
    print("=" * 70)
    print("%-16s %-24s %s" % ("", "A", "B"))
    for k in ("resolved", "turns", "prompt_max", "clipped"):
        flag = "   <-- differs" if a[k] != b[k] else ""
        print("%-16s %-24s %s%s" % (k, a[k], b[k], flag))

    print("\ntool usage:")
    keys = sorted(set(a["hist"]) | set(b["hist"]))
    for k in keys:
        d = b["hist"][k] - a["hist"][k]
        print("  %-14s A=%-4d B=%-4d %s" % (k, a["hist"][k], b["hist"][k],
                                            ("%+d" % d) if d else ""))

    print("\nfirst divergence in the tool sequence:")
    for n, (x, y) in enumerate(zip(a["tools"], b["tools"])):
        if x != y:
            lo = max(0, n - 2)
            print("  turn %d: A did %r, B did %r" % (n, x, y))
            print("    A ...%s" % a["tools"][lo:n + 3])
            print("    B ...%s" % b["tools"][lo:n + 3])
            break
    else:
        print("  identical for the first %d turns (A has %d, B has %d)"
              % (min(len(a["tools"]), len(b["tools"])), len(a["tools"]), len(b["tools"])))

    ta = (a["blob"].get("outcome") or {}).get("score_tail") or ""
    tb = (b["blob"].get("outcome") or {}).get("score_tail") or ""
    if ta != tb:
        print("\nscore tail:")
        print("  A: %s" % ta[-160:].replace("\n", " "))
        print("  B: %s" % tb[-160:].replace("\n", " "))


def scan_regressions():
    """Any instance whose surviving versions go RESOLVED -> miss."""
    seen = set()
    for p in glob.glob(os.path.join(TR, "*.trace.json")):
        seen.add(os.path.basename(p).replace(".trace.json", ""))
    for p in glob.glob(os.path.join(TR, "archive", "*.trace.json")):
        seen.add(os.path.basename(p).split("__2026")[0])
    hits = 0
    for iid in sorted(seen):
        vs = versions(iid)
        if len(vs) < 2:
            continue
        try:
            st = [summarize(p)["resolved"] for _, p in vs]
        except Exception:
            continue
        if any(st[k] and not st[k + 1] for k in range(len(st) - 1)):
            print("  REGRESSED %-42s %s" % (iid, ["OK" if x else "miss" for x in st]))
            hits += 1
    print("\n%d instance(s) went from solved to missed across surviving traces" % hits)


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--regressions":
        scan_regressions()
    elif len(sys.argv) == 2:
        show(sys.argv[1])
    elif len(sys.argv) == 4:
        diff(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
    else:
        print(__doc__)
