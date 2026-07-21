"""Find magic string keys in our own code that don't exist in the real data.

Every analysis error today was a bare string key that failed silently:
  .get("patch")         -> field is gold_patch      -> 100% "no patch"
  _clip_result(...)     -> function never existed   -> NameError, swallowed
  .get("result")/"exit" -> events carry "ok"        -> "0 of 1039" nonsense
  outcome["probe_status"] -> written AFTER the save -> "the probe is dead"

`.get()` on a schema you did not verify turns a typo into confident wrong data.
`[key]` would have raised on the first one.

This checks the harness's string keys against the ACTUAL shapes on disk, so a
key nobody has verified shows up before it produces a wrong conclusion.
"""
import json, os, re, glob, collections

LL = os.path.expanduser("~/Code/LLMOS")

# --- the real shapes, sampled from disk -----------------------------------
shapes = {}
inst = json.load(open(os.path.expanduser("~/swe/instances_full300.json")))[0]
shapes["instance"] = set(inst)

tr = sorted(glob.glob(os.path.expanduser("~/swe/traces_v2/archive/*.trace.json")),
            key=os.path.getmtime)[-1]
t = json.load(open(tr))
shapes["trace"] = set(t)
shapes["outcome"] = set(t.get("outcome") or {})
shapes["fix_state"] = set(t.get("fix_state") or {})
ev = (t.get("phase2_events") or [{}])[0]
shapes["event"] = set(ev)

print("REAL SHAPES ON DISK")
for k, v in shapes.items():
    print("  %-10s %s" % (k, sorted(v)))
print()

# --- what our code asks for ------------------------------------------------
known = set().union(*shapes.values())
asked = collections.Counter()
where = collections.defaultdict(set)
for path in glob.glob(os.path.join(LL, "*.py")):
    src = open(path, encoding="utf-8", errors="ignore").read()
    for m in re.finditer(r'\.get\(\s*["\']([a-z_][a-z0-9_]{2,30})["\']', src):
        asked[m.group(1)] += 1
        where[m.group(1)].add(os.path.basename(path))

# keys that look like they address our data structures but appear in none of them
SUSPECT = [k for k in asked
           if k not in known
           and any(w in k for w in ("patch", "test", "resolv", "probe", "repro",
                                    "state", "outcome", "event", "result", "exit",
                                    "instance", "commit", "fail", "pass", "gold"))]
print("KEYS OUR CODE ASKS FOR THAT ARE IN NO OBSERVED SHAPE")
if not SUSPECT:
    print("  (none)")
for k in sorted(SUSPECT, key=lambda k: -asked[k]):
    print("  %-24s asked %2dx  in %s" % (k, asked[k], ", ".join(sorted(where[k]))[:60]))
print()
print("note: some are legitimately optional or belong to other dicts -- this is a")
print("      list to CHECK, not a list of bugs. The point is that nothing checked")
print("      them before, so a typo silently became a finding.")
