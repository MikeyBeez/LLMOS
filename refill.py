"""Refill the failset from the HARD tail (Mikey: concentrate on the hard ones).

Four of seven graduations were instances that had passed before and failed once
-- flaky, not hard. They resolved on first contact and taught us nothing, while
consuming workshop cycles. They got in because refill drew from "instances not
yet touched", which is mostly easy ones.

Correct pool: the 122 instances that have NEVER resolved in any run. Ordered by
miss count, so the hardest come first -- an instance that has beaten us eleven
times has more to teach than one that has beaten us twice.

Also reports repo concentration, because 8 of the 12 hardest are sympy: that is
one wall, not eight problems, and it is worth knowing before spending twelve
iterations discovering it one instance at a time.

usage: refill.py [n]        (default 4; --show to inspect without writing)
"""
import json, os, sys, glob, collections

W = os.path.expanduser("~/swe/runs/workshop")
N = 4
SHOW = "--show" in sys.argv
for a in sys.argv[1:]:
    if a.isdigit():
        N = int(a)

hist = collections.defaultdict(list)
for p in (glob.glob(os.path.expanduser("~/swe/runs/*/results*.json"))
          + glob.glob(os.path.expanduser("~/swe/results*.json"))
          + glob.glob(os.path.join(W, "iter*.json"))):
    try:
        d = json.load(open(p))
    except Exception:
        continue
    if not isinstance(d, list):
        continue
    for x in d:
        if isinstance(x, dict) and x.get("id"):
            hist[x["id"]].append(bool(x.get("resolved")))

grad = {g["id"] for g in json.load(open(os.path.join(W, "graduated.json")))}
park = {p["id"] for p in json.load(open(os.path.join(W, "parked.json")))}
cur = json.load(open(os.path.join(W, "failset.json")))

never = [i for i, h in hist.items() if h and not any(h)]
pool = [i for i in never if i not in grad and i not in park and i not in cur]
pool.sort(key=lambda i: (-len(hist[i]), i))

print("hard tail: %d never-resolved | eligible after excluding "
      "graduated/parked/current: %d" % (len(never), len(pool)))
print()
repo = collections.Counter(i.split("__")[0] for i in pool)
print("repo concentration of the eligible hard tail:")
for r, c in repo.most_common(6):
    print("   %-16s %3d  %s" % (r, c, "#" * min(40, c)))
print()

pick = [i for i in cur] + pool[:max(0, N - len(cur))]
print("current failset (%d): %s" % (len(cur), cur))
print("adding (hardest first):")
for i in pool[:max(0, N - len(cur))]:
    print("   %2d prior misses  %s" % (len(hist[i]), i))

if SHOW:
    print("\n--show: nothing written")
else:
    json.dump(pick, open(os.path.join(W, "failset.json"), "w"))
    print("\nfailset now (%d): %s" % (len(pick), pick))
