#!/usr/bin/env python3
"""
swe_fn_triage.py -- single source of truth for SWE-bench home-harness miss triage.

WHY THIS EXISTS
---------------
Every overnight cycle re-implemented an inline "score_tail signature triage" to
answer one recurring question: "are there any NEW false-negative (FN) families in
the results, or is every miss either a catalogued env-FN or a genuine
declared-wrong miss?"  That ad-hoc regex repeatedly MISSED families (django
unittest "FAILED (failures=N)" summaries and the pytest<6 bare-"rootdir:" tell
were both invisible to it), which stalled the loop more than once.  This tool
centralises the taxonomy so it is written down, count-reconciled, and versioned
instead of being re-derived (differently) each cycle.

WHAT IT DOES
------------
Classifies every completed instance in results_full300.json into exactly one
bucket, reconciles the buckets against the total (asserts nothing is dropped),
cross-references the Docker-confirmed FN manifest (swe_false_negatives.json) to
show the reclaim backlog, and -- most importantly -- surfaces a REVIEW bucket:
any miss whose score_tail looks like an env/collection anomaly but does NOT match
a catalogued family.  A non-empty REVIEW bucket is the "possible new FN family"
radar the loop keeps needing.

ANSWER-LEAKAGE SAFETY
---------------------
Reads only: (a) results_full300.json score_tail (the model's OWN test-run tail)
and public flags (resolved / env_ok / id), and (b) swe_false_negatives.json
(public instance ids + external Docker evidence strings).  It NEVER reads or
emits gold_patch / test_patch / FAIL_TO_PASS / PASS_TO_PASS content, does not
execute anything, and derives no instance-specific knowledge that could be fed
back to the model.  It is a read-only reporting tool; it changes no state.

USAGE
-----
  python3 swe_fn_triage.py                       # human report
  python3 swe_fn_triage.py --json                # machine-readable buckets
  python3 swe_fn_triage.py --results <p> --manifest <p>
"""
import argparse
import collections
import json
import re
import sys

DEFAULT_RESULTS = "/home/bard/swe/results_full300.json"
DEFAULT_MANIFEST = "/home/bard/Code/LLMOS/swe_false_negatives.json"

# ---------------------------------------------------------------------------
# score_tail -> compact, digit/path-agnostic signature (last meaningful line)
# ---------------------------------------------------------------------------
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def normalize(score_tail):
    if not score_tail:
        return "<empty>"
    t = _ANSI.sub("", score_tail)
    lines = [ln for ln in t.splitlines() if ln.strip()]
    if not lines:
        return "<empty>"
    last = lines[-1].strip()
    last = re.sub(r"\d+\.\d+\s*s", "Xs", last)
    last = re.sub(r"\b\d+\b", "N", last)
    last = re.sub(r"/\S+", "<path>", last)
    return last


def repo_of(instance_id):
    # "django__django-15738" -> "django"; "scikit-learn__scikit-learn-10297" -> "scikit-learn"
    return instance_id.split("__", 1)[0]


# ---------------------------------------------------------------------------
# Family rule table.  First matching rule wins.  A rule is:
#   name, kind, repo ("*" = any), predicate(row, sig) -> bool
# kind in {reclaimable_fn, env_fail, network_coupled, genuine_miss, review}
#   reclaimable_fn  : correct patch scored as miss due to env/collection; a Docker
#                     eval of the model's OWN patch can (deterministically) confirm.
#   env_fail        : env bootstrap never produced a runnable test env (total loss).
#   network_coupled : FN that needs offline network (httpbin) -> not Docker-confirmable.
#   genuine_miss    : healthy env, tests ran, patch is actually wrong (Issue #2).
#   review          : anomalous tell that is NOT a catalogued family -> inspect.
# ---------------------------------------------------------------------------
NO_COLLECTORS = re.compile(
    r"found no collectors|errors? during collection|no tests ran", re.I
)
COLLECT_ERROR = re.compile(r"warnings?,\s*N errors?\s*in", re.I)
BARE_ROOTDIR = re.compile(r"^rootdir:")
ONLY_SKIP = re.compile(r"^N skipped\b")
DJANGO_UNITTEST_FAIL = re.compile(r"^FAILED \((failures|errors|expected)")
# a bare "N error(s) in Xs" with no "failed" is a collection/setup error, not a
# test-assertion failure -> worth a human look (possible new collect-FN).
BARE_ERROR = re.compile(r"^N errors?\s+in\b")


def classify(row):
    rid = row["id"]
    repo = repo_of(rid)
    sig = normalize(row.get("score_tail", ""))

    if row.get("resolved"):
        return ("RESOLVED", "resolved", sig)
    if row.get("env_ok") is False:
        return ("ENV_BOOTSTRAP_FAIL", "env_fail", sig)

    # --- reclaimable env/collection false-negative families ---
    # NO_COLLECTORS is repo-AGNOSTIC on purpose: the warnings-as-errors ->
    # "found no collectors" mechanism is universal; matplotlib is merely where it
    # manifests today. A new repo hitting it will land here (not in REVIEW), but
    # the per-repo count in the report makes a new-repo occurrence visible.
    if NO_COLLECTORS.search(sig):
        return ("WARN_AS_ERROR_NO_COLLECTORS", "reclaimable_fn", sig)
    if repo == "pytest-dev" and BARE_ROOTDIR.search(sig):
        return ("PYTEST_BARE_ROOTDIR", "reclaimable_fn", sig)
    if repo == "sphinx-doc" and COLLECT_ERROR.search(sig):
        return ("SPHINX_COLLECT_ERR", "reclaimable_fn", sig)
    if repo == "scikit-learn" and ONLY_SKIP.search(sig):
        return ("SKLEARN_IMPORTORSKIP_SKIP", "reclaimable_fn", sig)

    # --- network-coupled FN (requests/httpbin): not Docker-confirmable offline ---
    if repo == "psf":
        return ("REQUESTS_NETWORK_COUPLED", "network_coupled", sig)

    # --- new-family radar: anomalous env/collection tells not yet catalogued ---
    if BARE_ERROR.search(sig) or sig == "<empty>":
        return ("REVIEW_POSSIBLE_ENV_ANOMALY", "review", sig)

    # --- genuine declared-wrong misses (Issue #2 verification frontier) ---
    if repo == "django" and DJANGO_UNITTEST_FAIL.search(sig):
        return ("DJANGO_UNITTEST_FAIL", "genuine_miss", sig)
    return ("GENUINE_MISS", "genuine_miss", sig)


def load_confirmed(manifest_path):
    try:
        m = json.load(open(manifest_path))
    except Exception:
        return set()
    out = set()
    for e in m.get("confirmed_false_negatives", []):
        if e.get("docker_confirmed"):
            out.add(e["id"])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=DEFAULT_RESULTS)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    data = json.load(open(args.results))
    confirmed = load_confirmed(args.manifest)
    row_by_id = {row["id"]: row for row in data}

    buckets = collections.defaultdict(list)   # family -> [ids]
    kind_of = {}                              # family -> kind
    sig_by_family = collections.defaultdict(collections.Counter)
    for row in data:
        fam, kind, sig = classify(row)
        buckets[fam].append(row["id"])
        kind_of[fam] = kind
        sig_by_family[fam][sig] += 1

    total = len(data)
    counted = sum(len(v) for v in buckets.values())
    assert counted == total, f"RECONCILE FAIL: {counted} != {total} (a row was dropped)"

    KIND_ORDER = ["resolved", "reclaimable_fn", "network_coupled",
                  "env_fail", "review", "genuine_miss"]
    fam_order = sorted(buckets, key=lambda f: (KIND_ORDER.index(kind_of[f]), -len(buckets[f])))

    if args.json:
        out = {
            "total": total,
            "families": {
                f: {
                    "kind": kind_of[f],
                    "count": len(buckets[f]),
                    "ids": sorted(buckets[f]),
                    "reclaimable_docker_confirmed": sorted(set(buckets[f]) & confirmed),
                    "reclaimable_pending": sorted(
                        set(buckets[f]) - confirmed) if kind_of[f] == "reclaimable_fn" else [],
                }
                for f in fam_order
            },
        }
        print(json.dumps(out, indent=2))
        return

    print(f"SWE-bench home-harness miss triage  (total completed: {total})")
    print(f"  results:  {args.results}")
    print(f"  manifest: {args.manifest}  (docker-confirmed FNs: {len(confirmed)})")
    print("=" * 72)
    reclaimable_total = pending_total = 0
    for f in fam_order:
        ids = buckets[f]
        kind = kind_of[f]
        line = f"{len(ids):3d}  {f:<28s} [{kind}]"
        if kind == "reclaimable_fn":
            conf = sorted(set(ids) & confirmed)
            pend = sorted(set(ids) - confirmed)
            reclaimable_total += len(ids)
            pending_total += len(pend)
            line += f"  docker_confirmed={len(conf)} pending={len(pend)}"
        print(line)
        # show the signature makeup for the non-resolved buckets (compact)
        if kind != "resolved":
            for sig, n in sig_by_family[f].most_common(4):
                print(f"        {n:3d} | {sig[:64]}")
        # informational only: self_verified split of the pending backlog. self_verified
        # is NOT a reclaim filter -- Docker-confirmed FNs 5227/25570 are self_verified=False.
        if kind == "reclaimable_fn":
            _pend = sorted(set(ids) - confirmed)
            if _pend:
                _svt = sum(1 for _i in _pend if row_by_id[_i].get("fix_verified_by_model"))
                print(f"        pending self_verified: True={_svt} False={len(_pend) - _svt}"
                      "  (informational; Docker authoritative, self_verified NOT a filter)")
    print("=" * 72)
    print(f"reclaimable_fn: {reclaimable_total} total, {pending_total} pending Docker confirm")
    review = [f for f in fam_order if kind_of[f] == "review"]
    if any(buckets[f] for f in review):
        print("\n!! REVIEW bucket non-empty -- inspect for a NEW FN family:")
        print("   A collection/env anomaly is a POSSIBLE scoring FN REGARDLESS of")
        print("   self_verified -- Docker eval is authoritative and self_verified is")
        print("   NOT a reliable filter: Docker-confirmed FNs pytest-dev__pytest-5227")
        print("   and scikit-learn__scikit-learn-25570 both have self_verified=False.")
        for f in review:
            for rid in sorted(buckets[f]):
                r = row_by_id[rid]
                sv = bool(r.get("fix_verified_by_model"))
                print(f"     {rid:34s} self_verified={sv}  "
                      f"phase2={r.get('phase2_reason')}  env_kind={r.get('env_kind')}")
    else:
        print("\nREVIEW bucket empty: no uncatalogued env/collection anomaly.")


if __name__ == "__main__":
    main()
