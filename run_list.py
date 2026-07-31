"""LAGUNA REGRESSION (Mikey, 2026-07-23): run every instance we have EVER
resolved, under the new model, same harness, same protocol.

"Then run all the passing examples. Let's see if we can still pass those.
You may need to start and stop a few times if the model responds too
differently."

Design points:
  - The passing set is built from ALL results history (runs/*/results*.json,
    results*.json, workshop iter*.json, graduated.json): any instance with
    resolved=true at least once. Frozen to runs/laguna/compare_set.json on
    first build so restarts see the same set.
  - Same config as workshop.py: TTS-2 (two draws at temperature 1.0, break on
    resolved), form_rank on GIVEN evidence chooses, same env vars, same
    python pinning. Different model is the ONLY variable.
  - RESUMABLE: results dump after every instance; done ids are skipped, so
    stop/fix/restart loses at most the in-flight instance.
  - Answer leakage rules unchanged: nothing here reads gold/test patches.

usage: run_passing.py            (build set if absent, run/resume)
       run_passing.py --rebuild  (rebuild the passing set first)
writes ~/swe/runs/laguna/compare1.json
"""
import json, os, sys, glob, time, socket, subprocess

SWE = os.path.expanduser("~/swe")
LDIR = os.path.join(SWE, "runs", "ornith")
os.makedirs(LDIR, exist_ok=True)
SETF = os.path.join(LDIR, "compare_set.json")
OUT = os.path.join(LDIR, os.environ.get("OUT_NAME","regress18.json"))
# Independent attempts per instance, selected by form_rank. This is the
# test-time-scaling lever: DeepSWE reports 42.2% at k=1 and 57.9% at k=8
# on the same 32B model. Our own history says the same thing -- 45% ever
# resolved against ~15% single-pass.
MAX_ATTEMPTS = int(os.environ.get("MAX_ATTEMPTS", "1"))
# how many FAILED patches to retain per instance for post-hoc refereeing
KEEP_FAILED = int(os.environ.get("KEEP_FAILED", "3"))
MODEL_TAG = "ornith-1.0-35b-llamacpp"

insts = {i["instance_id"]: i
         for i in json.load(open(os.path.join(SWE, "instances_full300.json")))}


def build_set():
    ever = set()
    srcs = (glob.glob(os.path.join(SWE, "runs", "*", "results*.json"))
            + glob.glob(os.path.join(SWE, "results*.json"))
            + glob.glob(os.path.join(SWE, "runs", "workshop", "iter*.json")))
    for p in srcs:
        try:
            d = json.load(open(p))
        except Exception:
            continue
        if not isinstance(d, list):
            continue
        for x in d:
            if isinstance(x, dict) and x.get("id") and x.get("resolved"):
                ever.add(x["id"])
    try:
        for g in json.load(open(os.path.join(SWE, "runs", "workshop",
                                             "graduated.json"))):
            if isinstance(g, dict) and g.get("id"):
                ever.add(g["id"])
    except Exception:
        pass
    ids = sorted(i for i in ever if i in insts)
    json.dump(ids, open(SETF, "w"), indent=1)
    print("passing set built: %d instances -> %s" % (len(ids), SETF))
    return ids


# run_list.py: same engine, explicit instance list from $IDS (comma separated).
# Everything else -- config, resume, leak rules -- is unchanged from
# run_regress18.py, so results are directly comparable.
ids = [x.strip() for x in os.environ.get("IDS", "").split(",") if x.strip()]
missing = [i for i in ids if i not in insts]
if not ids:
    raise SystemExit("set IDS=inst1,inst2,...")
if missing:
    raise SystemExit("unknown instance ids: %s" % missing)
print("explicit list: %d instances" % len(ids))


def busy():
    pats = "[w]orkshop.py|[r]erun_failed4.py|[r]un300_v3.py"
    return bool(subprocess.run(["pgrep", "-f", pats],
                               capture_output=True, text=True).stdout.strip())


while busy():
    print("busy: waiting", flush=True)
    time.sleep(180)

sys.path.insert(0, os.path.expanduser("~/Code/LLMOS"))
os.environ.pop("DISABLE_KB", None)
os.environ["USE_SPEC_ENV"] = "1"
os.environ["TOOL_BUDGET_RECENT"] = "6000"
os.environ["TOOL_BUDGET_OLD"] = "1200"
os.environ["TOOL_RECENT_TURNS"] = "16"
os.environ["TRIAGE"] = "1"
os.environ["ATLAS_DIR"] = os.path.expanduser("~/swe/atlas")
import swe_agent_v2 as A

CANON = json.load(open(os.path.join(SWE, "canonical_python.json")))
try:
    results = json.load(open(OUT))
except Exception:
    results = []
done = {r["id"] for r in results}
print("resume: %d of %d already done" % (len(done), len(ids)), flush=True)


def form_rank(r):
    _g = r.get("given_tests_ok")
    given = ((8 if _g is True else (-8 if _g is False else 0))
             + int(r.get("syntax_ok", True) is not False) * 4
             + int((r.get("patch_bytes") or 0) > 0))
    self_authored = (int(bool(r.get("fix_verified_by_model"))) * 0.4
                     + int(bool(r.get("repro_green"))) * 0.2
                     + int(bool(r.get("probe_green"))) * 0.1)
    return given + self_authored


def wait_net():
    while True:
        try:
            socket.create_connection(("pypi.org", 443), timeout=10).close()
            return
        except OSError:
            time.sleep(60)


for iid in [i for i in ids if i not in done]:
    inst = insts[iid]
    pin = CANON.get(iid)
    if pin:
        os.environ["PIN_PYTHON"] = pin
    else:
        os.environ.pop("PIN_PYTHON", None)
    if pin in ("3.6", "3.7"):
        os.environ["PIN_BACKEND"] = "conda"
    else:
        os.environ.pop("PIN_BACKEND", None)
    attempts = []
    _base_budget = A.FIX_BUDGET
    for k in range(MAX_ATTEMPTS):
        wait_net()
        A.FIX_BUDGET = _base_budget if k == 0 else A.RETRY_FIX_BUDGET
        t0 = time.time()
        try:
            r = A.run_one(inst)
        except Exception as e:
            r = {"id": iid, "resolved": False, "error": str(e)}
        r["attempt"] = k + 1
        r["attempt_secs"] = round(time.time() - t0)
        # Keep the diff of each FAILED attempt -- a failure is the artifact
        # worth putting in front of an independent judge, and the next
        # attempt's revert destroys it. Successes are already archived in
        # traces/. Retain the last KEEP_FAILED per instance, pruned by mtime.
        if not r.get("resolved"):
            try:
                _pdir = os.path.join(LDIR, "patches")
                os.makedirs(_pdir, exist_ok=True)
                _wt = os.path.join(SWE, "work", iid)
                # tests excluded: the referee applies the graded test patch
                # itself, so including them here would double-apply.
                _d = subprocess.run("git diff -- . ':(exclude)tests'",
                                    shell=True, cwd=_wt, capture_output=True,
                                    text=True, timeout=120).stdout or ""
                _stem = os.path.join(_pdir, "%s.attempt%d" % (iid, k + 1))
                with open(_stem + ".patch", "w") as _fh:
                    _fh.write(_d)
                # a bare diff does not say what it scored
                with open(_stem + ".json", "w") as _fh:
                    json.dump({k2: r.get(k2) for k2 in
                               ("id", "attempt", "secs", "score_tail",
                                "syntax_ok", "seen_red", "repro_green",
                                "given_tests_ok", "patch_bytes")},
                              _fh, indent=1)
                r["patch_saved"] = len(_d)
                # prune: keep the newest KEEP_FAILED for this instance
                _mine = sorted(
                    (f for f in os.listdir(_pdir)
                     if f.startswith(iid + ".attempt") and f.endswith(".patch")),
                    key=lambda f: os.path.getmtime(os.path.join(_pdir, f)),
                    reverse=True)
                for _old in _mine[KEEP_FAILED:]:
                    for _ext in (".patch", ".json"):
                        try:
                            os.remove(os.path.join(
                                _pdir, _old[:-6] + _ext))
                        except OSError:
                            pass
            except Exception as _e:
                r["patch_saved"] = "error: %s" % _e
        attempts.append(r)
        if r.get("resolved"):
            break
    A.FIX_BUDGET = _base_budget
    chosen = dict(max(attempts, key=form_rank))
    chosen["attempts_made"] = len(attempts)
    chosen["model"] = MODEL_TAG
    results.append(chosen)
    json.dump(results, open(OUT, "w"), indent=2)
    n_res = sum(1 for x in results if x.get("resolved"))
    print("[%d/%d] %s -> %s   (running: %d/%d resolved)"
          % (len(results), len(ids), iid,
             "RESOLVED" if chosen.get("resolved") else "miss",
             n_res, len(results)), flush=True)

n_res = sum(1 for x in results if x.get("resolved"))
print("DONE: %d/%d still pass under %s" % (n_res, len(results), MODEL_TAG))
