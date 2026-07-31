"""Judge a patch in the official SWE-bench docker image.

    python3 referee.py django__django-15061            # every archived failure
    python3 referee.py django__django-15061 --tree     # the live work tree

WHY THIS EXISTS, AND WHY IT REFUSES THINGS.

Our harness is both the thing being tested and the thing reporting the result.
The official image is an independent judge: it applies the graded test patch
itself and runs the FAIL_TO_PASS labels with the environment SWE-bench intends.

Every run does three checks, in this order, and the first two are controls:

    base + tests, no fix  -> must FAIL   (the tests actually bite)
    gold + tests          -> must PASS   (the image is sane)
    candidate + tests     -> the question

If either control misbehaves the verdict is reported as inconclusive rather
than as a finding. Reading the gold patch here is post-hoc refereeing and never
happens at runtime.

DEFAULT SOURCE IS THE ARCHIVE, NOT THE WORK TREE. I refereed django-15061
against its live tree while a run was mid-way through attempt 4 on that same
instance, and "judged" a half-finished edit -- a trailing `subwidgets.` that no
submitted patch ever contained. --tree is still available because it is
sometimes what you want, but it refuses outright while a runner is executing.
"""
import argparse, glob, json, os, re, subprocess, sys, tempfile
import pandas as pd

SWE = "/home/bard/swe"
PATCHES = os.path.join(SWE, "runs", "ornith", "patches")


def image_for(iid):
    # django__django-15061 -> swebench/sweb.eval.x86_64.django_1776_django-15061
    return "swebench/sweb.eval.x86_64.%s:latest" % iid.replace("__", "_1776_")


def labels_for(inst):
    v = inst.FAIL_TO_PASS
    ids = json.loads(v) if isinstance(v, str) else list(v)
    out = []
    for nid in ids:
        if "::" in nid:                       # pytest style
            out.append(nid)
            continue
        m = re.search(r"\(([\w.]+)\)", nid)    # django "test_x (a.b.C)"
        out.append("%s.%s" % (m.group(1), nid.split(" ")[0]) if m else nid)
    return out


def runner_cmd(repo, labels):
    if repo == "django/django":
        return "./tests/runtests.py --parallel 1 " + " ".join(labels)
    return "python -m pytest -rA -q " + " ".join(labels)


def a_run_is_executing():
    return bool(subprocess.run(["pgrep", "-f", "run_list.py"],
                               capture_output=True, text=True).stdout.strip())


def judge(iid, inst, name, patch, image, labels):
    body = inst.test_patch + ("\n" + patch if (patch or "").strip() else "")
    with tempfile.NamedTemporaryFile("w", suffix=".patch", delete=False) as f:
        f.write(body)
        pf = f.name
    os.chmod(pf, 0o644)
    inner = ("cd /testbed && git checkout -- . && git apply /tmp/p.patch && "
             + runner_cmd(inst.repo, labels) + " 2>&1 | tail -16")
    try:
        out = subprocess.run(["docker", "run", "--rm",
                              "-v", pf + ":/tmp/p.patch", image,
                              "bash", "-lc", inner],
                             capture_output=True, text=True, timeout=1800)
        txt = (out.stdout or "") + (out.stderr or "")
    except subprocess.SubprocessError as e:
        txt = "docker failed: %s" % e
    finally:
        os.unlink(pf)
    ok = bool(re.search(r"^OK\b", txt, re.M)) or bool(
        re.search(r"^\d+ passed[^\n]*$", txt, re.M))
    print("-" * 70)
    print("%-34s -> %s" % (name, "PASS" if ok else "FAIL"))
    for line in txt.strip().splitlines()[-6:]:
        print("    " + line[:110])
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("instance")
    ap.add_argument("--tree", action="store_true",
                    help="judge the live work tree instead of the archive")
    a = ap.parse_args()

    df = pd.read_parquet(os.path.join(SWE, "lite_full.parquet"))
    m = df[df.instance_id == a.instance]
    if m.empty:
        sys.exit("unknown instance: %s" % a.instance)
    inst = m.iloc[0]
    image, labels = image_for(a.instance), labels_for(inst)
    print("instance : %s" % a.instance)
    print("image    : %s" % image)
    print("F2P      : %d label(s)\n" % len(labels))

    base = judge(a.instance, inst, "1. base, no fix (must FAIL)", "", image, labels)
    gold = judge(a.instance, inst, "2. GOLD (must PASS)", inst.patch, image, labels)
    if base or not gold:
        sys.exit("\nINCONCLUSIVE: controls misbehaved (base=%s gold=%s). "
                 "Nothing below would mean anything." % (base, gold))
    print("\ncontrols OK: fails without a fix, passes with gold.\n")

    cands = []
    if a.tree:
        if a_run_is_executing():
            sys.exit("REFUSING --tree: a runner is executing, so the work tree "
                     "is being mutated. Judge the archive instead.")
        wt = os.path.join(SWE, "work", a.instance)
        d = subprocess.run("git diff -- . ':(exclude)tests'", shell=True,
                           cwd=wt, capture_output=True, text=True).stdout or ""
        cands = [("live work tree", d, {})]
    else:
        for p in sorted(glob.glob(os.path.join(PATCHES,
                                               a.instance + ".attempt*.patch"))):
            meta = {}
            try:
                meta = json.load(open(p[:-6] + ".json"))
            except Exception:
                pass
            cands.append((os.path.basename(p), open(p).read(), meta))
    if not cands:
        sys.exit("no archived patches for %s (KEEP_FAILED archives them on the "
                 "next run)" % a.instance)

    print("judging %d candidate(s)\n" % len(cands))
    for name, patch, meta in cands:
        ok = judge(a.instance, inst, "3. %s" % name, patch, image, labels)
        ours = meta.get("score_tail", "?")
        print("    our harness said: %s | docker says: %s%s"
              % (ours, "PASS" if ok else "FAIL",
                 "   <-- DISAGREEMENT" if ok else ""))


if __name__ == "__main__":
    main()
