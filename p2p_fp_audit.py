#!/usr/bin/env python3
"""P2P false-POSITIVE audit via a two-state CAUSATION test.

The home scorer (swe_agent_v2.score) enforces only FAIL_TO_PASS, never
PASS_TO_PASS, so a model patch that fixes F2P but breaks a previously-passing
test is scored resolved=True while the authoritative Docker scorer would mark it
UNRESOLVED. But naively re-running P2P at home is NOT a valid detector: many P2P
tests fail at home for env reasons even at base commit (home/gold env
discrepancy), so a bare P2P failure would falsely flag correct solves.

Correct discriminator (per instance, in its surviving work-dir, keeping .venv):
  A) base + test_patch (NO model patch) -> run P2P
  B) base + test_patch + model_patch    -> run P2P
Verdict:
  genuine_regression (SUSPECTED false positive) : A green AND B fails
  env_discrepancy (NOT a scorer bug)            : A fails
  clean (true positive)                         : A green AND B green
Only genuine_regression is a false-positive candidate, to be CONFIRMED by the
authoritative Docker eval before any reclassification. Never modifies results;
restores each work-dir to clean base. No answer leakage: P2P is public scoring
metadata, executed at scoring time, never shown to the model.
"""
import json, os, sys, subprocess, time, argparse
sys.path.insert(0, os.path.expanduser("~/Code/LLMOS"))
import test_runner as tr
INSTS=os.path.expanduser("~/swe/instances_full300.json")
RES=os.path.expanduser("~/swe/results_full300.json")
WORK=os.path.expanduser("~/swe/work"); TRACES=os.path.expanduser("~/swe/traces_v2")
OUT=os.path.expanduser("~/swe/p2p_fp_audit_report.json")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ids",nargs="*",required=True)
    ap.add_argument("--timeout",type=int,default=150)
    ap.add_argument("--out",default=OUT)
    a=ap.parse_args()
    insts={i["instance_id"]:i for i in json.load(open(INSTS))}
    results={r["id"]:r for r in json.load(open(RES))}
    rep={"generated":time.strftime("%Y-%m-%d %H:%M:%S"),"method":"two-state causation (base+test vs base+test+model)","records":[]}
    for iid in a.ids:
        inst=insts.get(iid); wd=os.path.join(WORK,iid); trp=os.path.join(TRACES,iid+".patch")
        rec={"id":iid}
        if not inst or not os.path.isdir(wd): rec["verdict"]="skip:no-metadata/workdir"; rep["records"].append(rec); print(iid,rec["verdict"]); continue
        env_kind=(results.get(iid) or {}).get("env_kind") or "uv"
        venv=os.path.join(wd,".condaenv" if env_kind=="conda" else ".venv","bin","python")
        if not os.path.exists(venv): rec["verdict"]="skip:no-venv(env_incomplete)"; rep["records"].append(rec); print(iid,rec["verdict"]); continue
        if not os.path.isfile(trp): rec["verdict"]="skip:no-model-patch"; rep["records"].append(rec); print(iid,rec["verdict"]); continue
        p2p=inst.get("PASS_TO_PASS") or []
        if not p2p: rec["verdict"]="skip:empty-p2p"; rep["records"].append(rec); print(iid,rec["verdict"]); continue
        def run(c): return subprocess.run(c,shell=True,cwd=wd,capture_output=True,text=True)
        def p2p_run():
            r=tr.run_tests(wd,env_kind,p2p,env_vars={},repo=inst["repo"],timeout=a.timeout); return r["ok"],r["tail"]
        try:
            run("git checkout -- ."); run("rm -f _t.patch")
            open(os.path.join(wd,"_t.patch"),"w").write(inst["test_patch"])
            tp=run("git apply _t.patch").returncode
            okA,tailA=p2p_run()
            mp=run('git apply "%s"'%trp).returncode
            okB,tailB=p2p_run()
        finally:
            run("git checkout -- ."); run("rm -f _t.patch")
        if tp!=0: verdict="skip:test-patch-no-apply"
        elif okA and not okB and mp==0: verdict="GENUINE_REGRESSION(suspected FP)"
        elif not okA: verdict="env_discrepancy"
        elif okA and okB: verdict="clean(true positive)"
        else: verdict="inconclusive(model-patch-no-apply)" if mp!=0 else "inconclusive"
        rec.update({"verdict":verdict,"A_base+test":{"ok":okA,"tail":tailA},"B_scored":{"ok":okB,"tail":tailB},"n_p2p":len(p2p)})
        rep["records"].append(rec)
        print("%-40s %-32s A=%s B=%s"%(iid,verdict,okA,okB))
    json.dump(rep,open(a.out,"w"),indent=2)
    gen=[r["id"] for r in rep["records"] if r["verdict"].startswith("GENUINE")]
    print("\n=== GENUINE regressions (suspected FP, Docker-confirm):",gen or "NONE","===")
    print("report ->",a.out)
if __name__=="__main__": main()
