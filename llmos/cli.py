"""LLMOS shell — the (conversational-in-spirit) command surface.

    python3 -m llmos.cli run [goal]                 in-process run (deterministic MockCPU)
    python3 -m llmos.cli run "<goal>" --ollama       in-process run with a real local model
    python3 -m llmos.cli runp <goal> [<goal> ...]    multi-process run: one macOS process per agent
    python3 -m llmos.cli ps                          list known processes
    python3 -m llmos.cli replay <pid>                reconstruct a run's state from its trace
    python3 -m llmos.cli bench [--ollama --model M --iters N]   benchmark and collect metrics
    python3 -m llmos.cli stats                       aggregate the metrics table (where time goes)
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import tempfile
import time

from .authority import PolicyAuthority
from .cpu import MockCPU, OllamaCPU
from .kernel import Kernel
from .programs import PROGRAM_CAPS, PROGRAMS
from .replay import replay
from .store import Store

DB = os.path.expanduser("~/Code/LLMOS/state/llmos.db")
STATE_DIR = os.path.expanduser("~/Code/LLMOS/state")
CONTROL_SOCK = os.path.join(STATE_DIR, "llmos-control.sock")

_QUIET = lambda *a: None


def _kernel(cpu=None, authority=None):
    os.makedirs(STATE_DIR, exist_ok=True)
    store = Store(DB)
    return Kernel(store, cpu or MockCPU(PROGRAMS), authority=authority), store


def _dump_mem(store):
    keys = store.mem_list("mem")
    if keys:
        print("[mem] contents:")
        for k in keys:
            print(f"       mem/{k} = {store.mem_read('mem', k)}")


def cmd_run(args):
    cpu = OllamaCPU(model=args.model, host=args.host, log=print) if args.ollama else None
    authority = PolicyAuthority(grant=set(args.grant)) if args.grant else None
    kernel, store = _kernel(cpu, authority)
    if args.ollama:
        print(f"[cpu] OllamaCPU model={args.model}")
    if args.grant:
        print(f"[authority] PolicyAuthority granting on request: {sorted(set(args.grant))}")
    kernel.boot()
    pid = kernel.spawn(args.goal, capabilities=PROGRAM_CAPS.get(args.goal), budget=args.budget)
    kernel.run()
    pcb = kernel.procs[pid]
    print(f"\n[done] pid={pid} status={pcb.status.value} result={pcb.result}")
    _dump_mem(store)
    print(f"[hint] python3 -m llmos.cli replay {pid}")
    store.close()


def cmd_runp(args):
    from .procd import ProcKernel
    os.makedirs(STATE_DIR, exist_ok=True)
    store = Store(DB)
    pk = ProcKernel(store, sock_dir=STATE_DIR)
    pk.run(args.goals, ollama=args.ollama, model=args.model, budget=args.budget)
    print()
    _dump_mem(store)
    print("[hint] python3 -m llmos.cli ps")
    store.close()


def cmd_ps(args):
    _, store = _kernel()
    procs = store.list_processes()
    if not procs:
        print("(no processes yet — try: python3 -m llmos.cli run hello)")
    for p in procs:
        print(f"pid={p['pid']:>3} status={p['status']:<8} pc={p['pc']:<3} goal={p['goal']!r}")
    store.close()


def cmd_replay(args):
    _, store = _kernel()
    tmp = tempfile.mktemp(suffix=".db")
    ok, n, applied, diffs = replay(store, args.pid, tmp)
    if os.path.exists(tmp):
        os.unlink(tmp)
    print(f"[replay] pid={args.pid}: {n} instructions, {len(applied)} memory write(s) re-applied")
    print(f"[replay] state reproduced from trace: {'OK' if ok else 'MISMATCH ' + str(diffs)}")
    store.close()


# --- metrics: benchmark + aggregate ------------------------------------
def _pct(xs, p):
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def _print_stats(rows, label):
    if not rows:
        print(f"[stats] {label}: no metrics recorded")
        return
    cpu = [r["cpu_ms"] for r in rows if r["cpu_ms"] is not None]
    com = [r["commit_ms"] for r in rows if r["commit_ms"] is not None]
    n = len(rows)
    runs = len({r["pid"] for r in rows})
    tot_cpu, tot_com = sum(cpu), sum(com)
    total = tot_cpu + tot_com or 1e-9
    et = sum((r["eval_tokens"] or 0) for r in rows)
    pt = sum((r["prompt_tokens"] or 0) for r in rows)
    ev_ms = sum((r["eval_ms"] or 0) for r in rows)
    ld_ms = sum((r["load_ms"] or 0) for r in rows)
    retries = sum((r["retries"] or 0) for r in rows)
    faults = sum((r["fault"] or 0) for r in rows)
    print(f"\n=== {label} ===")
    print(f"instructions : {n}    runs: {runs}    instr/run: {n / runs:.1f}")
    print(f"cpu step ms  : mean {statistics.mean(cpu):8.2f}  p50 {_pct(cpu, .5):8.2f}  "
          f"p95 {_pct(cpu, .95):8.2f}  max {max(cpu):8.2f}")
    print(f"commit ms    : mean {statistics.mean(com):8.3f}  p50 {_pct(com, .5):8.3f}  "
          f"max {max(com):8.3f}")
    print(f"time split   : CPU(inference) {tot_cpu / total * 100:5.1f}%   "
          f"kernel(commit) {tot_com / total * 100:5.1f}%")
    print(f"wall total   : {total / 1000:.2f}s   per run: {total / runs / 1000:.2f}s")
    if et:
        gen_s = et / (ev_ms / 1000) if ev_ms else 0.0
        print(f"tokens       : generated {et}  prompt {pt}   gen speed {gen_s:.1f} tok/s")
        if ld_ms:
            print(f"model load   : {ld_ms / 1000:.2f}s total cold-load (should be ~0 when warm)")
    print(f"retries      : {retries}    faults: {faults}")


def cmd_bench(args):
    if args.goals:
        goals = args.goals
    elif args.ollama:
        goals = ["get the current time and save it to memory"]
    else:
        goals = ["hello", "ping", "readgood", "elevate"]
    grant = {"mem.write"} if "elevate" in goals else set()
    t_start = time.time()
    print(f"[bench] {'OllamaCPU ' + args.model if args.ollama else 'MockCPU (deterministic)'}  "
          f"goals={goals}  iters={args.iters}")
    for i in range(args.iters):
        for g in goals:
            cpu = OllamaCPU(model=args.model, host=args.host, log=(print if args.verbose else _QUIET)) if args.ollama else MockCPU(PROGRAMS)
            authority = PolicyAuthority(grant=grant) if grant else None
            kernel, store = _kernel(cpu, authority)
            kernel.log = print if args.verbose else _QUIET
            kernel.boot()
            pid = kernel.spawn(g, capabilities=PROGRAM_CAPS.get(g), budget=args.budget)
            kernel.run()
            store.close()
        print(f"[bench] iteration {i + 1}/{args.iters} done")
    store = Store(DB)
    rows = [r for r in store.metrics_rows() if r["ts"] >= t_start]
    label = f"{'OllamaCPU ' + args.model if args.ollama else 'MockCPU'} — this run"
    _print_stats(rows, label)
    store.close()


def cmd_stats(args):
    store = Store(DB)
    rows = store.metrics_rows(cpu_type=args.cpu)
    if not rows:
        print("[stats] no metrics yet — run:  python3 -m llmos.cli bench")
        store.close()
        return
    groups: dict = {}
    for r in rows:
        key = (r["cpu_type"], r["model"])
        groups.setdefault(key, []).append(r)
    print(f"[stats] {len(rows)} instructions across {len(groups)} CPU configuration(s)")
    for (ctype, model), grp in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        _print_stats(grp, f"{ctype}{(' / ' + model) if model else ''}  (all-time)")
    store.close()


def cmd_submit(args):
    """Submit a job to a running kerneld and watch its trace stream back live."""
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        s.connect(CONTROL_SOCK)
    except (FileNotFoundError, ConnectionRefusedError):
        print("kerneld is not running. Start it with:  PYTHONPATH=~/Code/LLMOS python3 -m llmos.kerneld")
        return
    s.sendall((json.dumps({"t": "submit", "goal": args.goal,
                           "budget": args.budget, "grant": list(args.grant or [])}) + "\n").encode())
    rfile = s.makefile("r", encoding="utf-8")
    for line in rfile:
        msg = json.loads(line)
        if msg["t"] == "log":
            print(msg["line"])
        elif msg["t"] == "done":
            print(f"\n[done] pid={msg['pid']} status={msg['status']} result={msg['result']}")
            break
        elif msg["t"] == "error":
            print(f"[error] {msg['error']}")
            break
    s.close()


def cmd_shutdown(args):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        s.connect(CONTROL_SOCK)
    except (FileNotFoundError, ConnectionRefusedError):
        print("kerneld is not running.")
        return
    s.sendall((json.dumps({"t": "shutdown"}) + "\n").encode())
    print("[shutdown] sent to kerneld")
    s.close()


def main():
    ap = argparse.ArgumentParser(prog="llmos")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="in-process run of a goal")
    r.add_argument("goal", nargs="?", default="hello")
    r.add_argument("--ollama", action="store_true", help="use a real local model as the CPU")
    r.add_argument("--model", default="ornith:35b", help="ollama model tag")
    r.add_argument("--host", default="http://127.0.0.1:11435", help="ollama host URL")
    r.add_argument("--budget", type=int, default=32, help="max instruction cycles")
    r.add_argument("--grant", nargs="*", default=[], help="capabilities a PolicyAuthority will grant on request")
    r.set_defaults(fn=cmd_run)

    rp = sub.add_parser("runp", help="multi-process run: one macOS process per agent")
    rp.add_argument("goals", nargs="+", help="one or more goals; each runs as its own OS process")
    rp.add_argument("--ollama", action="store_true")
    rp.add_argument("--model", default="ornith:35b")
    rp.add_argument("--budget", type=int, default=16)
    rp.set_defaults(fn=cmd_runp)

    p = sub.add_parser("ps", help="list processes")
    p.set_defaults(fn=cmd_ps)

    rpl = sub.add_parser("replay", help="reconstruct a run from its trace")
    rpl.add_argument("pid", type=int)
    rpl.set_defaults(fn=cmd_replay)

    b = sub.add_parser("bench", help="benchmark runs and collect timing/token metrics")
    b.add_argument("goals", nargs="*", help="goals to run (default: a deterministic mix, or one ollama goal)")
    b.add_argument("--ollama", action="store_true", help="benchmark a real model as the CPU")
    b.add_argument("--model", default="ornith:35b")
    b.add_argument("--host", default="http://127.0.0.1:11435", help="ollama host URL")
    b.add_argument("--iters", type=int, default=1, help="how many times to run each goal")
    b.add_argument("--budget", type=int, default=32)
    b.add_argument("--verbose", action="store_true", help="show per-instruction kernel log")
    b.set_defaults(fn=cmd_bench)

    st = sub.add_parser("stats", help="aggregate the metrics table")
    st.add_argument("--cpu", default=None, help="filter by cpu_type (MockCPU / OllamaCPU)")
    st.set_defaults(fn=cmd_stats)

    sm = sub.add_parser("submit", help="submit a job to a running kerneld and watch it live")
    sm.add_argument("goal")
    sm.add_argument("--budget", type=int, default=32)
    sm.add_argument("--grant", nargs="*", default=[])
    sm.set_defaults(fn=cmd_submit)

    sd = sub.add_parser("shutdown", help="ask a running kerneld to halt")
    sd.set_defaults(fn=cmd_shutdown)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
