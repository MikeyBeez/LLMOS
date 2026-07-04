"""LLMOS shell — the (conversational-in-spirit) command surface.

    python3 -m llmos.cli run [goal]                 in-process run (deterministic MockCPU)
    python3 -m llmos.cli run "<goal>" --ollama       in-process run with a real local model
    python3 -m llmos.cli runp <goal> [<goal> ...]    multi-process run: one macOS process per agent
    python3 -m llmos.cli ps                          list known processes
    python3 -m llmos.cli replay <pid>                reconstruct a run's state from its trace
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import tempfile

from .authority import PolicyAuthority
from .cpu import MockCPU, OllamaCPU
from .kernel import Kernel
from .programs import PROGRAM_CAPS, PROGRAMS
from .replay import replay
from .store import Store

DB = os.path.expanduser("~/Code/LLMOS/state/llmos.db")
STATE_DIR = os.path.expanduser("~/Code/LLMOS/state")
CONTROL_SOCK = os.path.join(STATE_DIR, "llmos-control.sock")


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
    cpu = OllamaCPU(model=args.model, log=print) if args.ollama else None
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
    r.add_argument("--model", default="qwen2.5:latest", help="ollama model tag")
    r.add_argument("--budget", type=int, default=32, help="max instruction cycles")
    r.add_argument("--grant", nargs="*", default=[], help="capabilities a PolicyAuthority will grant on request")
    r.set_defaults(fn=cmd_run)

    rp = sub.add_parser("runp", help="multi-process run: one macOS process per agent")
    rp.add_argument("goals", nargs="+", help="one or more goals; each runs as its own OS process")
    rp.add_argument("--ollama", action="store_true")
    rp.add_argument("--model", default="qwen2.5:latest")
    rp.add_argument("--budget", type=int, default=16)
    rp.set_defaults(fn=cmd_runp)

    p = sub.add_parser("ps", help="list processes")
    p.set_defaults(fn=cmd_ps)

    rpl = sub.add_parser("replay", help="reconstruct a run from its trace")
    rpl.add_argument("pid", type=int)
    rpl.set_defaults(fn=cmd_replay)

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
