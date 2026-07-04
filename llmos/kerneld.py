"""kerneld — the persistent LLMOS kernel daemon.

Boots one kernel and keeps it running, listening on a control socket for job
submissions. This is the 'boot the OS, then submit work' model: the OS is a
long-lived process, not a one-shot CLI. Each job streams its instruction trace
back to the submitting client live — the spectator role from INTERACTION.md.

Run directly for testing:   PYTHONPATH=~/Code/LLMOS python3 -m llmos.kerneld
Or under launchd via bin/llmos-kernel (see launchd/).
"""
from __future__ import annotations

import json
import os
import signal
import socket

from .authority import DenyAuthority, PolicyAuthority
from .cpu import MockCPU
from .kernel import Kernel
from .programs import PROGRAM_CAPS, PROGRAMS
from .store import Store

STATE_DIR = os.path.expanduser("~/Code/LLMOS/state")
DB = os.path.join(STATE_DIR, "llmos.db")
CONTROL_SOCK = os.path.join(STATE_DIR, "llmos-control.sock")


class KernelDaemon:
    def __init__(self, sock_path: str = CONTROL_SOCK):
        os.makedirs(STATE_DIR, exist_ok=True)
        self.sock_path = sock_path
        self.store = Store(DB)
        self.kernel = Kernel(self.store, MockCPU(PROGRAMS), authority=DenyAuthority())
        self.running = True

    def serve(self):
        self.kernel.boot()
        if os.path.exists(self.sock_path):
            os.unlink(self.sock_path)
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(self.sock_path)
        srv.listen(4)
        srv.settimeout(1.0)
        print(f"[kerneld] up pid={os.getpid()} control={self.sock_path}", flush=True)

        def _stop(*_):
            self.running = False
        signal.signal(signal.SIGTERM, _stop)
        signal.signal(signal.SIGINT, _stop)

        try:
            while self.running:
                try:
                    conn, _ = srv.accept()
                except socket.timeout:
                    continue
                try:
                    self._handle(conn)
                finally:
                    conn.close()
        finally:
            srv.close()
            if os.path.exists(self.sock_path):
                os.unlink(self.sock_path)
            self.store.close()
            print("[kerneld] halted", flush=True)

    def _send(self, conn, obj):
        try:
            conn.sendall((json.dumps(obj) + "\n").encode())
        except OSError:
            pass   # client went away mid-job; the job still completes server-side

    def _handle(self, conn):
        rfile = conn.makefile("r", encoding="utf-8")
        line = rfile.readline()
        if not line:
            return
        msg = json.loads(line)
        t = msg.get("t")
        if t == "shutdown":
            self._send(conn, {"t": "bye"})
            self.running = False
        elif t == "submit":
            self._run_job(conn, msg)
        else:
            self._send(conn, {"t": "error", "error": f"unknown request {t!r}"})

    def _run_job(self, conn, msg):
        goal = msg["goal"]
        budget = msg.get("budget", 32)
        grant = msg.get("grant") or []
        orig_log = self.kernel.log
        orig_auth = self.kernel.authority
        if grant:
            self.kernel.authority = PolicyAuthority(grant=set(grant))
        # stream every kernel log line to the client — the live spectator window
        self.kernel.log = lambda *a: self._send(conn, {"t": "log", "line": " ".join(str(x) for x in a)})
        try:
            pid = self.kernel.spawn(goal, capabilities=PROGRAM_CAPS.get(goal), budget=budget)
            self.kernel.run()
            pcb = self.kernel.procs[pid]
            self._send(conn, {"t": "done", "pid": pid, "status": pcb.status.value, "result": pcb.result})
        finally:
            self.kernel.log = orig_log
            self.kernel.authority = orig_auth


def main():
    KernelDaemon().serve()


if __name__ == "__main__":
    main()
