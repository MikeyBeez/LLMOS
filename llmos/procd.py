"""procd — the process supervisor for the hosted runtime.

Realizes the implementation plan's headline decision: one macOS process per agent.
The kernel is a supervisor that:
  - spawns each agent as a real child process (subprocess),
  - parks it immediately with SIGSTOP (scheduling is signalling),
  - SIGCONTs one agent at a time to run it (cooperative, single expensive CPU),
  - services its syscalls over a per-agent Unix domain socket, enforcing
    capabilities and writing the single-writer trace via the same kernel _commit,
  - reaps the process when it RETURNs.

Isolation, kill, preemption (SIGSTOP/SIGCONT) and process visibility all come from
macOS; we don't reimplement any of it.
"""
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys

from .kernel import Kernel, DEFAULT_CAPS
from .store import Store

REPO = os.path.expanduser("~/Code/LLMOS")


class ProcKernel:
    def __init__(self, store: Store, sock_dir: str, log=print):
        self.store = store
        self.sock_dir = sock_dir
        self.log = log
        self.k = Kernel(store, cpu=None, log=log)   # cpu unused: CPUs live in the agents

    def _sock_path(self, pid: int) -> str:
        return os.path.join(self.sock_dir, f"llmos-{pid}.sock")

    def run(self, goals, ollama=False, model="qwen2.5:latest", budget=16, caps=None):
        self.k.boot()
        env = dict(os.environ)
        env["PYTHONPATH"] = REPO   # shell-out hardening: never depend on inherited PATH/profile

        agents = []  # (pid, popen, server_socket, sock_path)
        try:
            # spawn every agent as a real child process, parked with SIGSTOP
            for goal in goals:
                pid = self.k.spawn(goal, capabilities=set(caps) if caps else set(DEFAULT_CAPS))
                sp = self._sock_path(pid)
                if os.path.exists(sp):
                    os.unlink(sp)
                srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                srv.bind(sp)
                srv.listen(1)
                srv.settimeout(60)

                cmd = [sys.executable, "-m", "llmos.agent_runner",
                       "--pid", str(pid), "--goal", goal, "--sock", sp, "--budget", str(budget)]
                if ollama:
                    cmd += ["--ollama", "--model", model]
                p = subprocess.Popen(cmd, env=env, cwd=REPO)
                os.kill(p.pid, signal.SIGSTOP)   # park at spawn
                self.log(f"[procd] spawned llmos pid={pid} as os process {p.pid} — SIGSTOP-parked")
                agents.append((pid, p, srv, sp))

            self.log(f"[procd] kernel ospid={os.getpid()} supervising {len(agents)} agent process(es)")

            # cooperative FIFO: wake one, run it to completion, reap, next
            for pid, p, srv, sp in agents:
                self.log(f"[sched] SIGCONT os process {p.pid} (llmos pid={pid})")
                os.kill(p.pid, signal.SIGCONT)
                conn, _ = srv.accept()
                self._service(conn, pid)
                conn.close()
                rc = p.wait()
                self.log(f"[procd] llmos pid={pid} (os {p.pid}) exited rc={rc}; reaped")
        finally:
            for pid, p, srv, sp in agents:
                try:
                    srv.close()
                except Exception:
                    pass
                if os.path.exists(sp):
                    os.unlink(sp)
                if p.poll() is None:      # still running on error -> don't leak a process
                    try:
                        p.kill()
                    except Exception:
                        pass
        self.log("[procd] all agents done; kernel halting")

    def _service(self, conn, pid):
        rfile = conn.makefile("r", encoding="utf-8")
        hello = json.loads(rfile.readline())
        self.log(f"[procd] agent hello: llmos pid={hello['pid']} os pid={hello['ospid']} "
                 f"(child of {hello['ppid']})")
        while True:
            line = rfile.readline()
            if not line:
                break
            msg = json.loads(line)
            if msg.get("t") == "step":
                result, done = self.k.commit_external(hello["pid"], msg["op"], msg["args"])
                conn.sendall((json.dumps({"t": "result", "result": result, "done": done}) + "\n").encode())
                if done:
                    break
