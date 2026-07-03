"""The LLMOS kernel — a small, deterministic orchestration loop.

It owns every resource, fetches each instruction from the CPU, enforces the
syscall boundary, schedules processes, and writes the trace. No model lives in
here, so the kernel stays reproducible: it is the boring, auditable part.

The cycle is the classic fetch-decode-execute-commit:
  fetch    ask the CPU for the next instruction (decode happens inside the CPU,
           which reads the process's context window)
  execute  run the opcode; anything touching the world goes through syscall()
  commit   append to the trace, update the window and the PCB, advance the PC
"""
from __future__ import annotations

import os

from .isa import Instruction, Op
from .pcb import PCB, Status
from .scheduler import Scheduler
from .syscall import SyscallTable, CapabilityError

_STATE = os.path.expanduser("~/Code/LLMOS/state")
_EXAMPLES = os.path.expanduser("~/Code/LLMOS/examples")
_DEFAULT_FS_POLICY = {
    "allowed": [os.path.join(_EXAMPLES, "trusted"), os.path.join(_EXAMPLES, "untrusted")],
    "untrusted": [os.path.join(_EXAMPLES, "untrusted")],
}

DEFAULT_CAPS = {"dev.clock", "mem.read", "mem.write", "fs.read"}
# capabilities a process loses the instant untrusted data enters its window
PRIVILEGED_CAPS = {"mem.write", "spawn"}


class Kernel:
    def __init__(self, store, cpu, log=print, fs_policy=None):
        self.store = store
        self.cpu = cpu
        self.sys = SyscallTable(store, fs_policy=fs_policy or _DEFAULT_FS_POLICY)
        self.sched = Scheduler()
        self.procs: dict[int, PCB] = {}
        self._next_pid = 1
        self.log = log

    # --- boot -----------------------------------------------------------
    def boot(self, boot_rom_keys: tuple = ()) -> None:
        self.log("[boot] mounting store:", self.store.path)
        # pids continue monotonically across runs so traces never collide
        existing = [p["pid"] for p in self.store.list_processes()]
        self._next_pid = (max(existing) + 1) if existing else 1
        missing = [k for k in boot_rom_keys if self.store.mem_read("boot", k) is None]
        if missing:
            self.log("[boot] WARN missing boot-ROM keys:", missing)
        self.log(f"[boot] scheduler up; next pid={self._next_pid}; kernel ready")

    # --- process lifecycle ----------------------------------------------
    def spawn(self, goal: str, capabilities=None, ppid: int | None = None, budget: int = 32) -> int:
        pid = self._next_pid
        self._next_pid += 1
        caps = set(capabilities) if capabilities is not None else set(DEFAULT_CAPS)
        pcb = PCB(pid=pid, goal=goal, ppid=ppid, capabilities=caps, budget=budget, status=Status.READY)
        self.procs[pid] = pcb
        self.store.save_process(pcb.to_dict())
        self.sched.add(pid)
        self.log(f"[spawn] pid={pid} goal={goal!r} caps={sorted(caps)}")
        return pid

    # --- the syscall channel (in-process now; a socket later) -----------
    def syscall(self, pcb, name: str, args: dict):
        return self.sys.dispatch(pcb, name, args)

    def commit_external(self, pid: int, op_str: str, args: dict):
        """Commit one instruction that arrived from an out-of-process agent over a
        socket. Reuses _commit, so the capability check, syscall dispatch, and the
        single-writer trace are identical to the in-process path. Returns
        (result, done)."""
        pcb = self.procs[pid]
        instr = Instruction(Op(op_str), args or {})
        done = self._commit(pcb, instr)
        self.store.save_process(pcb.to_dict())   # keep the process snapshot fresh for ps
        return pcb.context[-1]["result"], done

    def _apply_taint(self, pcb: PCB) -> None:
        """Prompt-injection defense: once untrusted data enters a process's window,
        the kernel revokes its privileged capabilities, so whatever action injected
        text tries to take is denied at the boundary — not left to the model."""
        if not pcb.tainted:
            dropped = sorted(pcb.capabilities & PRIVILEGED_CAPS)
            pcb.capabilities -= PRIVILEGED_CAPS
            pcb.tainted = True
            self.log(f"[security] pid={pcb.pid} ingested untrusted data -> revoked caps {dropped}")

    # --- the main loop --------------------------------------------------
    def run(self) -> None:
        while self.sched.has_work():
            pid = self.sched.next()
            self._run_slice(self.procs[pid])
        self.log("[kernel] no ready processes; run loop idle")

    def _run_slice(self, pcb: PCB) -> None:
        pcb.status = Status.RUNNING
        while True:
            if pcb.budget <= 0:
                self.log(f"[sched] pid={pcb.pid} budget exhausted -> preempt")
                pcb.status = Status.YIELDED
                self.sched.add(pcb.pid)
                break
            instr = self.cpu.step(pcb)          # FETCH (+ DECODE in the CPU)
            done = self._commit(pcb, instr)      # EXECUTE + COMMIT
            pcb.budget -= 1
            self.store.save_process(pcb.to_dict())
            if done:
                break
            if instr.op == Op.YIELD:
                pcb.status = Status.YIELDED
                self.sched.add(pcb.pid)
                self.log(f"[sched] pid={pcb.pid} yielded")
                break

    def _commit(self, pcb: PCB, instr) -> bool:
        """Execute one instruction, enforce capabilities, write the trace.
        Returns True when the process has finished."""
        op = instr.op
        args = instr.args or {}
        result = None
        done = False
        try:
            if op == Op.PLAN:
                result = {"plan": args.get("text", "")}
            elif op == Op.CALL:
                result = self.syscall(pcb, args["name"], args.get("args", {}))
            elif op == Op.READ_MEM:
                result = self.syscall(pcb, "mem.read", {"ns": args.get("ns", "mem"), "key": args["key"]})
                pcb.working_set.append(args["key"])
            elif op == Op.WRITE_MEM:
                result = self.syscall(pcb, "mem.write",
                                      {"ns": args.get("ns", "mem"), "key": args["key"], "value": args.get("value")})
            elif op == Op.SPAWN:
                child = self.spawn(args["goal"], args.get("capabilities"), ppid=pcb.pid)
                result = {"spawned": child}
            elif op == Op.YIELD:
                result = {"yield": True}
            elif op == Op.RETURN:
                pcb.result = args.get("result")
                pcb.status = Status.DONE
                result = {"return": pcb.result}
                done = True
            else:
                raise CapabilityError(f"illegal instruction: {op}")
        except CapabilityError as e:
            result = {"error": str(e)}
            self.log(f"[fault] pid={pcb.pid} {e}")
        except KeyError as e:
            result = {"error": f"malformed instruction, missing arg {e}"}
            self.log(f"[fault] pid={pcb.pid} malformed {op.value}: missing {e}")

        # security: ingesting untrusted data drops this process's privileged caps
        if isinstance(result, dict) and result.get("provenance") == "untrusted":
            self._apply_taint(pcb)

        # commit: single-writer trace, then window + PCB, then advance the PC
        self.store.trace_append(pcb.pid, pcb.pc, op.value, args, result)
        pcb.context.append({"pc": pcb.pc, "op": op.value, "args": args, "result": result})
        pcb.pc += 1
        self.log(f"[exec] pid={pcb.pid} pc={pcb.pc - 1} {op.value} -> {result}")
        return done
