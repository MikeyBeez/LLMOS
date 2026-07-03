"""The syscall dispatcher — the trust boundary.

The CPU never touches the world directly. It emits a syscall request; the kernel
validates it against the process's capabilities *here*, executes it via the right
device handler, and returns the result. Because this layer is deterministic and
sits outside the model, an untrusted/stochastic CPU cannot do anything the kernel
did not approve.

Device results may carry a "provenance" tag ('trusted' | 'untrusted'). The kernel
uses that tag to defend against prompt injection (see Kernel._apply_taint).
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any


class CapabilityError(Exception):
    """Raised when a process attempts a syscall it lacks the capability for, an
    unknown syscall (an illegal instruction), or a sandbox violation (e.g. reading
    outside the allowed filesystem roots)."""


class SyscallTable:
    """name -> (required_capability, handler). Handlers receive (pcb, args)."""

    def __init__(self, store, fs_policy=None):
        self.store = store
        # fs_policy: {"allowed": [roots...], "untrusted": [roots...]}
        self.fs_policy = fs_policy or {"allowed": [], "untrusted": []}
        self.table: dict[str, tuple[str, Any]] = {}
        self._register_builtins()

    def register(self, name: str, cap: str, handler) -> None:
        self.table[name] = (cap, handler)

    def _register_builtins(self) -> None:
        self.register("clock.now", "dev.clock", self._clock_now)
        self.register("mem.read", "mem.read", self._mem_read)
        self.register("mem.write", "mem.write", self._mem_write)
        self.register("fs.read", "fs.read", self._fs_read)

    def dispatch(self, pcb, name: str, args: dict) -> Any:
        if name not in self.table:
            raise CapabilityError(f"illegal syscall: {name!r}")
        cap, handler = self.table[name]
        if cap not in pcb.capabilities:
            raise CapabilityError(
                f"process {pcb.pid} lacks capability '{cap}' for syscall '{name}'"
            )
        return handler(pcb, args)

    # --- builtin devices -------------------------------------------------
    def _clock_now(self, pcb, args) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _mem_read(self, pcb, args) -> Any:
        return self.store.mem_read(args.get("ns", "mem"), args["key"])

    def _mem_write(self, pcb, args) -> dict:
        prov = "untrusted" if pcb.tainted else "trusted"
        self.store.mem_write(args.get("ns", "mem"), args["key"], args.get("value"), prov)
        return {"written": args["key"]}

    def _fs_read(self, pcb, args) -> dict:
        """Read a file, but only within the allowed roots (path-traversal safe),
        and tag the result with provenance so the kernel can defend downstream."""
        p = os.path.realpath(os.path.expanduser(args["path"]))
        allowed = [os.path.realpath(os.path.expanduser(a)) for a in self.fs_policy.get("allowed", [])]
        if not any(p == a or p.startswith(a + os.sep) for a in allowed):
            raise CapabilityError(f"fs.read denied: {p} is outside the allowed roots")
        try:
            with open(p, "r", encoding="utf-8") as f:
                content = f.read()
        except OSError as e:
            raise CapabilityError(f"fs.read failed: {e}")
        untrusted = [os.path.realpath(os.path.expanduser(u)) for u in self.fs_policy.get("untrusted", [])]
        prov = "untrusted" if any(p == u or p.startswith(u + os.sep) for u in untrusted) else "trusted"
        return {"content": content, "provenance": prov, "path": p}
