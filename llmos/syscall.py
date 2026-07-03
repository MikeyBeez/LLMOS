"""The syscall dispatcher — the trust boundary.

The CPU never touches the world directly. It emits a syscall request; the kernel
validates it against the process's capabilities *here*, executes it via the right
device handler, and returns the result. Because this layer is deterministic and
sits outside the model, an untrusted/stochastic CPU cannot do anything the kernel
did not approve.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


class CapabilityError(Exception):
    """Raised when a process attempts a syscall it lacks the capability for, or an
    unknown syscall (an illegal instruction)."""


class SyscallTable:
    """name -> (required_capability, handler). Handlers receive (pcb, args)."""

    def __init__(self, store):
        self.store = store
        self.table: dict[str, tuple[str, Any]] = {}
        self._register_builtins()

    def register(self, name: str, cap: str, handler) -> None:
        self.table[name] = (cap, handler)

    def _register_builtins(self) -> None:
        self.register("clock.now", "dev.clock", self._clock_now)
        self.register("mem.read", "mem.read", self._mem_read)
        self.register("mem.write", "mem.write", self._mem_write)

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
        # provenance drops to 'untrusted' for sandboxed processes
        prov = "untrusted" if "untrusted" in pcb.capabilities else "trusted"
        self.store.mem_write(args.get("ns", "mem"), args["key"], args.get("value"), prov)
        return {"written": args["key"]}
