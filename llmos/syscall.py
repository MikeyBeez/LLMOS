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

import ast
import math
import operator
import os
import re
import subprocess
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
        self.register("calc", "dev.calc", self._calc)
        self.register("fs.write", "fs.write", self._fs_write)
        self.register("fs.list", "fs.read", self._fs_list)
        self.register("shell.exec", "shell.exec", self._shell_exec)

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

    # a deterministic calculator. Two failure modes only: the CPU presents the wrong
    # INPUT, or the program CALCULATES wrong. We remove input errors by understanding
    # quantity words (dozen, half a dozen, score) so the model passes phrases verbatim
    # instead of converting them by hand; the calculation is exact and PEMDAS-correct.
    _OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
            ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
            ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos}
    _FUNCS = {"sqrt": math.sqrt, "abs": abs, "round": round, "min": min, "max": max,
              "floor": math.floor, "ceil": math.ceil, "int": int, "float": float, "pow": pow}
    _WORDNUM = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
                "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
                "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
                "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
                "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
                "hundred": 100, "thousand": 1000}
    _UNITS = {"dozen": 12, "dozens": 12, "score": 20, "gross": 144, "pair": 2, "pairs": 2, "couple": 2}

    def _resolve_words(self, expr: str) -> str:
        """Turn English quantities into arithmetic before evaluation, so the model can
        pass the problem's own words and cannot mis-convert them."""
        e = expr.lower()
        e = re.sub(r"\bhalf a dozen\b|\bhalf dozen\b", "6", e)
        e = re.sub(r"\bhalf a gross\b", "72", e)
        e = re.sub(r"\bhalf a score\b", "10", e)
        for w, n in self._WORDNUM.items():
            e = re.sub(rf"\b{w}\b", str(n), e)
        e = re.sub(r"\b(?:a|an)\s+(dozen|dozens|score|gross|pair|pairs|couple)\b", r"1 \1", e)
        for u, m in self._UNITS.items():
            e = re.sub(rf"(\d+(?:\.\d+)?)\s+{u}\b", lambda mo, mm=m: f"({mo.group(1)}*{mm})", e)
            e = re.sub(rf"\b{u}\b", str(m), e)
        e = re.sub(r"\bhalf\b", "0.5", e)
        return e

    def _eval_node(self, node):
        if isinstance(node, ast.Expression):
            return self._eval_node(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in self._OPS:
            return self._OPS[type(node.op)](self._eval_node(node.left), self._eval_node(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in self._OPS:
            return self._OPS[type(node.op)](self._eval_node(node.operand))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in self._FUNCS:
            return self._FUNCS[node.func.id](*[self._eval_node(a) for a in node.args])
        raise ValueError("only numbers, + - * / // % **, and sqrt/abs/round/min/max are allowed")

    def _calc(self, pcb, args) -> dict:
        raw = str(args.get("expr", "")).strip()
        resolved = self._resolve_words(raw)
        try:
            val = self._eval_node(ast.parse(resolved, mode="eval"))
        except Exception as e:
            return {"expr": raw, "resolved": resolved, "error": f"could not evaluate: {e}"}
        if isinstance(val, float) and val.is_integer():
            val = int(val)
        out = {"expr": raw, "value": val}
        if resolved != raw.lower():
            out["resolved"] = resolved
        return out

    # --- world-touching devices: sandboxed to fs_policy roots, capability-gated ---
    def _within(self, path, roots):
        """Return the realpath if it is inside one of the roots, else None (sandbox)."""
        p = os.path.realpath(os.path.expanduser(str(path)))
        for r in roots:
            rr = os.path.realpath(os.path.expanduser(r))
            if p == rr or p.startswith(rr + os.sep):
                return p
        return None

    def _writable_roots(self):
        return self.fs_policy.get("writable") or self.fs_policy.get("allowed", [])

    def _fs_write(self, pcb, args) -> dict:
        p = self._within(args["path"], self._writable_roots())
        if p is None:
            raise CapabilityError(f"fs.write denied: {args['path']!r} is outside the writable roots")
        content = args.get("content", "")
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(content)
        return {"written": p, "bytes": len(content)}

    def _fs_list(self, pcb, args) -> dict:
        p = self._within(args.get("path", "."), self.fs_policy.get("allowed", []))
        if p is None:
            raise CapabilityError("fs.list denied: path is outside the allowed roots")
        if os.path.isdir(p):
            return {"path": p, "entries": sorted(os.listdir(p))}
        return {"path": p, "entries": [], "note": "not a directory"}

    def _shell_exec(self, pcb, args) -> dict:
        """Run a shell command with cwd sandboxed to the writable roots, a timeout,
        and truncated output. This is the most powerful device: it is NOT granted by
        default, it is in PRIVILEGED_CAPS (a tainted process loses it), and it only
        runs inside the allowed roots. Network/seccomp hardening is future work."""
        roots = self._writable_roots()
        default_cwd = roots[0] if roots else "."
        cwd = self._within(args.get("cwd", default_cwd), roots) if roots else None
        if cwd is None:
            raise CapabilityError("shell.exec denied: cwd is outside the allowed roots")
        timeout = min(int(args.get("timeout", 60)), 600)
        try:
            r = subprocess.run(args["cmd"], shell=True, cwd=cwd, capture_output=True,
                               text=True, timeout=timeout)
            return {"exit_code": r.returncode, "stdout": r.stdout[-4000:], "stderr": r.stderr[-2000:]}
        except subprocess.TimeoutExpired:
            return {"error": f"timed out after {timeout}s", "exit_code": 124}
