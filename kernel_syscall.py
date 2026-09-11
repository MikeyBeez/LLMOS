"""LLMOS - the syscall dispatcher's capability gate.

ARCHITECTURE.md 4.7, 4.13, and item 4 of the section-8 build list. The CPU
(the model) cannot touch the world. It emits a syscall request; this module
decides whether that request runs, and says so in one place.

Three properties, and each one is here because a real system lost them.

ONE DOOR. check() is called from the single point in the agent loop where a
tool name and its arguments are known and nothing has executed yet. A
permission test spread across the handlers is a test one handler forgets.

DETERMINISTIC, AND NOT THE CPU. Nothing in here calls a model. It compares
strings and resolves paths. A checker that cannot be reasoned with cannot be
reasoned out of a decision, which is the whole point of putting it outside
the thing it checks.

THE TAPE. Every decision, allow or deny, is appended to a JSONL audit log
before the call runs. "What did it do" should be a query, not an
investigation.

MODES. LLMOS_CAPS=off | warn | enforce (default: warn).
  off      no checking, no tape. For bisecting.
  warn     check and record, but allow everything. Run a campaign in warn,
           read the denials that WOULD have happened, then flip.
  enforce  a denied syscall does not run.
warn exists so this can be switched on in front of a working agent without
betting the campaign on the allow-list being complete the first time.
"""

from __future__ import annotations

import fnmatch
import json
import os
import time
from dataclasses import dataclass, replace

MODES = ("off", "warn", "enforce")


def mode() -> str:
    m = (os.environ.get("LLMOS_CAPS") or "warn").strip().lower()
    return m if m in MODES else "warn"


# ---------------------------------------------------------------- capabilities

@dataclass(frozen=True)
class Capabilities:
    """What one process may do. Empty grants nothing -- that is deliberate.

    allow        tool names this process may call. fnmatch patterns are
                 allowed ("swe.*") so a capability set stays readable, but a
                 bare "*" is refused by the constructor: a set that grants
                 everything is not a capability set, it is the absence of one.
    write_roots  absolute directories a mutating tool may write under. A
                 mutating tool with no write_roots is denied.
    needs_human  tool names that always stop for a person, even when allowed.
    label        what this set is for, so a denial in the log is legible.
    """

    allow: frozenset = frozenset()
    write_roots: tuple = ()
    needs_human: frozenset = frozenset()
    label: str = "unnamed"

    def __post_init__(self):
        if "*" in self.allow:
            raise ValueError(
                "a capability set may not contain a bare '*'. Name the tools, "
                "or use a prefix pattern like 'swe.*'. A set that allows "
                "everything is how the wiki happened.")
        bad = [r for r in self.write_roots if not os.path.isabs(r)]
        if bad:
            raise ValueError(f"write_roots must be absolute paths: {bad}")
        object.__setattr__(self, "allow", frozenset(self.allow))
        object.__setattr__(self, "needs_human", frozenset(self.needs_human))
        object.__setattr__(
            self, "write_roots",
            tuple(os.path.realpath(r).rstrip(os.sep) for r in self.write_roots))

    def grants(self, tool: str) -> bool:
        if tool in self.allow:
            return True
        return any(fnmatch.fnmatchcase(tool, p) for p in self.allow if "*" in p)

    def with_write_root(self, path: str) -> "Capabilities":
        """A copy confined to one checkout. The kernel calls this per task."""
        return replace(self, write_roots=(os.path.realpath(path),))

    @classmethod
    def from_tools(cls, tools, write_roots=(), needs_human=(), label="phase"):
        """Derive the allow-list from the tool schemas the phase was handed.

        The capability set for a phase IS the set of tools that phase was
        given -- so there is no second list to maintain and no way for the two
        to drift apart. This is the same rule mutating_tool_names() follows:
        read the fact off the thing that already holds it.

        What it catches on day one, with nobody authoring anything: a call to
        a tool from a different phase, a hallucinated tool name, and a write
        that resolves outside the checkout.
        """
        names = set()
        for t in tools or ():
            if isinstance(t, dict):
                fn = t.get("function") if t.get("type") == "function" else None
                n = (fn or t).get("name")
                if n:
                    names.add(n)
        if not names:
            raise ValueError(
                "no tool names found in the schema list; refusing to build a "
                "capability set that would grant nothing by accident")
        return cls(allow=frozenset(names), write_roots=tuple(write_roots),
                   needs_human=frozenset(needs_human), label=label)


NOTHING = Capabilities(label="nothing")


# ------------------------------------------------------------------- verdicts

@dataclass(frozen=True)
class Verdict:
    allowed: bool
    rule: str          # which check decided, for the log and for tests
    reason: str = ""   # what the model is told, in words it can act on

    def as_tool_error(self) -> str:
        """The body handed back as the tool result when a call is refused.

        Said plainly, because a model that is told only 'denied' will retry
        the same call, and a retry loop burns the turn budget the task needs.
        """
        return json.dumps({
            "error": "DENIED by the kernel; NOTHING was executed",
            "rule": self.rule,
            "reason": self.reason,
            "hint": "This is not a transient failure and retrying the same "
                    "call will be denied again. Use a tool you do have, or "
                    "say plainly that the task needs a capability you were "
                    "not given.",
        })


# ------------------------------------------------------ the path-bearing args

# Argument names that name a file or directory. Derived from the SWE tool
# schemas; extend here rather than at a call site.
PATH_ARGS = ("path", "file", "filename", "file_path", "target", "dest",
             "destination", "dir", "directory", "src", "source")


def _paths_in(args) -> list:
    if not isinstance(args, dict):
        return []
    out = []
    for k, v in args.items():
        if k.lower() in PATH_ARGS and isinstance(v, str) and v.strip():
            out.append(v)
        elif k.lower() in ("paths", "files") and isinstance(v, (list, tuple)):
            out.extend(p for p in v if isinstance(p, str) and p.strip())
    return out


def _under(path: str, roots: tuple) -> bool:
    """True if path resolves inside one of roots.

    realpath first, so a symlink out of the checkout is caught. The check is
    on the resolved parent when the file does not exist yet, because a write
    creates it.
    """
    p = os.path.realpath(path if os.path.isabs(path) else os.path.abspath(path))
    for r in roots:
        if p == r or p.startswith(r + os.sep):
            return True
    return False


# --------------------------------------------------------------- the mutators

def _mutating_names() -> frozenset:
    """Which tools write to the checkout -- read off the code that writes.

    swe_fix_tools.mutating_tool_names() derives this by walking the handlers,
    and it raises rather than returning empty. Import it late so this module
    stays importable and testable on its own.
    """
    try:
        from swe_fix_tools import mutating_tool_names
        return frozenset(mutating_tool_names())
    except Exception:
        return frozenset()


_MUTATORS = None


def mutators() -> frozenset:
    global _MUTATORS
    if _MUTATORS is None:
        _MUTATORS = _mutating_names()
    return _MUTATORS


# ----------------------------------------------------------------- the tape

class Audit:
    """Append-only JSONL. One line per decision, written before the call runs.

    Best-effort by design: a failed write must never take down the agent, so
    every error here is swallowed. The tape is evidence, not a dependency.
    """

    def __init__(self, path=None):
        self.path = path or os.environ.get(
            "LLMOS_CAPS_LOG",
            os.path.join(os.path.expanduser("~"), ".llmos", "syscalls.jsonl"))

    def write(self, rec: dict) -> None:
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "a") as fh:
                fh.write(json.dumps(rec, default=str) + "\n")
        except Exception:
            pass


# ------------------------------------------------------------------- the gate

def check(tool: str, args, caps: Capabilities) -> Verdict:
    """Decide one syscall. Pure: no I/O except path resolution, no model.

    Order matters. The allow-list runs first because it is the cheapest and
    the most absolute; a tool that was never granted is refused before its
    arguments are examined at all.
    """
    tool = (tool or "").strip()
    if not tool:
        return Verdict(False, "no-tool", "empty tool name")

    if not caps.grants(tool):
        return Verdict(
            False, "not-in-allow-list",
            f"'{tool}' is not in the capability set '{caps.label}' for this "
            f"task. Granted: {sorted(caps.allow)}")

    if tool in caps.needs_human:
        return Verdict(
            False, "needs-human",
            f"'{tool}' always requires a person to approve it and no approval "
            f"is attached to this call")

    if tool in mutators():
        if not caps.write_roots:
            return Verdict(
                False, "no-write-root",
                f"'{tool}' writes to the checkout but this capability set "
                f"grants no write root")
        for p in _paths_in(args):
            if not _under(p, caps.write_roots):
                return Verdict(
                    False, "outside-write-root",
                    f"'{p}' resolves outside the write roots "
                    f"{list(caps.write_roots)}. Symlinks are resolved before "
                    f"this check.")

    return Verdict(True, "allowed")


# --------------------------------------------------------------- the one door

class Gate:
    """What the agent loop holds. check + tape + mode, in one object.

    Used at the single dispatch point:

        v = gate.admit(tool, args, turn=turn, task=inst)
        if not v.allowed:
            ...hand v.as_tool_error() back as the tool result; do not dispatch
    """

    def __init__(self, caps: Capabilities, audit: Audit = None, mode_=None):
        self.caps = caps
        self.audit = audit or Audit()
        self._mode = mode_ or mode()
        self.denied = []          # what WOULD have been denied, in warn mode

    def admit(self, tool, args, **ctx) -> Verdict:
        if self._mode == "off":
            return Verdict(True, "caps-off")

        v = check(tool, args, self.caps)

        self.audit.write({
            "ts": time.time(),
            "mode": self._mode,
            "caps": self.caps.label,
            "tool": tool,
            "args": args if isinstance(args, dict) else str(args)[:400],
            "allowed": v.allowed,
            "rule": v.rule,
            "reason": v.reason,
            "enforced": v.allowed or self._mode == "enforce",
            **ctx,
        })

        if not v.allowed:
            self.denied.append((tool, v.rule))
            if self._mode == "warn":
                # Recorded, not blocked. This is the dial that lets the gate
                # go in front of a working agent before the allow-list is
                # known to be complete.
                return Verdict(True, "warn-only", v.reason)
        return v
