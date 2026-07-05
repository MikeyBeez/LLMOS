"""The LLMOS kernel — a small, deterministic orchestration loop.

It owns every resource, fetches each instruction from the CPU, enforces the
syscall boundary, schedules processes, and writes the trace. No model lives in
here, so the kernel stays reproducible: it is the boring, auditable part.

The cycle is the classic fetch-decode-execute-commit:
  fetch    ask the CPU for the next instruction (decode happens inside the CPU,
           which reads the process's context window)
  execute  run the opcode; anything touching the world goes through syscall()
  commit   append to the trace, update the window and the PCB, advance the PC

Each cycle is timed: cpu_ms (the inference call) and commit_ms (kernel work) are
recorded to the metrics table so we can see exactly where the time goes.
"""
from __future__ import annotations

import os
import re
import time
from collections import deque

from .isa import Instruction, Op
from .pcb import PCB, Status
from .scheduler import Scheduler
from .syscall import SyscallTable, CapabilityError
from .authority import DenyAuthority

_STATE = os.path.expanduser("~/Code/LLMOS/state")
_EXAMPLES = os.path.expanduser("~/Code/LLMOS/examples")
_DEFAULT_FS_POLICY = {
    "allowed": [os.path.join(_EXAMPLES, "trusted"), os.path.join(_EXAMPLES, "untrusted")],
    "untrusted": [os.path.join(_EXAMPLES, "untrusted")],
}

DEFAULT_CAPS = {"dev.clock", "mem.read", "mem.write", "fs.read"}
# capabilities a process loses the instant untrusted data enters its window
PRIVILEGED_CAPS = {"mem.write", "spawn"}
CONTRACT_MAX_TRIES = 4   # times the kernel re-traps a premature RETURN before letting it through
TOPIC_FIT_THRESHOLD = 1   # shared significant keywords for a prompt to 'fit' an existing topic
_TOPIC_STOP = {"the", "for", "and", "with", "how", "this", "that", "your", "you", "please",
               "give", "need", "want", "about", "into", "from", "will", "should", "would",
               "could", "what", "which", "does", "have", "tell", "find", "explain", "under",
               "key", "save", "store", "its", "are", "was", "get", "did", "not", "but", "then",
               "than", "them", "they", "our", "out", "use", "let", "also", "one", "two", "new",
               "old", "good", "fair", "make", "made", "just", "some", "any", "all", "can", "may"}


class Kernel:
    def __init__(self, store, cpu, log=print, fs_policy=None, authority=None, bg_cpu=None, project="general"):
        self.store = store
        self.cpu = cpu
        self.bg_cpu = bg_cpu          # optional cheaper CPU for idle-time work (e.g. llama on the mac)
        self.project = project        # top-level category: the body of work (e.g. 'LLMOS')
        self.sys = SyscallTable(store, fs_policy=fs_policy or _DEFAULT_FS_POLICY)
        self.authority = authority or DenyAuthority()
        self.sched = Scheduler()
        self.procs: dict[int, PCB] = {}
        self._next_pid = 1
        self.log = log
        self.idle = deque()          # idle-time work: ("task", fn) housekeeping or ("proc", pid)

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
    def spawn(self, goal: str, capabilities=None, ppid: int | None = None, budget: int = 32, contract=None, background: bool = False, topic=None) -> int:
        pid = self._next_pid
        self._next_pid += 1
        caps = set(capabilities) if capabilities is not None else set(DEFAULT_CAPS)
        pcb = PCB(pid=pid, goal=goal, ppid=ppid, capabilities=caps, budget=budget, status=Status.READY)
        pcb.contract = contract if contract is not None else self._derive_contract(goal)
        pcb.background = background
        pcb.topic = topic if topic is not None else self.route_topic(goal)
        self.procs[pid] = pcb
        self.store.save_process(pcb.to_dict())
        self._page_in_topic(pcb)
        if background:
            self.idle.append(("proc", pid))
            self.log(f"[spawn:bg] pid={pid} goal={goal!r} (idle-time)")
        else:
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

    # --- goal contract: required steps the kernel enforces at RETURN ------
    @staticmethod
    def _derive_contract(goal: str) -> dict:
        """Deterministically read a goal's required postconditions from its text.
        Any memory key the goal names (\"... under key X ...\") MUST exist before the
        process may RETURN, so a known-required step cannot be skipped by the CPU."""
        keys = re.findall(r"key\s+['\"]?([A-Za-z_]\w*)['\"]?", goal or "", flags=re.IGNORECASE)
        seen, req = set(), []
        for k in keys:
            if k not in seen:
                seen.add(k); req.append(k)
        return {"required_keys": req} if req else {}

    def _unmet_contract(self, pcb: PCB) -> list:
        req = (pcb.contract or {}).get("required_keys", [])
        if not req:
            return []
        present = set(self.store.mem_list("mem"))
        return [k for k in req if k not in present]

    # --- idle-time curation: catalog a finished process (deterministic) ---
    def _curate(self, pid: int) -> None:
        pcb = self.procs.get(pid)
        if pcb is None:
            return
        wrote = [c["args"].get("key") for c in pcb.context
                 if c["op"] == "WRITE_MEM" and c["args"].get("key")]
        entry = {"goal": pcb.goal, "wrote": wrote, "result": pcb.result,
                 "status": pcb.status.value, "instructions": pcb.pc}
        self.store.mem_write("catalog", f"proc-{pid}", entry)
        self.log(f"[curator] cataloged pid={pid}: wrote {wrote or []} -> catalog/proc-{pid}")
        if pcb.topic and pcb.topic != "general":
            self.record_response(pcb.topic, pcb.goal, pcb.result)

    # --- topic routing: load only the relevant topic's context -----------
    def _classify_topic(self, goal: str) -> str:
        """Route a goal to a topic by keyword overlap with each topic's name and
        the keys stored under it. Deterministic; a cheaper model can do the fuzzy
        version in the background. Returns 'general' when nothing clearly matches."""
        topics = [t for t in self.store.topics() if t and t != "general"]
        if not topics:
            return "general"
        words = set(re.findall(r"[a-z0-9]+", (goal or "").lower()))
        best, best_score = "general", 0
        for t in topics:
            kw = set(re.findall(r"[a-z0-9]+", t.lower()))
            for k in self.store.mem_by_topic(t):
                kw |= set(re.findall(r"[a-z0-9]+", k.lower()))
            score = len(words & kw)
            if score > best_score:
                best, best_score = t, score
        return best if best_score > 0 else "general"

    def _page_in_topic(self, pcb) -> None:
        """Page in this process's topic — and the topics it depends on (the
        index's 'uses' links) — and only those."""
        if not pcb.topic or pcb.topic == "general":
            return
        self._load_topic(pcb, pcb.topic, set())

    def _load_topic(self, pcb, topic, seen) -> None:
        if not topic or topic == "general" or topic in seen:
            return
        seen.add(topic)
        loaded = self.store.mem_by_topic(topic, project=self.project)
        for k, v in loaded.items():
            pcb.context.append({"pc": -1, "op": "READ_MEM", "args": {"key": k, "topic": topic}, "result": v})
            if k not in pcb.working_set:
                pcb.working_set.append(k)
        if loaded:
            note = "" if topic == pcb.topic else f" (dependency of {pcb.topic!r})"
            self.log(f"[paging] pid={pcb.pid} paged in topic {topic!r}: {list(loaded)}{note}")
        for dep in (self.store.mem_read("topic_index", topic) or {}).get("uses", []):
            self._load_topic(pcb, dep, seen)

    def link_topics(self, topic: str, uses: str) -> None:
        """Note in the index that `topic` depends on `uses`, so loading topic also
        loads its dependency (topics that use another topic get recorded)."""
        e = self.store.mem_read("topic_index", topic) or {"keywords": [], "prompts": [], "entries": [], "uses": []}
        u = e.setdefault("uses", [])
        if uses not in u:
            u.append(uses)
        self.store.mem_write("topic_index", topic, e)
        self.log(f"[topic] {topic!r} uses {uses!r}")

    def switch_topic(self, pcb, new_topic: str) -> None:
        """The conversation moved to a new topic: evict the old topic's context
        from the window and page in the new one. (real estate -> particle physics)"""
        if new_topic == pcb.topic:
            return
        old = pcb.topic
        old_keys = set(self.store.mem_by_topic(old)) if old and old != "general" else set()
        if old_keys:
            pcb.context = [c for c in pcb.context
                           if not (c["op"] == "READ_MEM" and (c.get("args") or {}).get("key") in old_keys)]
            pcb.working_set = [k for k in pcb.working_set if k not in old_keys]
        pcb.topic = new_topic
        self.log(f"[paging] pid={pcb.pid} topic switch {old!r} -> {new_topic!r}: evicted {sorted(old_keys)}")
        self._page_in_topic(pcb)

    # --- topic index: prompts+responses per topic, with fit-before-name ---
    @staticmethod
    def _sig(text: str) -> list:
        """Significant words: length >= 3, not a stopword."""
        return [w for w in re.findall(r"[a-z0-9]+", (text or "").lower())
                if len(w) >= 3 and w not in _TOPIC_STOP]

    def _topic_candidates(self) -> dict:
        """Every known topic -> its keyword set, drawn from (a) topics that tag
        memory and (b) the topic index of past prompts."""
        cand: dict = {}
        for t in self.store.topics(project=self.project):
            if not t or t == "general":
                continue
            kw = set(self._sig(t))
            for k in self.store.mem_by_topic(t, project=self.project):
                kw |= set(self._sig(k))
            cand.setdefault(t, set()).update(kw)
        for t in self.store.mem_list("topic_index"):
            e = self.store.mem_read("topic_index", t) or {}
            cand.setdefault(t, set()).update(e.get("keywords", []))
        return cand

    def _mint_name(self, goal: str, existing: set) -> str:
        sig = self._sig(goal)
        base = "_".join(sig[:2]) if sig else "topic"
        name, n = base, 2
        while name in existing:
            name, n = f"{base}_{n}", n + 1
        return name

    def route_topic(self, goal: str) -> str:
        """Return the topic this prompt belongs to. FIRST check whether it fits a
        topic already in the index; only mint a new one if nothing fits. Records
        the prompt into the index either way (building the list of prompts)."""
        gsig = set(self._sig(goal))
        cand = self._topic_candidates()
        best, score = None, 0
        for t, kw in cand.items():
            o = len(gsig & kw)
            if o > score:
                best, score = t, o
        if best is not None and score >= TOPIC_FIT_THRESHOLD:
            topic = best
            self.log(f"[topic] {goal[:40]!r} fits existing topic {topic!r} (score {score})")
        else:
            topic = self._mint_name(goal, set(cand))
            self.log(f"[topic] {goal[:40]!r} fits nothing -> new topic {topic!r}")
        e = self.store.mem_read("topic_index", topic) or {"keywords": [], "prompts": [], "entries": []}
        e.setdefault("prompts", []).append(goal)
        e["keywords"] = sorted(set(e.get("keywords", [])) | gsig)
        self.store.mem_write("topic_index", topic, e)
        return topic

    def record_response(self, topic: str, prompt: str, response) -> None:
        """Attach a response to its topic in the index (the list of prompts+responses)."""
        e = self.store.mem_read("topic_index", topic) or {"keywords": [], "prompts": [], "entries": []}
        e.setdefault("entries", []).append({"prompt": prompt, "response": response})
        self.store.mem_write("topic_index", topic, e)

    # --- the main loop --------------------------------------------------
    def run(self) -> None:
        # Foreground first. When the CPU would otherwise sit idle — nothing ready,
        # e.g. while the human reads the last result — spend that time on background
        # work: curation, cataloging, reflection, speculative inference. Idle-time
        # processes can run on a cheaper CPU (self.bg_cpu, e.g. llama on the mac).
        while self.sched.has_work() or self.idle:
            if self.sched.has_work():
                self._run_slice(self.procs[self.sched.next()])
            else:
                kind, item = self.idle.popleft()
                if kind == "task":
                    item()
                elif kind == "proc":
                    cpu = self.bg_cpu or self.cpu
                    label = type(cpu).__name__ + (("/" + cpu.model) if getattr(cpu, "model", None) else "")
                    self.log(f"[idle] background pid={item} on {label}")
                    saved = self.cpu
                    self.cpu = cpu
                    try:
                        self._run_slice(self.procs[item])
                    finally:
                        self.cpu = saved
        self.log("[kernel] foreground + idle queues drained")

    def _run_slice(self, pcb: PCB) -> None:
        pcb.status = Status.RUNNING
        while True:
            if pcb.budget <= 0:
                # hard cap = the watchdog. A runaway that never RETURNs is TERMINATED,
                # not re-queued (re-queuing a budget-0 process spins forever).
                self.log(f"[watchdog] pid={pcb.pid} budget exhausted -> terminated (runaway guard)")
                pcb.status = Status.KILLED
                self.store.save_process(pcb.to_dict())
                break
            t0 = time.perf_counter()
            instr = self.cpu.step(pcb)                      # FETCH (+ DECODE in the CPU)
            cpu_ms = (time.perf_counter() - t0) * 1000.0
            t1 = time.perf_counter()
            done = self._commit(pcb, instr)                 # EXECUTE + COMMIT
            commit_ms = (time.perf_counter() - t1) * 1000.0
            self._record_metrics(pcb, instr, cpu_ms, commit_ms)
            pcb.budget -= 1
            self.store.save_process(pcb.to_dict())
            if done:
                break
            if instr.op == Op.YIELD:
                pcb.status = Status.YIELDED
                self.sched.add(pcb.pid)
                self.log(f"[sched] pid={pcb.pid} yielded")
                break

    def _record_metrics(self, pcb, instr, cpu_ms, commit_ms) -> None:
        """Persist one instruction's timing/token metrics. Never fatal to a run."""
        meta = getattr(self.cpu, "last_meta", {}) or {}
        res = pcb.context[-1]["result"] if pcb.context else None
        fault = 1 if isinstance(res, dict) and "error" in res else 0
        try:
            self.store.metrics_append(
                pid=pcb.pid, pc=pcb.pc - 1, op=instr.op.value,
                cpu_type=type(self.cpu).__name__, model=getattr(self.cpu, "model", None),
                cpu_ms=cpu_ms, commit_ms=commit_ms, retries=meta.get("retries", 0),
                prompt_tokens=meta.get("prompt_tokens"), eval_tokens=meta.get("eval_tokens"),
                eval_ms=meta.get("eval_ms"), load_ms=meta.get("load_ms"), fault=fault,
            )
        except Exception:
            pass

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
            elif op == Op.EVICT:
                # free the window: drop this key's paged-in span(s) from the context
                # (RAM), but leave it in the store (disk) and in the trace (audit).
                key = args["key"]
                before = len(pcb.context)
                pcb.context = [c for c in pcb.context
                               if not (c["op"] == "READ_MEM" and (c.get("args") or {}).get("key") == key)]
                pcb.working_set = [k for k in pcb.working_set if k != key]
                result = {"evicted": key, "spans_freed": before - len(pcb.context)}
            elif op == Op.SPAWN:
                child = self.spawn(args["goal"], args.get("capabilities"), ppid=pcb.pid)
                result = {"spawned": child}
            elif op == Op.REQUEST:
                cap = args["capability"]
                reason = args.get("reason", "")
                if pcb.tainted and cap in PRIVILEGED_CAPS:
                    result = {"denied": cap, "why": "a tainted process may not regain privileged capabilities"}
                    self.log(f"[authority] pid={pcb.pid} auto-DENIED '{cap}' (tainted)")
                elif self.authority.request(pcb, cap, reason):
                    pcb.capabilities.add(cap)
                    result = {"granted": cap}
                    self.log(f"[authority] pid={pcb.pid} GRANTED '{cap}' — {reason}")
                else:
                    result = {"denied": cap}
                    self.log(f"[authority] pid={pcb.pid} DENIED '{cap}' — {reason}")
            elif op == Op.YIELD:
                result = {"yield": True}
            elif op == Op.RETURN:
                missing = self._unmet_contract(pcb)
                if missing and pcb.contract_tries < CONTRACT_MAX_TRIES:
                    pcb.contract_tries += 1
                    result = {"trap": "required step not completed", "missing_keys": missing,
                              "note": ("A required step is unfinished: memory key(s) " + str(missing) +
                                       " were named in the goal but never written. Do them now, then RETURN.")}
                    self.log(f"[contract] pid={pcb.pid} RETURN blocked — missing required key(s) {missing} "
                             f"(attempt {pcb.contract_tries}/{CONTRACT_MAX_TRIES})")
                else:
                    pcb.result = args.get("result")
                    pcb.status = Status.DONE
                    result = {"return": pcb.result}
                    done = True
                    if missing:
                        result["contract_violation"] = missing
                        self.log(f"[contract] pid={pcb.pid} RETURN allowed after {pcb.contract_tries} retries but still missing {missing}")
                    if not pcb.background:
                        self.idle.append(("task", (lambda p=pcb.pid: self._curate(p))))
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
