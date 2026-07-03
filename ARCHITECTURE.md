# LLMOS — Architecture

*An operating system where the LLM is the CPU.*

Status: design draft, v0.1 (2026-07-03). Spec-first. No code yet — this document defines the machine before we build it.

---

## 1. Thesis

A conventional OS exists to multiplex a scarce, fast, deterministic resource — the CPU — across many programs, while mediating their access to memory, storage, devices, and each other. LLMOS keeps the entire structure of that job and swaps the resource: the scarce execution unit is now an **LLM forward pass**, and the "programs" are goals expressed as structured intents.

Everything downstream of that swap changes character but not role. Memory is still a hierarchy — it's just that the fast tier is a context window and the disk is content-addressable by *meaning*. Scheduling still decides who runs next — it's just measured in tokens and dollars per quantum instead of clock cycles. System calls still cross a privilege boundary — they're just tool calls. The value of the OS framing is that it forces us to build *all* the moving parts, not just the fun one, and it tells us exactly what those parts are, because Linux already enumerated them.

## 2. The core inversion — what is the CPU, and what is the kernel?

The single most important design decision, made up front:

- **The LLM is the CPU, not the kernel.** It executes one instruction per inference call. It is stochastic, powerful, and *untrusted* — it must not touch the world directly.
- **The kernel is a small, deterministic orchestration loop** (plain Python/Rust, no model in it). The kernel owns all resources, fetches the next instruction, dispatches syscalls, manages memory, schedules processes, and writes the trace. It is boring on purpose. Boring is auditable.

This is a microkernel stance. The intelligence lives in the CPU (the model) and in the *programs* (intent graphs / skills); the kernel is dumb plumbing that stays reproducible. When the CPU misbehaves — hallucinates an instruction, loops, overflows its context — it is the deterministic kernel that catches it, because you cannot trust the thing that made the mistake to also be the thing that detects it.

## 3. Subsystem map at a glance

Linux part → LLMOS analog → does Mikey already have a piece of it?

- Boot / init (BIOS → bootloader → init) → mount memory, load identity ROM, start scheduler → **yes** (`brain_init` + boot-core memories)
- CPU / ISA / execution unit → LLM forward pass; instruction = structured intent → **partly** (`forms.json` is a proto-ISA)
- Kernel → deterministic orchestration loop → **to build**
- Process + PCB → an agent with its own context window + serializable state → partly (outer-loop is one process)
- Scheduler → chooses which process gets the next inference cycle → **yes, in spirit** (outer-loop self-paced cadence)
- Memory management (registers/RAM/disk/paging/swap) → tokens / context window / brain, with recall-as-page-fault → **yes** (the brain)
- System call interface → tool calls through a validated boundary → **yes** (MCP + `tools-registry`)
- Interrupts & signals → async events + trigger-fired handlers → **yes** (protocols / triggers)
- Filesystem / VFS → unified namespace over brain, files, self-ledger, reachi memory → **partly** (multiple stores exist, no VFS)
- Device drivers → adapters for model backends, web, browser, clock → partly (MCP servers are drivers)
- IPC → shared memory (brain), mailboxes, pipes → partly (MAS protocols)
- Concurrency / sync → locks + atomic writes on shared memory keys → **to build** (already hit a race on `claude-loop-trace`)
- Security / permissions / sandbox → capability sets + privilege rings → **yes** (computer-use read/click/full tiers)
- Services / daemons → long-running supervised processes → **yes** (launchd agents, harness maintenance, outer-loop)
- Shell / UI → conversational job control → **yes** (Cowork / chat)
- Clock / timers / cron → wall clock + scheduled wakes → **yes** (`get-current-time`, scheduled-tasks, launchd)
- Package management / linking → skills + protocols as installable programs → **yes** (`.skill` / plugin bundles)
- Logging / observability / /proc → the trace ledger + live process introspection → **yes** (harness ledger + telemetry)
- Fault handling / panic / OOM / watchdog → typed faults + self-repair → **partly** (stop-trigger, struggle, active-inference)

The headline: LLMOS is not a from-scratch build. Roughly two-thirds of the subsystems already exist as separate tools in your stack. The project is mostly *giving them a kernel* — one loop, one process abstraction, one namespace — so they stop being scattered utilities and become an operating system.

---

## 4. The subsystems in detail

Each subsystem below: what Linux does, how LLMOS does it, and what to build.

### 4.1 Boot process
**Linux:** firmware → bootloader → kernel init → `init`/systemd → userspace. Deterministic bring-up in a fixed order.
**LLMOS:** a boot sequence that (1) mounts persistent memory (the brain = disk), (2) loads the **boot ROM** — pinned identity/config that must load by exact key, never by fuzzy recall, (3) restores last checkpointed state, (4) starts the scheduler daemon, (5) spawns the init process. Your `brain_init` plus the boot-core memory list is already exactly this: load-by-exact-key config first, then recent state. That "load by key, not by semantic recall" rule is the LLMOS equivalent of a boot ROM being at a fixed physical address — you cannot boot off a fuzzy guess.
**Build:** formalize `boot.py`: fixed load order, fail loudly if the ROM keys are missing.

### 4.2 The kernel
**Linux:** monolithic core mediating everything; the one privileged component.
**LLMOS:** the deterministic loop of Section 2. Its main cycle is the classic fetch-decode-execute:
1. **fetch** — pick the running process's next instruction (intent) from its program.
2. **decode** — resolve which memories/tools that intent needs; page them into the window.
3. **execute** — one LLM inference call produces either an output or a syscall request.
4. **commit** — validate, run any syscall, write the result back to the process's context, append to the trace, update the PCB, hand control to the scheduler.
**Build:** this loop is the first real code artifact.

### 4.3 CPU, ISA, and the execution unit
**Linux:** the physical CPU decodes a fixed instruction set; one instruction is unambiguous.
**LLMOS:** the LLM *is* the execution unit; **one inference call is one instruction cycle**. The ISA is a schema for a structured intent — a goal, its typed inputs, and the expected output shape. Programs are sequences or graphs of these intents. Your `forms.json` is a proto-ISA already: named schemas with required fields. The critical difference from a silicon ISA is that decoding is *interpretive* — the CPU reads intent, it doesn't bit-match an opcode. That is the source of both LLMOS's power and its whole class of novel hazards (Sections 5 and 4.13).
**Build:** define the instruction schema (an "intent" record) and a tiny set of core opcodes: `PLAN`, `CALL` (syscall), `WRITE_MEM`, `READ_MEM`, `SPAWN`, `YIELD`, `RETURN`.

### 4.4 Process model and the PCB
**Linux:** a process is an isolated address space + a Process Control Block (pid, registers, program counter, state, open files).
**LLMOS:** a process is **an agent with its own context window** (its address space), a goal, and a **PCB** that is fully serializable so a process can be checkpointed, evicted, and resumed:
- `pid`, parent pid
- `goal` (the program it's running)
- `program_counter` (where in the intent graph)
- `working_set` (which memory keys are currently paged in)
- `status` (running / ready / blocked / yielded / zombie)
- `capabilities` (which devices, tools, and memory namespaces it may touch)
- `budget` (tokens/dollars remaining before preemption)
Isolation = separate context windows. That windows are physically separate is *stronger* isolation than Linux gets for free — one process literally cannot read another's registers without going through IPC.
**Build:** the PCB struct + serialize/restore.

### 4.5 Scheduler
**Linux:** decides which ready process gets the CPU; policies like CFS, priorities, preemption on a timer.
**LLMOS:** decides which ready process gets the **next inference cycle**. The quantum is a *budget* — N cycles, or a token/cost cap — after which the kernel preempts and re-evaluates. Policy options to support: cooperative (process runs until it `YIELD`s), preemptive (kernel interrupts at budget), priority, and event-driven (a process wakes when its awaited event arrives). Your outer-loop is already a self-scheduling process that sets its own next quantum via `update_scheduled_task` — that's cooperative scheduling with self-chosen cadence. LLMOS generalizes it to *many* processes competing for one expensive CPU.
**Build:** a ready-queue + a pluggable policy function; start with cooperative + a hard budget cap as the safety net.

### 4.6 Memory management
**Linux:** registers → cache → RAM → swap/disk, with an MMU doing virtual-to-physical translation and paging.
**LLMOS:** the same hierarchy, re-tiered:
- **registers** = the tokens the model is actively attending to this cycle.
- **RAM** = the context window. Bounded, fast, expensive, volatile.
- **disk** = the brain (semantic store) + flat files + the self-ledger. Unbounded, persistent, addressable **by meaning**.
- **the MMU / page-fault handler** = `brain_recall`. When an instruction needs a datum that isn't resident in the window, that's a *page fault*: the kernel recalls it from disk into RAM. This is the single cleanest correspondence in the whole design, and you already run it every session.
- **swap / eviction** = when the window fills, an eviction policy (least-recently-useful, or summarize-and-compact) writes spans back to disk via `brain_remember`. Context compression = memory compaction.
- **virtual address space** = each process sees only its own working set; the same underlying memory key can be shared (see IPC) but each process's *view* is private.
This is MemGPT's insight, generalized into a real VM subsystem: the context window is not the memory, it's the *cache*.
**Build:** an eviction policy + a "working set" tracker on each PCB. Decide the page-fault trigger: explicit (`READ_MEM` opcode) vs. automatic (kernel predicts a fault and prefetches).

### 4.7 System call interface
**Linux:** user code traps into the kernel via a syscall; the kernel validates and performs the privileged operation.
**LLMOS:** the CPU **cannot touch the world**. When it wants to read a file, search the web, or write memory, it emits a **syscall request** — a tool call in a fixed schema. The kernel (1) checks the request against the process's `capabilities`, (2) executes it via the right device driver, (3) returns the result into the process's context. Your MCP tool layer *is* the syscall mechanism, and `tools-registry` is the syscall table. The trap boundary is what makes the untrusted CPU safe: nothing happens to the world unless a syscall the kernel approved made it happen.
**Build:** a syscall dispatcher that enforces capabilities before executing any tool.

### 4.8 Interrupts and signals
**Linux:** hardware IRQs and software signals preempt the CPU; an interrupt vector maps events to handlers.
**LLMOS:**
- **interrupts** = asynchronous events: an incoming user message, a tool result arriving from a slow device, a timer firing, a message from another process. The kernel holds an **interrupt vector**: event type → handler / which process to wake.
- **handlers** = your protocols/triggers. A matched condition fires a routine — that is precisely an interrupt handler.
- **signals** = kernel-to-process control messages: `STOP` (your stop-trigger), `CHECKPOINT`, `KILL`, `YIELD`. A process must be able to receive a signal between instructions.
**Build:** an event queue the kernel drains each loop turn, plus the vector table.

### 4.9 Filesystem and VFS
**Linux:** everything is a file; a Virtual File System unifies many backends behind one path namespace.
**LLMOS:** you currently have several stores — the brain, flat files in `~/Code`, the self-ledger, the reachi memory API — with no unifying namespace. LLMOS adds a **VFS**: one namespace mounting all of them.
- `/mem/...` → the brain (keys are inodes, types are directories)
- `/proc/...` → live process state (read the PCBs of running processes)
- `/dev/...` → devices (see 4.10)
- `/self/...` → the identity ledger
- `/fs/...` → real files on disk
"Reading a file" = fetching a record by name; semantic recall is an *additional* addressing mode the VFS offers that no POSIX filesystem has — address by meaning, not just by path.
**Build:** the mount table + a uniform `read`/`write`/`list` interface each backend implements.

### 4.10 Device drivers
**Linux:** drivers present hardware behind a uniform interface (`/dev/sda`, char/block devices).
**LLMOS:** devices are external capabilities, each behind a driver:
- `/dev/cpu0` → the model backend itself (gemma on pop, ornith:35b, Claude API). **The CPU is a hot-swappable device** — you can change or fine-tune the model without rewriting programs. This is a genuinely new idea with no silicon analog.
- `/dev/web` → web search/fetch
- `/dev/browser` → the Chrome/computer-use device
- `/dev/clock` → wall-clock time (`get-current-time`)
- `/dev/fs` → the local filesystem
A driver is the adapter that speaks a device's protocol (Ollama HTTP, MCP, a REST API) and presents the kernel a uniform `open/read/write/close`. Your existing pain — ornith cold-loading off the HDD and blowing timeouts — is a *device driver latency* problem, and modeling it as one tells you where the fix goes: driver-level timeouts and a keep-warm policy, which is exactly the `OLLAMA_KEEP_ALIVE=-1` decision you already made.
**Build:** a driver interface; wrap each backend.

### 4.11 Inter-process communication
**Linux:** pipes, sockets, shared memory, message queues, signals.
**LLMOS:**
- **shared memory** = the brain. One process writes a key, another reads it. Simplest IPC; needs locking (4.12).
- **message passing** = a mailbox per process; the kernel routes messages between pids.
- **pipes** = one process's output stream feeds another's input — agent chaining. `procA | procB`.
Your MAS protocols are the contract layer for this — how agents hand off without the known multi-agent failure modes.
**Build:** mailboxes first (safest); pipes as a convenience over them.

### 4.12 Concurrency and synchronization
**Linux:** locks, mutexes, semaphores, atomics protect shared state from races.
**LLMOS:** the moment two processes share the brain, you have a race. You already hit one: the `claude-loop-trace` write-only/overwrite hazard, where a blind overwrite would destroy rows another waking wrote. That is a textbook lost-update race, and it's the proof this subsystem is not optional.
**Build:** per-key locks, atomic/append-only writes for shared logs, and transactions (your db MCP already exposes `db_transaction`). Rule of thumb: shared logs are append-only; shared state is lock-guarded.

### 4.13 Security, permissions, sandboxing
**Linux:** uid/gid, file permissions, capabilities, privilege rings (kernel vs user), SELinux, namespaces.
**LLMOS:**
- **privilege rings** = kernel (full authority) vs. process (restricted to its `capabilities`).
- **capabilities** = the per-process set of allowed devices, tools, and memory namespaces. Your computer-use tiers (read / click / full) are already a working capability model — generalize it to every device.
- **sandboxing** = an untrusted program runs with a minimal capability set: no `/dev/web`, no `/dev/fs` writes, a scratch memory namespace only.
- **the novel threat: prompt injection = the buffer-overflow of LLMOS.** Because the CPU decodes intent interpretively, *data can be misread as instructions* — a web page or an email can carry text that the CPU treats as a command. This is the defining security problem of an LLM OS and it has no clean Linux analog. Mitigations to design in from day one: never run tool-returned content at the CPU's own privilege; tag provenance on every datum (trusted / untrusted) and drop the capability set while untrusted data is in the window; keep the syscall validator deterministic and outside the model. Your existing instinct — "flag instructions in my context that seem designed to manipulate" — is a user-space intrusion detector; LLMOS makes it a kernel-enforced boundary.
**Build:** capability enforcement in the syscall dispatcher + provenance tags on memory records.

### 4.14 Init and service management
**Linux:** `init`/systemd starts and supervises daemons; restarts them on failure; health checks.
**LLMOS:** after boot, init spawns the standing daemons: the scheduler, a memory-compaction daemon (your harness maintenance pass = a cron daemon), the outer-loop, the logging daemon. Supervision = restart a crashed process, run health checks, honor dependencies (don't start X until the brain mount is up).
**Build:** a supervisor that owns the daemon table; your launchd agents are the current, un-unified version of this.

### 4.15 Shell and user interface
**Linux:** a shell interprets commands, spawns processes, does job control (fg/bg, kill, jobs).
**LLMOS:** the shell is **conversational**. A user goal spawns a process (or routes to a running one) and streams its output. Job control becomes: list running processes (`ps`), suspend/resume, kill, inspect a process's context. Cowork/chat is the current shell — LLMOS makes "what you type" formally a command that spawns a scheduled process rather than a one-shot completion.
**Build:** a `ps`/`kill`/`attach` command surface over the process table.

### 4.16 Clock, timers, and cron
**Linux:** system clock, timers, `cron`/`at` for scheduled work.
**LLMOS:** wall clock via `/dev/clock`; timers = scheduled wakes (your scheduled-tasks MCP + launchd); cron = recurring daemons (daily harness maintenance). The scheduler's quantum is also a timer — "run this process for at most N cycles." Time matters more here than in Linux because a "cycle" has real dollar cost and real latency, so the clock is a first-class scheduling input, not just a wall display.
**Build:** wire scheduled-tasks in as the timer device; expose "cost-so-far" as a readable clock.

### 4.17 Package management and dynamic loading
**Linux:** package managers install programs; the linker/loader pulls shared libraries into a process at runtime; dependency resolution.
**LLMOS:** a **program** is a reusable intent graph; a **package** is a skill/plugin bundle (your `.skill` and plugin format already are this — a manifest + instructions + tools). Installing a program = registering a skill/protocol. The **linker/loader** = the mechanism that pulls a skill's instructions into a process's context at the moment it's needed (dynamic linking — you don't load every skill into every window, you link on demand). Dependency resolution = a program declares the devices/capabilities it requires and the kernel refuses to run it if they're absent.
**Build:** a package manifest format (mostly done via skills) + a loader that links a skill into a process on first use.

### 4.18 Logging and observability
**Linux:** syslog, dmesg, `/proc`, `/sys`, metrics.
**LLMOS:** every instruction, syscall, page-fault, and context-switch appends to **the trace** — your harness ledger. `/proc` = live introspection of the process table. `dmesg` = the kernel log. Metrics = tokens/cost/latency per process (your telemetry). Observability is not a nicety here: because the CPU is nondeterministic (Section 5), the trace is the *only* thing that makes an execution auditable and replayable.
**Build:** a structured trace record per instruction; you have most of the ledger already.

### 4.19 Fault handling, recovery, and panic
**Linux:** exceptions, the OOM killer, watchdog timers, kernel panic.
**LLMOS:** typed faults, each with a handler:
- **illegal instruction** — the CPU emitted malformed/unschema output → re-decode, or trap to a repair routine.
- **segfault** — a process referenced a memory key it lacks capability for → deny + log.
- **OOM** — context overflow → force compaction, or kill the lowest-priority process.
- **device timeout** — a driver hung (the ornith cold-load) → retry with backoff, or fail the syscall cleanly.
- **livelock / runaway** — a process loops without progress → the **watchdog** kills it. Your stop-trigger and struggle protocols are this watchdog.
- **panic** — unrecoverable kernel state → checkpoint every PCB and halt gracefully, never corrupt the brain.
And the capability Linux does **not** have: **self-repair.** Your active-inference loop (predict → observe → surprise → update) turns a fault into a *learning event* — the OS proposes a fix to the offending program/protocol so the same fault is less likely next time. That's covered next.

---

## 5. The central design tension: a nondeterministic CPU

Every hard part of LLMOS traces back to one fact: **a silicon CPU is deterministic; an LLM CPU is not.** Run the same instruction twice and you may get two different results. An operating system's whole contract — reproducibility, debuggability, "it did exactly this and here's why" — assumes determinism.

The resolution is three disciplines, designed in from instruction zero:

1. **Pin the knobs.** Temperature 0 and a fixed seed for anything that must be reproducible. Accept that some drift remains across model versions.
2. **Trace everything.** Every instruction, its inputs (which memories were paged in), the raw CPU output, and the syscall results go to an append-only ledger. If you can't reproduce the computation, you can at least *replay the record* of it.
3. **Make replay a first-class operation.** Given a trace, the kernel can re-run a process deterministically from any checkpoint using the recorded CPU outputs — for debugging, for audits, for "why did it do that."

A nondeterministic CPU is not a bug to be suppressed; it's the defining property to be *managed*. Trace + replay is what converts "a smart process did something" into "an operating system executed an auditable program."

## 6. What LLMOS has that Linux never could

Cataloguing the analogs is most of the work, but the point isn't to cosplay Linux — it's to notice where the LLM-as-CPU model gives you capabilities silicon can't:

- **A self-modifying microcode layer.** Programs and protocols improve from experience (your propose / reflect / graduation loop). Linux's code is static between releases; LLMOS's programs evolve at runtime.
- **Semantic addressing.** Disk is addressable by meaning, not just by path. A page fault can be answered by "the most relevant memory," not only "the byte at this offset."
- **A swappable, improvable CPU.** You can change or fine-tune the execution unit. No motherboard lets you do that.
- **Interpretive decode.** Instructions are understood, not bit-matched — which is why one program can be written in near-natural-language and still run, and also why prompt injection exists. Power and hazard from the same property.

These are the reasons to build LLMOS rather than just admire the metaphor. They're also where the research lives.

## 7. Boot sequence — a concrete walkthrough

What actually happens when the machine starts, end to end:

1. **Power on.** The kernel loop starts (plain code, no model yet).
2. **Mount disk.** Connect the brain, the self-ledger, and the filesystem into the VFS mount table.
3. **Load boot ROM.** Fetch the pinned identity/config memories *by exact key* (your boot-core list). Fail loudly if any are missing — you do not boot off a fuzzy guess.
4. **Restore state.** Read the last checkpoint: which processes existed, their PCBs, the trace head.
5. **Attach devices.** Bring up drivers: `/dev/cpu0` (pick the model backend), `/dev/clock`, `/dev/web`, etc. Health-check each.
6. **Start the scheduler daemon.**
7. **Init.** Spawn standing daemons (compaction, logging, outer-loop).
8. **Spawn the shell.** The machine is now ready to accept a goal and schedule a process to pursue it.

Step 3's "by exact key, not recall" rule is load-bearing and you learned it the hard way (the waking-one recall miss). It belongs in the boot spec as a hard invariant.

## 8. What already exists vs. what to build

**Already exists (wrap, don't rewrite):** brain (disk + page-fault via recall), tool layer + tools-registry (syscalls), protocols/triggers (interrupt handlers), forms.json (proto-ISA), outer-loop (a self-scheduling process), harness ledger + telemetry (the trace + /proc), computer-use tiers (capabilities), scheduled-tasks + launchd (timers/daemons), skills/plugins (packages), db transactions + reachi memory API (storage backends), stop-trigger/struggle (watchdog), active-inference (self-repair).

**To build (the genuinely new core):**
1. **The kernel loop** (fetch-decode-execute-commit). The keystone.
2. **The process model + PCB** (serializable, checkpointable).
3. **The scheduler** (ready-queue + budget preemption; cooperative first).
4. **The syscall dispatcher with capability enforcement.**
5. **The VFS** (one namespace over the existing stores).
6. **Memory eviction/compaction policy** (make the window a real cache).
7. **Concurrency guards** on shared memory (locks + append-only logs).

That is a small, sharp list. The first runnable milestone is items 1–4 with everything else stubbed: a kernel that boots, spawns one process, runs a short intent program, makes a couple of capability-checked syscalls, and writes a replayable trace. "Hello, world" for LLMOS is a process that boots, does one syscall, writes one memory, yields, and gets cleanly resumed.

## 9. Open design questions

- **Page faults: explicit or automatic?** Does a program `READ_MEM` deliberately, or does the kernel predict what's needed and prefetch? Explicit is debuggable; automatic is smarter and riskier.
- **Scheduling policy for one very expensive CPU.** With a single costly execution unit, is preemption even worth the context-switch cost (re-priming a window is expensive)? Maybe LLMOS is mostly cooperative with a hard watchdog, not preemptive.
- **How isolated are processes, really?** Fully separate windows (strong, costly) vs. shared window with tagged regions (cheap, leaky)?
- **Where does determinism end?** Which subsystems must be reproducible (kernel, scheduler, syscall validator — yes) and which are allowed to be fuzzy (the CPU's actual reasoning — necessarily)?
- **Is the kernel itself ever allowed to call the model?** The clean answer is no (keep the kernel deterministic). But scheduling *might* want judgment. Resist it initially; a model in the kernel is a model in ring 0.

## 10. Glossary

- **CPU** — the LLM; one inference call = one instruction cycle.
- **Kernel** — the deterministic orchestration loop; owns resources, never the model.
- **Instruction / intent** — a structured op: goal + typed inputs + expected output.
- **Program** — a sequence or graph of intents.
- **Process** — an agent with its own context window, goal, and PCB.
- **PCB** — process control block; the serializable state that makes a process checkpointable.
- **RAM** — the context window (the cache, not the memory).
- **Disk** — the brain + files + ledger; addressable by meaning.
- **Page fault** — needed datum absent from the window; answered by recall.
- **Syscall** — a capability-checked tool call; the only way the CPU touches the world.
- **Interrupt** — an async event; handled by a trigger/protocol.
- **Capability** — the set of devices/tools/namespaces a process may touch.
- **Trace** — the append-only ledger of every instruction; the basis for replay.
- **Panic** — unrecoverable state; checkpoint and halt without corrupting the brain.

---

*Next step after this spec: pick the first milestone (kernel loop + one process + capability-checked syscalls + trace) and write it as a runnable prototype, per the chat-spec / Claude-Code-runner workflow.*
