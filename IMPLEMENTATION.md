# LLMOS — Implementation Plan (v1)

*How the architecture in [ARCHITECTURE.md](ARCHITECTURE.md) gets built.*

Status: v1 plan, 2026-07-03. Decided with Mikey.

---

## 1. Stance: LLMOS is a hosted runtime, not a from-scratch OS

v1 of LLMOS runs **as a guest on macOS** — an operating-system-within-an-operating-system. macOS is treated as the hardware and the hardware-abstraction layer. We reuse every macOS facility we can and write only the parts that are genuinely new (the LLM-CPU machinery).

The right mental model is a **process virtual machine** — the JVM or Erlang's BEAM — not a bootable kernel. Bytecode is intents; the interpreter is an LLM; the host OS provides everything below that. BEAM is the closest precedent: it schedules many cheap logical processes onto a few real OS threads, which is exactly LLMOS's shape (one scarce, expensive execution unit; many logical processes).

Writing a standalone OS that replaces Linux/macOS — our own bootloader, kernel, filesystem, and drivers on bare metal — is an explicit **later phase**, out of scope for v1 (see Section 8).

## 2. The macOS delegation map

The governing rule: **do not rewrite what macOS already provides.** Nearly every subsystem in the architecture spec is delegated.

**Delegated to macOS (we do not build these):**

- **launchd** → boot, init, service supervision, timers/cron (`StartCalendarInterval`), and crash auto-restart (`KeepAlive`). Our daemons don't need a supervisor; launchd is the supervisor.
- **Unix processes** → the process table, memory isolation, resource limits (`ulimit`), and lifecycle.
- **Unix signals** → interrupts and preemption. `SIGSTOP`/`SIGCONT` suspend and resume a process; `SIGTERM`/`SIGKILL` terminate. The scheduler *is* a signal-sender.
- **APFS filesystem** → disk and the VFS backing store.
- **SQLite** → structured persistent memory (the brain already uses it), transactions, and locking. Our concurrency guards are SQLite transactions, not a lock manager we write.
- **Unix domain sockets / pipes** → IPC between kernel and agents.
- **sandbox-exec + POSIX permissions + a dedicated user** → real, kernel-enforced capability restriction for untrusted agents.
- **Ollama / llama.cpp** → the CPU. We never write inference.
- **Terminal / a CLI** → the shell surface. **os_log / Console** → logging.

**Built by us (the irreducible LLMOS core — plain, deterministic Python):**

1. the **kernel loop** (fetch-decode-execute-commit)
2. the **ISA**: an intent schema + a small opcode set and its interpreter
3. the **PCB** and its mapping to host processes
4. the **scheduler policy** (who gets the expensive CPU next)
5. the **syscall dispatcher + capability check** (the trust boundary)
6. the **paging policy** (recall/evict over the brain)
7. the **trace + replay**

Seven small pieces of glue. Everything else is macOS.

## 3. The process model: one macOS process per agent

**Decision:** each LLMOS process is a real macOS process the kernel spawns and supervises.

**Why:** normally you avoid a heavyweight OS process per logical task because spawning is costly. Here the LLM CPU is so slow and expensive that spawn time is negligible next to a single inference call. Because the bottleneck is the CPU, the luxurious option is affordable — so we take it and inherit, for free: isolation, `kill`, `SIGSTOP`/`SIGCONT` preemption, `sandbox-exec` sandboxing, `ulimit` resource caps, and `ps` visibility. We reimplement none of it. This is also the most faithful reading of "use as much of macOS as you can."

**How it works:**

- The **kernel** is one long-lived macOS process (a launchd daemon). It owns the SQLite stores, the trace, the scheduler, and a control socket.
- To run a goal, the kernel **spawns an agent** — a child macOS process running `agent_runner.py` with a PCB and a program.
- The agent runs the CPU (calls Ollama) to execute its instructions. Inference is the CPU's *internal computation*, so the agent does it directly — no world-effect there.
- Anything that touches the world — a file, the web, a memory write, spawning a child — is emitted as a **syscall request over the control socket** to the kernel. The kernel validates it against the agent's capabilities, executes it, and returns the result. The model never touches the world, and the kernel stays deterministic (no model in ring 0).
- **Scheduling is signalling:** with a single expensive CPU (one Ollama backend), the kernel runs one agent at a time by `SIGCONT`-ing the chosen agent and `SIGSTOP`-ing the rest. v1 is cooperative — an agent runs until it `YIELD`s or hits its budget, then the kernel stops it and picks the next.
- **The trace has a single writer:** agents report each executed instruction to the kernel over the socket, so the kernel alone appends to the ledger. Single-writer means no race — the `claude-loop-trace` lost-update problem cannot happen by construction.
- **Untrusted agents** (running tool-returned or web content) are launched under `sandbox-exec` with a restrictive profile and a reduced capability set: scratch memory namespace only, no `/dev/web`, no filesystem writes. This is the concrete answer to the prompt-injection threat from the spec — provenance drops privilege, and macOS enforces it.

## 4. Concrete v1 components

All plain Python on the Mac mini, under `~/Code/LLMOS/llmos/`:

- `kernel.py` — the deterministic loop; owns stores, socket, scheduler, trace.
- `isa.py` — the intent schema + opcodes: `PLAN`, `CALL` (syscall), `READ_MEM`, `WRITE_MEM`, `SPAWN`, `YIELD`, `RETURN`.
- `pcb.py` — the process control block dataclass; serialize to SQLite/JSON for checkpoint and resume.
- `scheduler.py` — ready queue + policy. v1: cooperative, single run-queue, hard budget cap as the safety net.
- `syscall.py` — the dispatcher and trust boundary: validate capability → execute (shell / MCP / file / device) → return result + append to trace.
- `memory.py` — the VFS and paging policy over the brain (SQLite) and the filesystem; mounts `/mem /proc /dev /self /fs`.
- `cpu.py` — the `/dev/cpu0` driver: talk to Ollama (local mini, or pop's GPU), temperature 0 + fixed seed, parse output into either a result or a structured syscall request.
- `trace.py` — append-only ledger (a SQLite table) + a `replay` function.
- `agent_runner.py` — the guest process: given a PCB + program, executes instructions, calling back to the kernel for syscalls.
- `bin/llmos` — the shell CLI: `run "goal"`, `ps`, `kill <pid>`, `attach <pid>`, `replay <pid>`.

## 5. launchd integration (with the gotchas pre-handled)

Daemons (`llmos-kernel`, and later a compaction and a logging daemon) are launchd agents. Two of your existing protocols apply directly and are baked in from the first plist:

- **Named process wrappers:** a LaunchAgent that execs a bare `python3` shows up anonymously in Login Items. Each daemon execs a named wrapper (e.g. a small script named `llmos-kernel`) so it's legible in `ps` and Login Items as `llmos-kernel`, not `python3`.
- **Shell-out hardening:** launchd respawns daemons with a *different environment* than an interactive shell (different `PATH`, no shell profile). The kernel must resolve absolute paths to `sqlite3`, `ollama`, etc., and not depend on shell state. This is the class of bug that only appears on restart, never in the terminal — so we assume the launchd environment from the start.

## 6. The CPU is network-transparent

The kernel lives on the mini; inference runs wherever the model is. `/dev/cpu0` is a driver pointing at a backend: local Ollama on the mini for small/fast models, or pop's GPU (e.g. ornith:35b over the existing tunnel) for heavy ones. Swapping the CPU — a different model, or a fine-tune — is a driver config change, not a code change. The device's known failure mode (a cold model load blowing the timeout) is handled at the driver: keep-warm plus driver-level timeouts, the same fix already applied on pop.

## 7. The v1 milestone — "hello, world" for LLMOS

Concrete end-to-end path that proves the whole spine:

1. launchd starts `llmos-kernel`.
2. The kernel boots: opens the SQLite stores, loads the boot-ROM keys by exact key, opens its control socket.
3. `llmos run "note the current time to memory"` sends the goal to the kernel.
4. The kernel creates a PCB and spawns one agent process (`llmos-proc-1`), then `SIGCONT`s it.
5. The agent executes a short program: `PLAN` → `CALL` a capability-checked syscall (read `/dev/clock`) → `WRITE_MEM` the result into the brain → `YIELD`.
6. Every instruction is reported to and traced by the kernel (single writer).
7. The agent yields; the kernel checkpoints the PCB, marks the process done, logs it.
8. `llmos replay 1` re-runs the process deterministically from the trace.

If that runs and replays, the kernel loop, ISA, PCB, scheduler, syscall boundary, memory write, and trace all exist. Everything after is adding opcodes, devices, and scheduling policy.

## 8. Explicitly deferred to the later standalone-OS phase

Out of scope for v1, by decision — these belong to the eventual "replace Linux/macOS" phase:

- our own bootloader and bare-metal kernel
- our own filesystem and block-device drivers
- driving hardware directly / running without a host OS
- multi-machine distribution of the process pool

v1 earns its keep as a hosted runtime first. We only consider going lower than macOS once the LLM-CPU model has proven itself as a guest.
