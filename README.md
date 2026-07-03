# LLMOS

An operating system where the **LLM is the CPU**.

Instead of a deterministic silicon CPU executing machine instructions, the execution unit is an LLM forward pass, and programs are goals expressed as structured intents. Everything a real OS does — scheduling, memory management, system calls, interrupts, a filesystem, drivers, IPC, security, fault recovery — still has to exist. LLMOS builds all of it around that one swapped part.

## The core stance

- **The LLM is the CPU** — powerful, stochastic, and untrusted. It executes one instruction per inference call and never touches the world directly.
- **The kernel is a small, deterministic loop** — plain code, no model inside it. It owns every resource, dispatches system calls, manages memory, schedules processes, and writes an auditable trace. It's a microkernel: the intelligence lives in the CPU and in the programs, not in the plumbing.

## Design docs

- **[ARCHITECTURE.md](ARCHITECTURE.md)** — every OS subsystem and its LLMOS design, the central design tension (a nondeterministic CPU), and the boot sequence.
- **[IMPLEMENTATION.md](IMPLEMENTATION.md)** — v1 runs as a hosted runtime on macOS (a process VM, JVM/BEAM-style), delegating boot/supervision (launchd), isolation/preemption (Unix processes + signals), storage/locking (SQLite), and inference (Ollama) to the host. One macOS process per agent.
- **[INTERACTION.md](INTERACTION.md)** — how the system interacts with the human: the four roles (goal-setter, capability authority, spectator, teacher) and the ask-channel where a process escalates to you for a capability grant.

## Status

**v0.1 — the deterministic kernel core runs.** The fetch-decode-execute-commit loop, the intent ISA, a serializable PCB, a cooperative scheduler with a budget safety-net, a capability-checked syscall dispatcher (the trust boundary), SQLite-backed memory + a single-writer trace, and trace replay. Runs against a deterministic `MockCPU`, and against a real local model via `OllamaCPU` (a 7B model has driven a free-form goal end to end).

**v0.2 — process-per-agent, the hosted-runtime model.** `procd` supervises each agent as a **real macOS process**: it forks the agent, parks it with `SIGSTOP`, `SIGCONT`s it to schedule (cooperative, one expensive CPU), services its syscalls over a per-agent Unix domain socket, and reaps it on exit. The CPU runs in the agent; capabilities and the trace stay in the kernel. Isolation, preemption, and process visibility come from macOS — we reimplement none of it.

**Security — filesystem sandboxing + prompt-injection defense.** An `fs.read` device reads only within allowed roots and tags results trusted/untrusted. The moment untrusted data enters a process's window, the kernel revokes its privileged capabilities, so an injected instruction to persist or spawn is denied at the boundary — not left to the model to resist.

**Interaction — the human ask-channel.** A sandboxed process that needs a privileged capability emits `REQUEST`; the kernel routes it to an `Authority`. Headless that's a policy; interactively it's you (a decision box), and your approval is the grant. A tainted process is auto-denied any privilege re-grant. See [INTERACTION.md](INTERACTION.md).

### Run it

```
cd ~/Code/LLMOS

# smoke test (process completes, trace is correct, replay reconstructs state,
# and a capability-denied write faults without persisting)
PYTHONPATH=. python3 tests/test_hello.py

# in-process: boot, spawn one process, run the built-in "hello" program
#   PLAN -> CALL(clock) -> WRITE_MEM -> YIELD -> RETURN
python3 -m llmos.cli run hello

# in-process, but with a REAL local model emitting the instructions
python3 -m llmos.cli run "get the current time and save it to memory" --ollama

# multi-process: one macOS process per agent, supervised by the kernel
python3 -m llmos.cli runp hello ping

# security: a trusted read succeeds; an injection in an untrusted file is blocked
python3 -m llmos.cli run readgood
python3 -m llmos.cli run readbad

# the ask-channel: a sandboxed process requests a capability
python3 -m llmos.cli run elevate --grant mem.write   # approved -> it writes
python3 -m llmos.cli run elevate                      # default deny -> blocked

# list processes, and reconstruct a run's state from its trace
python3 -m llmos.cli ps
python3 -m llmos.cli replay 1
```

### Module map

- `llmos/kernel.py` — the deterministic fetch-decode-execute-commit loop (+ `commit_external` for out-of-process agents)
- `llmos/isa.py` — instructions/opcodes (`PLAN CALL READ_MEM WRITE_MEM SPAWN YIELD RETURN`)
- `llmos/pcb.py` — the process control block (serializable, checkpointable)
- `llmos/scheduler.py` — cooperative ready-queue + budget preemption
- `llmos/syscall.py` — the syscall dispatcher and capability trust boundary
- `llmos/store.py` — SQLite-backed memory, single-writer trace, process snapshots
- `llmos/cpu.py` — the swappable CPU: `MockCPU`, `ReplayCPU`, `OllamaCPU`
- `llmos/programs.py` — built-in deterministic demo programs
- `llmos/authority.py` — who may grant a requested capability (Deny / Policy / Human)
- `llmos/agent_runner.py` — one agent = one real macOS process, talking to the kernel over a socket
- `llmos/procd.py` — the process supervisor (spawn, SIGSTOP/SIGCONT scheduling, reap)
- `llmos/replay.py` — reconstruct state from the trace
- `llmos/cli.py` — the shell: `run`, `runp`, `ps`, `replay`

## Next

Teach a local model the full ISA so a real LLM can drive multi-step goals reliably; stream the live trace to an attached human (`attach`) and turn corrections into durable protocols; run `procd` as a launchd daemon with a named wrapper (`llmos-kernel`) and shell-out hardening; add a `web` device (untrusted by default) and `sandbox-exec` profiles for web-content agents.
