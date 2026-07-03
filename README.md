# LLMOS

An operating system where the **LLM is the CPU**.

Instead of a deterministic silicon CPU executing machine instructions, the execution unit is an LLM forward pass, and programs are goals expressed as structured intents. Everything a real OS does — scheduling, memory management, system calls, interrupts, a filesystem, drivers, IPC, security, fault recovery — still has to exist. LLMOS builds all of it around that one swapped part.

## The core stance

- **The LLM is the CPU** — powerful, stochastic, and untrusted. It executes one instruction per inference call and never touches the world directly.
- **The kernel is a small, deterministic loop** — plain code, no model inside it. It owns every resource, dispatches system calls, manages memory, schedules processes, and writes an auditable trace. It's a microkernel: the intelligence lives in the CPU and in the programs, not in the plumbing.

## Design docs

- **[ARCHITECTURE.md](ARCHITECTURE.md)** — every OS subsystem and its LLMOS design, the central design tension (a nondeterministic CPU), and the boot sequence.
- **[IMPLEMENTATION.md](IMPLEMENTATION.md)** — v1 runs as a hosted runtime on macOS (a process VM, JVM/BEAM-style), delegating boot/supervision (launchd), isolation/preemption (Unix processes + signals), storage/locking (SQLite), and inference (Ollama) to the host. One macOS process per agent.

## Status: v0.1 — the kernel core runs

The deterministic spine is built and tested (`llmos/`): the fetch-decode-execute-commit kernel loop, the intent ISA, a serializable PCB, a cooperative scheduler with a budget safety-net, a capability-checked syscall dispatcher (the trust boundary), SQLite-backed memory + a single-writer trace, and trace replay. It runs today against a deterministic `MockCPU`; the `OllamaCPU` driver is wired for a real local model.

### Run it

```
cd ~/Code/LLMOS

# smoke test (process completes, trace is correct, replay reconstructs state,
# and a capability-denied write faults without persisting)
PYTHONPATH=. python3 tests/test_hello.py

# boot the kernel, spawn one process, run the built-in "hello" program:
#   PLAN -> CALL(clock) -> WRITE_MEM -> YIELD -> RETURN
python3 -m llmos.cli run hello

# list processes, and reconstruct a run's state from its trace
python3 -m llmos.cli ps
python3 -m llmos.cli replay 1
```

### Module map

- `llmos/kernel.py` — the deterministic fetch-decode-execute-commit loop
- `llmos/isa.py` — instructions/opcodes (`PLAN CALL READ_MEM WRITE_MEM SPAWN YIELD RETURN`)
- `llmos/pcb.py` — the process control block (serializable, checkpointable)
- `llmos/scheduler.py` — cooperative ready-queue + budget preemption
- `llmos/syscall.py` — the syscall dispatcher and capability trust boundary
- `llmos/store.py` — SQLite-backed memory, single-writer trace, process snapshots
- `llmos/cpu.py` — the swappable CPU: `MockCPU`, `ReplayCPU`, `OllamaCPU`
- `llmos/replay.py` — reconstruct state from the trace
- `llmos/cli.py` — the shell: `run`, `ps`, `replay`

## Next

Wire the `OllamaCPU` to a model taught the ISA (so a real LLM emits the instructions), then layer the macOS-process isolation from the implementation plan on top of the existing kernel↔agent syscall channel — one macOS process per agent, with `SIGSTOP`/`SIGCONT` scheduling.
