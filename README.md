# LLMOS

An operating system where the **LLM is the CPU**.

Instead of a deterministic silicon CPU executing machine instructions, the execution unit is an LLM forward pass, and programs are goals expressed as structured intents. Everything a real OS does — scheduling, memory management, system calls, interrupts, a filesystem, drivers, IPC, security, fault recovery — still has to exist. LLMOS builds all of it around that one swapped part.

## The core stance

- **The LLM is the CPU** — powerful, stochastic, and untrusted. It executes one instruction per inference call and never touches the world directly.
- **The kernel is a small, deterministic loop** — plain code, no model inside it. It owns every resource, dispatches system calls, manages memory, schedules processes, and writes an auditable trace. It's a microkernel: the intelligence lives in the CPU and in the programs, not in the plumbing.

## Why it's tractable

Most of the subsystems already exist as separate tools and only need a kernel to unify them: a persistent memory store as disk (with semantic recall as the page-fault handler), a tool layer as the system-call interface, triggers as interrupt handlers, a self-paced loop as a scheduler, a ledger as the instruction trace, and capability tiers as the security model. The new work is a small, sharp list — the kernel loop, a serializable process model, a scheduler, a capability-checked syscall dispatcher, a VFS, an eviction policy, and concurrency guards.

## Status

Design draft, v0.1 (spec-first). See **[ARCHITECTURE.md](ARCHITECTURE.md)** for the full enumeration of every OS subsystem and its LLMOS design, the central design tension (a nondeterministic CPU), and the first milestone.

## First milestone

A kernel that boots, spawns one process, runs a short intent program, makes a couple of capability-checked syscalls, writes a replayable trace, yields, and cleanly resumes. "Hello, world" for LLMOS.
