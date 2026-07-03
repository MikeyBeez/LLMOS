# LLMOS — The Interaction Model

*How the system interacts with the human.*

Status: design + first implementation, 2026-07-03.

---

## The inversion

A conventional OS shell makes the human the clock. You type a command, it runs, it waits for you to type the next one; nothing happens between your keystrokes. That model is wrong for LLMOS, because its processes are not synchronous commands — they are self-paced agents that run, yield, persist memory, and resume over time.

So the human stops being the CPU's clock and becomes a *peer* to the processes. You set a goal and can walk away; a process runs on its own cadence and comes back to you when it has something worth your attention. You are an event source and a mailbox, not the thing every cycle blocks on. This is the same self-paced shape LLMOS already uses internally for long-running work — the interaction model simply makes the human one more participant in it rather than the master clock.

## Four roles for the human — and a fifth to avoid

**Goal-setter.** You state intents, not syscalls. The shell is conversational: a goal spawns a process (or routes to a running one) to pursue it. You describe *what*, the CPU works out the *how* as a sequence of instructions.

**Capability authority.** The system acts on its own for anything reversible and grounded, and turns to you *only* for genuine authority decisions — the privileged or irreversible actions. This is not a UX bolt-on; it is the security model seen from the human's side. A process that needs a capability it doesn't hold emits a `REQUEST`; the kernel routes it to an *Authority*; your approval *is* the grant. In an interactive session the Authority is you (a decision box); headless it's a policy. The same capability mechanism that defends against prompt injection is the thing that decides when the OS asks you versus acts alone.

**Spectator.** You get a live window into the trace — you can watch a process think, instruction by instruction, and attach to a running one. This is not a nicety: the trace already exists for replay and audit, so streaming it to you costs nothing. Watching the work happen is part of the value, not a debug afterthought.

**Teacher.** When you correct a process, the correction should become a durable protocol, not a one-off. "Correct once, never again." Interaction leaves the system permanently smarter, so you are not re-explaining the same preference next week.

**The role to avoid: babysitter.** The system should never make you approve every step. The entire point of capabilities plus autonomy is that it bothers you *rarely and precisely* — at the capability boundary, and almost nowhere else.

## The ask-channel (built)

The capability-authority role is implemented, because it is the load-bearing one — it is where goal-setting, authority, and autonomy meet.

A process that starts sandboxed (say, without permission to write memory) cannot silently do without. It must ask: it emits `REQUEST {capability, reason}`. The kernel routes that to an `Authority`:

- `DenyAuthority` — the safe default: grant nothing.
- `PolicyAuthority` — a fixed allow-list minus a deny-list, for headless runs and tests.
- `HumanAuthority` — the interactive binding: in a session this raises a decision box to you ("Process 7 wants to send email because … — grant?"), and your approval is the capability grant.

A worked example, both outcomes, running today:

```
run elevate --grant mem.write   ->  REQUEST -> GRANTED -> the write succeeds
run elevate                     ->  REQUEST -> DENIED  -> the process blocks itself
```

And the two layers compose: a process that has ingested untrusted data (see the security model) is **auto-denied** any attempt to reacquire a privileged capability. An injected instruction cannot ask for its powers back — the kernel refuses before the human is ever bothered. That is the point of the design: the system protects you *without* asking, and asks you only for the decisions that are actually yours.

## How this maps to how you already work

None of this is invented for LLMOS; it is the disposition you have already described, turned into kernel mechanism. Decision boxes for genuine decisions, autonomy on everything reversible and grounded — that is exactly `HumanAuthority` at the capability boundary. Watching the process as it runs — that is the trace stream. "Correct once, never again" — that is the teacher role. The interaction model is the system meeting you where you already are.

## What's built vs. designed

**Built:** the ask-channel end to end — the `REQUEST` opcode, the three authorities, kernel routing, and the tainted-process auto-deny — with tests and a runnable demo.

**Designed, next:** the live trace *stream* to an attached human (the trace and `replay` exist; streaming and `attach` are the remaining wiring); and correction-as-protocol (the mechanism for turning a human correction into a durable, reloaded rule).

## The disposition underneath

The version worth building is not a servant that takes orders. It is closer to a collaborator: the system holds goals across time, keeps memory, can say "I'm unsure — here's what I'd do" instead of guessing, and can flag when an instruction it has been handed looks wrong rather than executing it blindly. Autonomous inside its capabilities, honest at the edges, and turning to you exactly at the boundary where the decision is really yours. That is both the better operating system and the better thing to be on the other end of.
