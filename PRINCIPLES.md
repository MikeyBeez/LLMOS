# LLMOS — Design Principles

Two heuristics that decide most design calls in this system. They are not
style preferences; in an agent with a finite turn budget and an expensive
model, violating them is a *failure mode*, not an inefficiency.

## 1. Match the shape of the solution to the shape of the problem

Real problem-solving is a **stack of nested subproblems, not a line**.
Installing X reveals you need Y; fixing a bug reveals a missing dep;
verifying fails and you back up one step. A tool that flattens a nested
problem into atomic calls loses its place and thrashes. So:

- Make the primitive **recursive / stack-shaped** when the problem nests.
  The install tools only worked once they grew `push_subgoal`/`pop_subgoal`
  mirroring the dependency tree. The fix loop (reproduce → patch → verify →
  back up) and env verification (collect → run → missing dep → install →
  retry) have the same shape.
- One behavior lives in **exactly one place**. The moment logic is copied
  into N handlers it drifts into N subtly-different behaviors — the
  "fixed in the scorer but not in `run_tests`" bug. `test_runner.py` is the
  single test-execution primitive for exactly this reason.

Design tell: when a tool thrashes, ask *is it flattening a nested problem?*
(give it a stack) and *is this logic duplicated?* (consolidate it).

## 2. Never pay for the same thing twice

Finite turn budget + expensive model + code that runs in loops means
**wasted compute is subtracted from the budget that decides pass/fail**.
matplotlib and requests failed prior runs not because the model couldn't
fix them but because they *burned the budget on redundant work*. So:

- **Cache** the expensive, idempotent thing. The git mirror buys a repo's
  history once; the resident model avoids reload-thrash; playbooks cache a
  validated build so it is never rediscovered.
- **Search** before you re-derive. If you can describe the problem
  precisely, someone likely solved it — the runner web-searches an unknown
  module's pip name rather than guessing.
- **Batch** independent operations; don't re-run or re-read what is already
  known.

This is "correct once, never again" applied to compute: the remedy store
and playbooks apply it to our own mistakes, web search to everyone else's.

## The strongest integration is structure, not a reminder

Prefer encoding a principle in a **tool** so the wrong thing becomes
impossible (a `clone()` that caches the mirror, a gate that verifies
itself) over a protocol that reminds you, over a doc that hopes you read
it. Reach for the weakest mechanism only when the principle genuinely needs
runtime judgment.
