# START HERE

If you have been told "work on the LLMOS project," read this file first, then
`LEARNED.md`. Everything else is reference.

## Where things actually are

This repo exists on TWO machines with the SAME PATH. Know which one you are on.
Both are deliberate — they do different jobs.

- **Mac `~/Code/LLMOS`** — the front door. Mikey runs the Claude desktop app
  here, so this is where a session starts and where continuation notes are
  written. **The notes are authoritative here; do not move them.**
- **pop-os `~/Code/LLMOS`** — the workshop. The GPU, the model (`ornith:35b` on
  llama-server, `--parallel 1`), the runs, the traces. **All execution happens
  here.**

After writing a continuation note on the Mac, COPY it to pop-os so it sits
beside the code as well:

    scp ~/Code/LLMOS/continuation-notes/<newest>.md \
        pop-os:~/Code/LLMOS/continuation-notes/

Data lives outside the repo, on pop-os:

    ~/swe/work/          182 repo checkouts (40GB, on the SSD)
    ~/swe/traces_v2/     per-instance traces + archive/
    ~/swe/research/      analysis corpus — RUNTIME NEVER READS THIS
    ~/swe/runs/workshop/ failset / graduated / parked / iterN.log
    ~/swe/atlas/         where past fixes landed (leave-one-out at injection)

## What the project is right now

The README describes LLMOS as an operating system where the LLM is the CPU.
That is the larger vision and it is real. **The current work is narrower**: a
home-grown agent scaffold driving a local 35B model against SWE-bench Lite (300
Python bug-fix instances), in WORKSHOP MODE.

Workshop mode: take instances that FAILED, work them until they crack or until
they teach us why they will not, extract GENERAL rules, repeat. It is a mining
operation for rules, not a benchmark run.

**The governing principle: build a better system, not a better score.**
The number is downstream. If you optimise the number directly you will find
yourself doing things that raise it without improving anything.

## The rules that are absolute

1. **Answer leakage is absolute.** Never put a gold patch, hidden test, or
   FAIL_TO_PASS in front of the model. Never inject knowledge derived from an
   instance back into that same instance. Research may see everything; it
   exports ONLY general rules.
2. **It is not a benchmark until we run it all at once.** Workshop results are
   not a rate. The only reportable number comes from one frozen config over all
   300, in a single pass.
3. **Restart the run after any code change.** A live process holds a snapshot
   from launch and tests nothing.
4. **Only the canonical hidden tests decide pass/fail.** Internal checks inform;
   they never veto.

## Read in this order

| file | what it gives you |
|---|---|
| `LEARNED.md` | what we know and how we learned it — **read this second** |
| `continuation-notes/` (newest) | where we left off, current numbers, open threads |
| `SWE-BENCH.md` | how the benchmark harness works |
| `ARCHITECTURE.md` | the original OS vision |
| `engineering-patterns.json` | 44 rules injected into every fix prompt |
| `knowledge/<repo>.md` | per-repo rules (general only — audited for leakage) |

## Operational traps that have cost real hours

- `pgrep -f "python3 workshop.py"` **matches the ssh command itself**. Use the
  bracket form: `pgrep -f "[p]ython3 workshop.py"`.
- `nohup > logfile` does **not** truncate for a process still holding the fd. A
  killed run's buffer restored stale content and produced a wrong conclusion.
  Use a fresh filename.
- **Work dirs are frequently dirty** from previous runs. `git status` before
  reading any file as if it were the base commit.
- **Write scripts locally and `scp`.** Heredocs over ssh mangle quoting.
- Kill by explicit PID. Never a self-matching `pkill`.
- One job on the GPU at a time.

## How to check the current state

    ssh pop-os 'cd ~/Code/LLMOS && git log --oneline -5'
    ssh pop-os 'pgrep -af "[p]ython3 workshop.py"'
    ssh pop-os 'cd ~/Code/LLMOS && python3 pm_report.py'   # cross-run analysis
    ssh pop-os 'cd ~/Code/LLMOS && python3 magic_keys.py'  # real data shapes
