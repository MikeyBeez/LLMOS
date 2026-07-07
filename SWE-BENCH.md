# SWE-bench on LLMOS

Running [SWE-bench](https://www.swebench.com/) — the benchmark where an agent has to fix real bugs in real Python repositories — with LLMOS driving a **local** model on a **consumer graphics card**, and what that turned up about the model versus the system around it.

## The short version

The same model that posts **73.4% on SWE-bench Verified** on datacenter hardware first read **7%** here. We took it to **34%** (26 of 76 scored) without ever touching the model — every point came from fixing the operating system around it. The model was fine the whole time; the harness was broken in five different ways, and the graphics card was starving it. This is the concrete backing for the write-up, *Benchmarks Measure the Wrong Machine*.

## The setup

- **Model:** `ornith:35b` — a 35B-parameter mixture-of-experts coder (~3B active per token, a thinking model with native tool use). Its published score is 73.4% on SWE-bench Verified.
- **Hardware:** a consumer NVIDIA RTX 5070 Ti, 16 GB VRAM. At the 4-bit quantization needed to fit a model this size, the weights are ~22 GB — larger than the card — so about a third of the model runs on the CPU (36% CPU / 64% GPU) and it's slower. This is the ordinary situation for local AI, not a corner case.
- **Benchmark:** the sympy slice of SWE-bench Lite — all 77 sympy instances. sympy is the one large repo that provisions cleanly without the Docker images, so it's a uniform, sizable sample.
- **No Docker:** instead of the official per-instance Docker images, the harness clones each repo at its base commit and builds a fresh virtualenv itself (see the version checker below).

## How a single instance runs

For each instance the harness does three things and then throws the repo away, keeping only the outcome and the trace:

1. **Set up.** Clone the repo at its `base_commit`, and build an environment with the *right Python* for it (see `envcheck.py`). The model gets only the repo and the GitHub issue text — never the grading tests, the gold patch, or any hint.
2. **Solve.** LLMOS spawns a process whose CPU is the model, driven through ornith's native tool-calling with four tools — a sandboxed shell, file read (with line ranges), a search-and-replace file edit, and finish. The agent works a reproduce → locate → fix → verify loop, and the kernel's memory manager keeps the bug description pinned in the window (see [MEMORY.md](MEMORY.md)). The model's own reproductions and the scorer both run in the instance's virtualenv.
3. **Score.** Apply the instance's test patch, run its `FAIL_TO_PASS` tests, and mark it resolved only if they pass. The repo is then deleted.

## What went wrong, and the fix each time

The climb from 7% to 34% is a list of system bugs, not model improvements:

- **The grader ran the wrong Python.** The biggest one. The virtualenv used a modern Python that the older sympy versions can't even import (`distutils` and `collections.Mapping` are gone in 3.12), so the tests died on an import error *before* evaluating the fix, and the harness scored a correct patch as a failure. `envcheck.py` reads each repo's declared Python support and provisions that exact version on demand with [uv](https://github.com/astral-sh/uv) — no Docker, no sudo. **This alone moved the score from 7% to 25%**, on the same patches the model had already produced.
- **The sandbox rejected valid edits.** The model passed correct repo-relative paths like `sympy/core/expr.py`; the sandbox resolved them against the wrong directory and denied them as "outside the writable roots." Fixed by resolving relative paths against the repo root.
- **The model fought the working directory.** Its shell was already in the repo, but it kept prefixing commands with a 40-character absolute `cd` it couldn't remember and mangled. Fixed by telling it it's already there and never to `cd`.
- **A starved context window dropped the task.** On hard problems the 16 GB card's small window filled up and pushed the issue out of view, so the model explored with no memory of what it was fixing. Fixed with a working-set memory manager (pin the task, drop superseded re-reads, compact old steps into a faithful digest) and a larger 64K window once the memory fit.
- **Analysis paralysis.** Some instances located the bug in the first minute and then read the same file 30+ times without editing. A polite "make the change" is ignorable, so after an exploration deadline the toolset is stripped to edit-and-finish: the model *cannot* keep exploring.
- **The thinking model was cut off mid-thought.** ornith reasons before it acts; the generation cap (2048 tokens) guillotined its reasoning before it emitted the tool call on hard steps — 92% of the "no action" steps ended exactly at the cap. Raising the cap unblocks it, but on this card the fix is itself too slow to run at scale — the hardware limits the model *and* the remedy.

## Result

```
apparent 7%   →   25% (right Python)   →   34% (everything, 64K window)
```

Final: **26 of 76 scored = 34.2%**, same model, same card. Of the 76: 26 solved, 39 edited-but-wrong (the model's genuine ceiling on these hard bugs), and 11 that never produced a patch (mostly the mid-thought truncation).

## Honest caveats

This is not the certified SWE-bench figure and shouldn't be quoted as one. Scoring is our own approximation — per-repo Python via uv rather than the official Docker harness — it's the sympy slice only, and it's a single attempt with no sampling. Treat 34% as a self-consistent internal measurement on consumer hardware. The remaining gap to 73% is precision (4-bit vs full), context (the card can't hold a big window), scaffold maturity, sampling, and real model misses — most of which a consumer cannot change, which is the whole point.

## Files

- `swe_agent.py` — the harness and agent: setup, the native-tool-calling coding loop, the memory manager, and scoring.
- `envcheck.py` — the version checker: pick the repo's Python from its metadata, provision it with uv, and step down on known version-signature failures.
- `swe_select.py` — pick instances from the SWE-bench Lite parquet.
- [MEMORY.md](MEMORY.md) — the working-set / KV-cache memory manager design.

## Run it

```
# on the GPU box, with the model served by ollama and uv installed
pip install uv --break-system-packages
python3 swe_select.py 77                 # write ~/swe/instances.json (77 sympy instances)
PYTHONPATH=~/Code/LLMOS python3 swe_agent.py 77   # run; results stream to ~/swe/results.json
```
