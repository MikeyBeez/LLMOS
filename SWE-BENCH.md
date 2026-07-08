# SWE-bench on LLMOS

Running [SWE-bench](https://www.swebench.com/) — the benchmark where an agent has to fix real bugs in real Python repositories — with LLMOS driving a **local** model on a **consumer graphics card**, and what that turned up about the model versus the system around it.

The whole point of the accompanying essay (*Benchmarks Measure the Wrong Machine*) is that a benchmark that hides the system it ran on is useless to a person on different hardware. So this document tries to hide nothing: the exact configuration, every value it moved through, what went wrong, how we compensated, and how we found each problem. If any of it reads as tedious, that tedium is the argument.

## The short version

The same model that posts **75.6% on SWE-bench Verified** on datacenter hardware first read **7%** here. We took it to **34%** (26 of 76 scored) without ever touching the model — every point came from fixing the operating system around it and the environment it ran in. The model was fine the whole time; the harness was broken in several ways, and the graphics card was starving the context.

## What we ran on

- **Model:** `ornith:35b` — **Ornith-1.0-35B** from DeepReinforce (released June 2026), a self-scaffolding coding model (it learns its own agentic scaffold via RL) built on the `qwen35moe` mixture-of-experts architecture: 34.7B total parameters, ~3B active per token (about 8 experts), a reasoning ("thinking") block before it acts, native tool-calling. Quantization `Q4_K_M`, ~22 GB on disk. Native context length **262,144** tokens. Published score **75.6% on SWE-bench Verified** — the family runs from 69.4% (9B dense) to 82.4% (397B flagship). There is an irony worth naming: the published number partly reflects the scaffold the model *taught itself*, and here we replaced that with a hand-built one (LLMOS) on hardware that can barely hold the model.
- **GPU box ("pop"):** consumer NVIDIA RTX 5070 Ti, **16,303 MiB VRAM** (~16 GB), 31 GB system RAM, roughly a $750 card. The 22 GB of weights do not fit in 16 GB, so ollama runs the model split **36% on the CPU / 64% on the GPU** — about 8 GB spilled to system memory. It is slower for it. This is the ordinary situation for local AI, not a corner case.
- **Inference server:** ollama (llama.cpp underneath), temperature 0 and fixed seed for determinism.
- **Benchmark:** the sympy slice of SWE-bench Lite — all **77 sympy instances**. sympy is the one large repo that provisions cleanly without the Docker images, so it is a uniform, sizable sample.
- **No Docker:** instead of the official per-instance Docker images (whose big-layer pulls stalled on this box anyway), the harness clones each repo at its base commit and builds a fresh virtualenv itself.

## The configuration, and every value it moved through

These are inference and harness knobs, not model changes. They are the "system" the essay says a benchmark must disclose. Pulled from the git history:

**Context window (`num_ctx`).** ollama's server default is 4096; the base `OllamaCPU` used 8192. The SWE agent started at **32768**, was cut to **16384** ("fits 5070 Ti better; model is CPU-split at 22 GB"), and for the final run raised to **65536** (64K). For a stretch we did not even know our own effective window — ollama reported the model loaded with a small allocation while its API claimed the larger number, and it took direct inspection with `ollama ps` to confirm what the model was actually running with. If the people operating the system can't tell what context it's using, that is the opacity problem in miniature. The model's native ceiling is 262,144, but on a 16 GB card the KV cache (~80 KB/token) makes anything near that impossible: 64K of KV is ~5 GB on top of the 22 GB of weights, and 256K would be ~20 GB of KV alone.

**Generation cap (`num_predict`).** Base default 512. The agent went **1024 → 2048 → 8192**. The jump to 8192 was the last bug we found: ornith is a thinking model, and at 2048 it was being cut off mid-reasoning before it emitted its tool call — 92% of the "no action" steps ended at exactly 2048 tokens. Raising the cap unblocks it, but at 64K on this card the fix is itself too slow to run at scale (each hard step generates up to 8192 tokens on a CPU-split model — minutes per step), so the final scored run used 2048 and we documented 8192 as a validated-but-unaffordable fix.

**Keep-alive.** Default 30m; the final run used **24h** so the 64K model stayed resident. Reloading it at 64K costs ~50 seconds, and mid-run evictions were stalling the batch — a single reload looked exactly like a hang.

**Turn budget.** Max agent steps per instance went **14 → 40 → 30 → 40**.

**Forcing / memory knobs.** `FORCE_AFTER = 8` (nudge to edit after 8 tool calls with none); `EDIT_DEADLINE = 16` (hard: strip the toolset to edit-and-finish). The memory manager's watermark went **8000 → 9000 (with a 6-step compaction chunk) → 48000** for the 64K run.

## How an instance runs

For each instance the harness does three things, then deletes the repo and keeps only the outcome and the trace:

1. **Set up.** Clone at `base_commit`, and build a virtualenv with the *right Python* for that repo version (`envcheck.py`). The model gets only the repo and the GitHub issue text — never the grading tests, the gold patch, or any hint.
2. **Solve.** LLMOS spawns a process whose CPU is the model, driven through native tool-calling with a sandboxed shell, a line-range file read, a search-and-replace edit, and finish. It works a reproduce → locate → fix → verify loop; the memory manager keeps the issue pinned; the model's reproductions and the scorer both run in that virtualenv.
3. **Score.** Apply the test patch, run `FAIL_TO_PASS`, resolved only if they pass.

## The traces were the instrument

None of the fixes below were guesses. Every instance writes a full execution trace — every instruction the model emitted, every syscall result — plus per-step prompt and generation token counts, saved to disk (`~/swe/traces/*.trace.json`) even though the repo is thrown away. Reading those traces is how we found each problem: the empty tool-call list that exposed the JSON-escape break, the "outside the writable roots" denials on valid paths, the count showing 87% of file reads were being truncated and 61% were re-reads, the per-step token curve showing 8 instances hitting the context ceiling, the trace of one instance reading `power.py` 38 times without editing, and the token histogram showing 92% of stalled steps ending exactly at the generation cap. A benchmark that reports only a final score throws this away. Keeping it is what let us tell the model apart from the system.

## What went wrong, and the fix each time

The climb from 7% to 34% is a list of system bugs, not model improvements:

- **The grader ran the wrong Python.** The biggest one. The virtualenv used a modern Python that older sympy versions can't import (`distutils` and `collections.Mapping` are gone in 3.12), so tests died on an import error *before* evaluating the fix, and a correct patch scored as a failure. `envcheck.py` reads each repo's declared Python and provisions that exact version on demand with [uv](https://github.com/astral-sh/uv) — no Docker, no sudo — with a step-down retry on known version signatures. **This alone moved the score from 7% to 25%**, on patches the model had already produced.
- **The tool transport broke on quotes.** The first agent packed shell commands into hand-escaped JSON strings; nested quotes broke the parse, so the model's actions never executed (empty call lists). Fixed by switching to ornith's native tool-calling.
- **The sandbox rejected valid edits.** The model passed correct repo-relative paths (`sympy/core/expr.py`); the sandbox resolved them against the wrong directory and denied them. Fixed by resolving relative paths against the repo root; also made the edit whitespace-tolerant.
- **The model fought the working directory.** Its shell was already in the repo, but it kept prefixing a 40-character absolute `cd` it couldn't remember. Fixed by telling it it's already there and never to `cd`.
- **Reads were truncated and re-read.** 87% of file reads were being clipped to ~1,500 characters (a bug where a "budget" parameter was ignored), so the model edited half-blind and re-read the same file over and over — 61% of all reads were re-reads. Fixed by honoring the read budget and adding line-range reads (`grep -n`, then read that window).
- **The context window dropped the task.** On hard problems the window filled and pushed the issue out of view. This is the one we fought hardest — see the next section.
- **Analysis paralysis.** Some instances located the bug in the first minute and then read the same file 30+ times without editing. A polite "make the change" is ignorable, so after `EDIT_DEADLINE` tool calls the toolset is stripped to edit-and-finish — the model *cannot* keep exploring.
- **The thinking model was cut off mid-thought.** Covered above under `num_predict`.

## How we tried to compensate for the small context window

The 16 GB card is the root constraint, and most of the engineering was working around it rather than removing it. In order:

First we tried to make the small window go further. The memory manager ([MEMORY.md](MEMORY.md)) treats the context as a managed working set instead of an append-only log: it pins the issue so it can never be evicted, drops superseded re-reads (keyed by file *and* line range, so distinct windows of a file coexist but exact duplicates are dropped), and once the verbatim tail crosses a token watermark it compacts the older steps into a short faithful digest — "searched A and B, the bug is at C lines 120–140, edit X failed" — while keeping the full versions recallable. The compaction is quantized into blocks so the digest (and therefore the KV-cache prefix) stays stable between jumps rather than churning every step. The mechanism is "reseed a clean context, don't do cache surgery": you never edit the middle of the cache, you rebuild a smaller prompt and let the engine recompute only the suffix. We also added line-range reads specifically so the model could pull a 40-line window instead of a whole file, which is the biggest single thing keeping the window small.

Then, once the memory manager kept the working set lean, we could afford to raise the raw window — first from 16K, finally to 64K — which removes context-shift entirely: the model can no longer lose the issue. The measured payoff was real but partial. On the eight instances that had provably saturated the 16K window, bounding and enlarging the context converted a couple of previously-hopeless zero-patch failures into solves and stopped all of them from hitting the wall — but it did not lift the resolve rate on that hard set, because those particular bugs are ones the model gets wrong even when it can see everything. Context management is a genuine correctness lever; it is not a substitute for a model that knows the fix, and it is not a substitute for enough memory to run the model properly in the first place.

## Result

```
apparent 7%   →   25% (right Python)   →   34% (everything, 64K window)
```

Final: **26 of 76 scored = 34.2%**, same model, same card. Of the 76: 26 solved, 39 edited-but-wrong (the model's genuine ceiling on these hard bugs), and 11 that never produced a patch (mostly the mid-thought truncation, which is only affordable to fix on hardware we don't have).

## Honest caveats

This is not the certified SWE-bench figure and shouldn't be quoted as one. Scoring is our own approximation — per-repo Python via uv rather than the official Docker harness — it's the sympy slice only, and it's a single attempt with no sampling. Runs on a temperature-0 mixture-of-experts model are not perfectly deterministic, so individual instances flip between runs; treat 34% as a self-consistent internal measurement on consumer hardware, not a leaderboard number. The remaining gap to 75.6% is precision (4-bit vs full), context (the card can't hold a big window at full speed), scaffold maturity, sampling, and real model misses — most of which a consumer cannot change, which is the whole point.

## Files

- `swe_agent.py` — the harness and agent: setup, the native-tool-calling coding loop, the memory manager, and scoring.
- `envcheck.py` — the version checker: pick the repo's Python from its metadata, provision it with uv, step down on version-signature failures.
- `swe_select.py` — pick instances from the SWE-bench Lite parquet.
- [MEMORY.md](MEMORY.md) — the working-set / KV-cache memory manager design, in full.

## Run it

```
# on the GPU box, with the model served by ollama and uv installed
pip install uv --break-system-packages
python3 swe_select.py 77                 # write ~/swe/instances.json (77 sympy instances)
PYTHONPATH=~/Code/LLMOS python3 swe_agent.py 77   # run; results + traces stream to ~/swe/
```
