# LEARNED

What we know, and how we learned it. **Append only.** Newest session at the top.

This is deliberately NOT a status file and NOT benchmark results — those live in
`continuation-notes/` and `~/swe/runs/`. This is the durable part: the findings
that would still be true if we threw the scaffold away and started again.

Each entry says what was believed, what turned out to be true, and what the
evidence was. A finding without its evidence is a rumour.

---

## 2026-07-20

### A map is a translation layer between abstractions

The organising idea of the session. A bug lives at one level of abstraction; the
agent reads at another; between them is a crossing that nothing translates, so
the bug is invisible — the code looks reasonable and the behaviour is wrong.

You need one map per **boundary**, not per subsystem, and there are only about
five. Boundaries found so far:

| boundary | map | status |
|---|---|---|
| symptom → file location | atlas (our own solves, leave-one-out) | working |
| source → runtime data movement | flow map (who stores what, under which key) | working |
| runtime object → rendered output | layout map (block, baseline, binding) | partial |
| issue vocabulary → codebase vocabulary | none | **worst crossing we have** |
| intended edit → argument encoding | **eliminated** (line-anchored patch) | solved |

The last row is the important one. For each boundary you can either supply the
translation **or eliminate the crossing**, and eliminating is cheaper and more
reliable. Line-anchoring didn't help the model escape strings correctly; it
removed the need to, by making the anchor two integers.

A map exposes **the variables the code branches on**, in the subsystem's own
vocabulary — sympy's `baseline`/`binding`, python's `sys.modules[key]`, flask's
`rule.subdomain`. Every bug we examined was a wrong value in one of those.

### Hand-solve the instance BEFORE building anything for it

The method change, and it works. Solving it yourself tells you **what
information was decisive** — and that is the map specification. Deriving a map
from theory produced one that showed the symptom (what the output looks like)
rather than the cause (whether the code can even place a block correctly).

Two hand-solves, two completely different outcomes, both worth having:

- **sympy-23191** (12 prior misses) — solvable in four steps. Yielded a new
  general map: does this layout code compose BLOCKS or splice STRINGS?
  `compose_map.py` flags the broken printer and clears the healthy ones.
- **sympy-21171** (9 prior misses) — not solvable from the repository. Yielded a
  knowledge-base entry, a caveat on an existing rule, and a decision to stop
  grinding on it. Worth as much as the solve.

**Leakage guard:** the solve tells you WHICH general map to build, never what
the map contains. Test: could this map have been built without seeing the
answer? `compose_map.py` runs on any function in any repo and has no idea 23191
exists.

### Some knowledge is not in the repository and cannot be derived from it

sympy-21171 needed a LaTeX typesetting rule documented by the AMS: `\left` and
`\right` auto-size, and nesting them compounds it, so inner delimiters must be
plain or explicitly sized. **All 85 sibling methods in the same file violate it**
— they get away with it because none of them contain inner delimiters.

So "read the neighbours" — the rule that fixed flask-4992 the same day — points
exactly the wrong way here. The refinement: match the neighbours for **naming
and structure**; follow the published spec for **format correctness**.

Externally-cited rules are also the *safest* knowledge-base content, because
provenance is public and verifiably not derived from a hidden test. That is a
stronger guarantee than "I audited it and it reads general."

### The agent's self-written test carries almost no information

Measured over 175 runs: 30% false positive, 33% false negative. Base resolve
rate 63%; given a green self-check 70%; given red 54%. **Seven points of
information.** Median one assertion per reproduction; 56 of 175 had none.

`form_rank` was weighting those signals 14 of 15 points when choosing between
attempts. We were selecting on noise.

Consequence: decide with the tests you were GIVEN — the repo's own base-commit
tests, and at scoring time the full canonical set. `score()` had been running
`FAIL_TO_PASS` only, which is 1 of 7 given tests at the median.

### The agent could see the fix and not express it

Text anchors containing a backslash failed **77%** of the time (n=338) against
**19%** for anchors without (n=1147). `old_snippet not found` is the single most
common error in the whole system: 259 occurrences across 175 runs.

The mechanism, caught verbatim: the agent knew exactly which space to delete and
**over-escaped** when re-emitting the line — it correctly doubled a backslash
and then also escaped a quote that needed nothing.

Fixing the interface rather than the model: xarray-5131 went 2528s / 15 patches
/ 15 failures → **453s / 7 patches / 4 failures**, and the model reached for
`start_line` unprompted.

### A third of our wins never knew they had won

36 of 104 resolved runs ended by **exhausting the turn budget**, not by
declaring done. The patch was already correct and the agent kept working.

Turn count does not discriminate: at p90, resolved and missed runs both sit at
74 turns. So a shorter leash cuts wins at nearly the rate it cuts wandering.
What is missing is not a limit but a **completion check** — nothing ever asks
whether there is any remaining reason to continue.

### Classify by cause, not by the words in the report

First attempt bucketed instances by keywords in the issue text and concluded
deprecation/syntax was ~3%. That measured symptom vocabulary: anything
containing a traceback became "crash", and the hardest printing instance matched
nothing at all and became "other".

Classifying by what the **gold patch does** instead:

| fix shape | never solved | solved ≥once | fail rate |
|---|---|---|---|
| signature / API lag | 4 | 1 | **80%** |
| output format string | 19 | 9 | **68%** |
| substantial rewrite | 12 | 7 | 63% |
| condition / branch | 53 | 46 | 54% |
| import / registration | 5 | 6 | 45% |

Reports describe symptoms. Patches describe causes.

### Where the hard tail actually is

122 of 300 instances have never resolved. **sympy is 47 of 115 eligible**, and
**23 of sympy's 48 are printing/rendering** — about 19% of the entire
never-resolved tail. The failure rate there is no worse than other categories;
there is simply far more of it. The win is volume, not difficulty.

### Nothing was looking at our failures

144 archived misses had produced 11 research records. **92% of failures were
never examined by anything.** The escape signal had been sitting in 25 separate
failures for over a week.

Post-mortems are now automatic on every run, wins and losses alike, so the
corpus carries its own control group.

### Verify, don't remember (about myself)

Six times in one session I asserted a field or function name from memory and was
wrong: `h_submit` being on the dispatch path (it isn't — `submit` maps to
`RETURN` and the handler never runs), `_run` returning a dict (it returns
`CompletedProcess`), `_clip_result` (never existed), events carrying
`result`/`exit` (they carry `ok`), "we never ran a full pass" (161 instances at
40.4% were on disk), `inst["patch"]` (the field is `gold_patch`).

**Every one was caught by running something. None by thinking harder.** Grep and
a script that reads the file were right first time, every time.

Countermeasure: `schema.py` — one canonical space for all four data shapes, with
`sget()` that raises with a did-you-mean instead of returning `None`. A magic
string key is an unvalidated boundary crossing, the same hazard as the agent's
`old_snippet`, and the same fix applies: make the boundary explicit rather than
reasoning harder about it.

### Silent failure is worse than loud failure

A collector wrapped in `except Exception: pass` wrote nothing for an entire run
because of a missing `import time`, and looked healthy throughout. Not breaking
the caller and failing silently are **different requirements**. Return the error;
let the caller surface it.

Same shape: four outcome fields were assigned *after* the trace was serialised,
so every saved trace had an incomplete outcome — which reads exactly like a
feature that never ran, and fooled me into reporting the invariant probe as dead
when 48 of 57 runs had locked one.
