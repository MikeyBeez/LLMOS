# Environment Creation Knowledge Base

Hard-won notes for building a working Python environment from a freshly cloned
repository. Read top-to-bottom the first time; after that, jump to the section
you're fighting with. Add a note the moment you learn something — the whole point
is that nobody should have to rediscover a gotcha twice.

**Convention for adding notes:** append to the relevant section. If it's about one
library or repo, put it under *Per-repo / per-package notes* with the name in bold.
Date-stamp anything you're unsure about so a future reader can weigh it.

---

## 0. The one rule that matters most

**The #1 cause of a broken env is the wrong Python version.** Fix the interpreter
first; a surprising share of "mysterious" build and test failures simply evaporate.

And beware the trap: **`env builds + imports` does NOT mean `env is correct`.** A
wrong-Python environment can pass a smoke test and still change how the code and
its tests behave (import machinery, deprecation handling, dependency resolution).
"env_ok" is necessary, not sufficient.

---

## 1. Choosing the Python version

In rough order of trust:

1. **Explicit pins** — `.python-version` (pyenv), `runtime.txt` (Heroku-style). If present, believe them.
2. **CI matrix** — `.github/workflows/*.yml`, `.travis.yml`, `tox.ini` `envlist`. This is what the project *actually tested* against; the strongest evidence.
3. **Packaging metadata** — `requires-python` / `python_requires` (a floor) and `Programming Language :: Python :: 3.x` classifiers (the supported set). These give a *range*, not the answer.
4. **Release-date bound** — never pick a Python that did not exist when the code was written. Repos are usually run on a *conservative* (older) Python, not the newest available.

The declared *floor* is often the right pick (e.g. scikit-learn's `NUMPY_MIN_VERSION`
comment literally says "match oldest-supported-numpy for the minimum Python").

Tool: **`pyselect`** derives this from the repo's config with no external table. Its
*supported set* brackets the true answer ~85% of the time; its single best-guess is
a tunable heuristic. Use the set as the reliable output.

---

## 2. Provisioning the interpreter — which backend

- **uv** (`python-build-standalone`) provides **3.8 and newer only**. It **cannot** produce Python **3.6 or 3.7** — they simply aren't in its managed builds.
- For **3.6 / 3.7**: mint the interpreter with **micromamba / conda-forge** (verified: `python=3.6.15` installs fine). Then use pip for the packages.
- This box has no other 3.6/3.7 source: system pythons are only 3.10 and 3.12, and there is no pyenv or deadsnakes.

**Rule of thumb:** `3.8+` → uv. `3.6 / 3.7` → micromamba for the *interpreter only*, then pip.

Conda's job is deliberately kept small: hand over the interpreter, and supply a
package that PyPI can't give you — no wheel for this Python, a version that's been
**yanked or removed from PyPI**, or an sdist that no longer builds (see §4). It is
not the default package manager.

---

## 3. Installing packages

- **Prefer pip + wheels.** PyPI manylinux wheels cover essentially everything down to cp36. Verified: sklearn 0.22.2 / pandas 1.1.5 / numpy 1.19.5 / scipy 1.5.4 all install as cp36 wheels with zero compilation.
- **Do not let loose `>=` bounds float to the newest release.** `numpy>=1.17` resolves to numpy 2.x today and breaks old code — the "too new" disease, one layer below the Python version. Pin instead.
- **The pin rule:** `== max( declared floor , oldest version with a wheel for the pinned Python )`. The floor alone can be un-installable (see §4); the wheel index is the reality check.
- **Optional test deps are not pulled automatically.** For scikit-learn, pandas and matplotlib are *optional* extras, not core deps — `pip install scikit-learn` does **not** bring them, so pandas-gated tests silently skip. Install the needed test deps explicitly (or the repo's `tests` extra).
- **Install a curated minimal set, not the whole `.[tests]` extra.** The full extra drags in fragile old tooling (e.g. `scikit-image==0.16.2`, `pyamg==4.0.0`) that may have no wheel for the pinned Python and will derail the build. Install only what the target tests need (for sklearn: numpy, scipy, pandas, matplotlib, pytest, joblib, threadpoolctl, cython).

---

## 4. When you must build from source  *(Mikey's rule)*

If a package has **no wheel** for the pinned Python, pip will try to compile it —
and you have to let it, correctly:

- Confirm first: `pip download --python-version 3.9 --only-binary=:all: <pkg>==<ver>`. If it says "no matching distribution," there's no wheel.
- Build with **`pip install --no-build-isolation`** (and `--no-use-pep517` for some old packages), with **cython + setuptools + a compatible numpy already installed** in the env. This is exactly what the SWE-bench spec does for old scikit-learn.
- Known concrete cases: `numpy==1.19.2` and `scipy==1.5.2` have **no cp39 wheel** (cp39 numpy wheels start at 1.19.3, scipy at 1.5.4). If you need those exact versions on 3.9, they compile. If you only need "that series," bump to `1.19.5` / `1.5.4` and install clean.
- **Sometimes the version is simply gone.** Old releases can be *yanked* (still installable if you pin the exact version, but not auto-selected) or, more rarely, *removed* from PyPI outright — and old `sdist`s frequently no longer build with a modern toolchain. When PyPI can't hand you an installable old version, **conda-forge** usually still has a prebuilt binary for it. That — not just a missing wheel — is a primary reason to fall back to conda.
- The wheels that are *genuinely* missing on Ubuntu are the ones wrapping a **system C library** — the classic is **PyAudio** (needs `apt-get install portaudio19-dev` first). Numeric / web / plotting / doc packages almost always ship manylinux wheels.

---

## 5. Gotchas — "wrong-floor" errors that lie about their cause

- **Warnings-as-errors repos (astropy).** A benign `DeprecationWarning` from a *dependency* becomes a fatal collection error and masks any fix. Real case: installing a too-new matplotlib pulled a pyparsing whose `oneOf` is deprecated → astropy's `conftest` (which imports matplotlib when present) died at collection, failing the instance regardless of the code. Lesson: don't install deps a repo doesn't need, and version-match the ones it does.
- **Missing system library masquerading as a Python error.** `fatal error: portaudio.h: No such file or directory` looks like a pip/package/Python problem; it's an un-installed *system* lib pip can't see. Always check one floor down.
- **`importorskip` silently skips.** When an optional dep (e.g. pandas) is absent, gated tests are *skipped*, not failed — and a naive scorer counts a skip as a miss with no error message. If graded tests "all skipped," suspect a missing optional dep, not a bad fix.
- **Inline regex flags moved.** Unrelated but same genre: `(?i)` mid-pattern raises "flags not at start of expression" on newer Pythons; use `re.IGNORECASE`. (Shows up as a fatal error two layers from the real code.)

---

## 6. Per-repo / per-package notes

- **scikit-learn** — needs numpy, scipy, pandas, matplotlib for its tests; pandas is a *test* extra (floor 1.0.5), not core. Era split: v0.20–0.22 → Python **3.6**; v1.3 → **3.9**. `NUMPY_MIN_VERSION` for v1.3 is `1.19.2`, which has no cp39 wheel → use `1.19.5`. The v1.3 pandas bugs cluster around the new `set_output` / `transform_output="pandas"` feature and nullable dtypes.
- **astropy** — treats warnings as errors; its `conftest.py` imports matplotlib if it's installed. Do not add a too-new matplotlib. Canonical Python for the 5.x era is 3.9.
- **matplotlib** — treats warnings as errors (`filterwarnings = error`). matplotlib 3.x calls pyparsing's camelCase API (`ParserElement.enablePackrat`, `setParseAction`); pyparsing **>=3.1** raises `PyparsingDeprecationWarning` on those, fatal at import → pytest reports `found no collectors` and a CORRECT patch is scored as a miss (false negative). The scorer now auto-pins `pyparsing<3.1` for matplotlib via `pin_warn_as_error_deps()` in `swe_agent_v2.py` (runs inside `score()` before FAIL_TO_PASS). Docker-confirmed false negatives (authoritative): 23964, 23987, 24149. Home-verified but PENDING Docker confirmation: 23913. Docker-confirmed REAL miss: 24265. Reclaim via the scoring-layer tool in §8. Pattern is general: for any warnings-as-errors repo, an unrelated too-new *pure-python* dep can convert collection into a false negative — pin it era-compatible in `WARN_AS_ERROR_DEP_PINS`. Canonical era: matplotlib 3.5–3.6 → Python 3.9–3.11.
- **pytest** — import-machinery bugs (e.g. importlib double-import) are *Python-version sensitive*; run on the canonical interpreter or the bug won't reproduce as intended. 8.x era → 3.9.
- *(add rows as you meet new repos)*

---

## 7. Meta

- SWE-bench publishes the exact environment per instance (`MAP_REPO_VERSION_TO_SPECS`: Python + packages + install command). Using it is **legitimate** (the environment is given; only the gold patch / test patch / FAIL_TO_PASS are off-limits). But the *better* system **derives** the env from the repo's own evidence so it generalizes to any repo — use the spec only as the answer key to grade the deriver against, never as its input.


---

## 8. False-negative reclaim (scoring-layer correction)

Some correct model patches are scored UNRESOLVED because of a harness/env artifact (chiefly the warnings-as-errors collection error in §6). These are FALSE NEGATIVES, not real misses. The **authoritative** arbiter is the SWE-bench Docker eval (`swebench.harness.run_evaluation`): re-run the model's OWN prediction patch there; if it resolves, the home miss was a false negative.

Workflow (general, answer-leakage-safe — operates only on public instance ids + the resolved flag, never on gold/test patches):
1. Docker-audit suspected FNs (`--predictions_path` = the model's own patches, `--max_workers 2`).
2. For each Docker-CONFIRMED FN, add an entry to `~/Code/LLMOS/swe_false_negatives.json` with `docker_confirmed: true` and an `evidence` pointer to the audit artifact. Record Docker-confirmed real misses under `confirmed_real_misses` so future cycles don't re-audit them.
3. At/after end-of-run, run `python3 ~/Code/LLMOS/reclaim_false_negatives.py --out <results>.corrected.json`. It flips only `docker_confirmed:true` records, writes a SEPARATE corrected file (it NEVER overwrites the live results — the runner rewrites the whole file per instance and would clobber an in-place edit), and prints before/after scores. `docker_confirmed:false` entries (e.g. home-verified only) are reported PENDING and left untouched.

Current manifest: reclaim now — 23964/23987/24149; PENDING Docker confirmation — 23913; confirmed real miss — 24265. Extending to a new repo: Docker-confirm first, then add the entry (do not reclaim on home-reproduction alone).

---

## 9. Docker-eval is not free — guard it (cold cache + silent deadlock)

The authoritative Docker eval (§8) is the ONLY thing that can confirm a false negative, but it is fragile in unattended overnight cycles:

- **Cold cache.** The docker image store gets pruned to 0 between sessions (observed 2026-07-15: `docker system df` → Images 0, Build Cache 0). When cold, a single matplotlib audit must rebuild base+env+instance images from scratch (many minutes). Do NOT assume a prior audit's images survive to the next cycle.
- **Silent deadlock.** An unguarded `run_evaluation` can hang indefinitely with NO output: all Python threads parked in `futex_do_wait`, ~1% CPU, no docker build activity, stale CLOSE-WAIT sockets to HuggingFace (dataset load). In that state it burns the entire cycle and produces nothing.

Mitigation: launch audits through `~/Code/LLMOS/docker_eval_guard.sh`. It validates the predictions file, preflights the image cache (warns when cold), enforces a hard `timeout` (SIGTERM→SIGKILL), force-removes leftover `sweb.*.<run_id>` containers on timeout, and returns **exit 2** for timeout/deadlock vs **exit 1** for a genuine eval failure — so a hang fails fast and visibly. Example:

    ~/Code/LLMOS/docker_eval_guard.sh --preds ~/swe/mpl_preds_23913.json --run-id mpl23913 --instances matplotlib__matplotlib-23913 --timeout 2400

23913 (the §6/§8 PENDING item) still needs this guarded Docker confirmation before it can be reclaimed; its prediction patch is staged at `~/swe/mpl_preds_23913.json`.

## 10. Scorer telemetry: capture the result-count summary, not just the last line
`test_runner.run_tests` originally stored `tail = (stdout+stderr).splitlines()[-1][:160]`
and `swe_agent_v2.score()` persisted it as `score_tail`. Problem for false-negative
triage: the single last line of stdout+stderr is frequently NOT the pytest/unittest
result summary. When a run has trailing stderr (a DeprecationWarning traceback), a
crash, or ends on a `rootdir:`/node-path line (seen on pytest-dev instances and on
matplotlib "found no collectors" node-id failures), the `N passed/failed/error/skipped`
counts get buried mid-output and never reach `score_tail`. That is exactly the signal
FN triage needs (real miss vs env/collection FN).

Fix (telemetry-only, cannot change ok/passed/exit -> cannot change any score):
`_build_tail(out)` now scans from the end for the runner result-count summary line
(`_SUMMARY_RE`: N passed/failed/error/skipped/warnings, `Ran N tests`, `OK`,
`FAILED`), strips ANSI, and returns `"<summary>  ||  <last line>"` when the summary
is not already the last line -- otherwise the last line unchanged. This PRESERVES
diagnostic substrings (e.g. "found no collectors") so existing signature matching
still works, while always surfacing the counts. Unit-tested on: clean pytest (byte-
for-byte unchanged), stderr-buried summary, collection-error-only, collection-error
+summary, django OK/FAILED, ANSI-coded lines, passed+warnings, empty. General lesson:
harness observability strings must target the *semantic* result line, not a positional
last line, because subprocess stream interleaving is not stable. Inert for a running
process (module imported once); active on relaunch and for all future runs.

## 11. Full scorer output persisted for offline false-negative triage

The final FAIL_TO_PASS scorer output used to be thrown away: run_tests kept only
score_tail (one line) and result["stdout"] (last 1500 chars, not persisted to
results). Twice this loop was blocked triaging a miss because the tail was
uninformative (e.g. bare "rootdir: /home/bard/swe/work/<id>" for several
pytest-dev misses that actually hit ModuleNotFoundError/collection-error) —
distinguishing an env-collection FALSE NEGATIVE from a real miss then required a
full re-run (Docker, cold-cache, deadlock-prone).

Fix (commit e221321): run_tests takes an optional log_path; score() passes
~/swe/score_logs/<instance_id>.log. _write_score_log writes the FULL scorer
stdout+stderr with a header (cmd / exit / ok). Properties:
  - TELEMETRY-ONLY: write-only, returns None, wrapped in try/except so a fs
    error cannot disturb scoring. ok/passed/exit/tail computed exactly as before
    (verified byte-identical with vs without log_path).
  - NO ANSWER LEAKAGE: it is the scorer's own test output written to disk for the
    operator; never feeds the model.
  - Inert for an already-running runner (module imported once); active on
    relaunch / all future runs. No relaunch was done for a telemetry change.

Triage workflow going forward: for any post-relaunch miss, read
~/swe/score_logs/<id>.log to see the real failure. Signatures that mark a likely
FALSE NEGATIVE (env/collection, not a wrong patch): "found no collectors",
"errors during collection", "ModuleNotFoundError"/"ImportError" at collection,
"collected 0 items". Confirm via Docker (docker_eval_guard.sh) before flipping in
swe_false_negatives.json. A wrong-patch real miss instead shows the target test
"FAILED"/"X failed" after a clean collection.

## 12. Sphinx (and any turn-capped repo) phase-1 bootstrap-budget deaths — surface the gate's reason

Symptom in results_full300.json: `env_ok:false, env_kind:null, python:null, installs:[],
phase1_reason:"budget"`. The trace tells the real story: venv created,
`install_repo_editable` OK, `run_sanity` OK — then 20-36 `run_smoke_test` calls returning
ok=false and the 50-turn phase-1 cap hit WITHOUT a successful `declare_env_ready`. Seen on
the whole current sphinx env-fail cluster (8721/8474/8282/7975; 8801 has a stale successful
trace from an earlier Jul-13 run, so its results env-fail is an older-run artifact).

Root cause (scaffold, not env): the env-ready gate (`_boot_gate`) auto-runs
`auto_verify_env` on declare, but `phase_run` discarded its specific result and returned
only a generic "verification gate not passed; run_sanity and run_smoke_test must both
return ok=true first". With no signal about what the harness check actually found, the
model re-declares or keeps guessing smoke tests (often re-picking FAIL_TO_PASS tests, which
are correctly refused) until the budget dies — on environments that are frequently fine.

Fix (commit ad5f1a4): `_boot_gate` captures `auto_verify_env`'s result via new
module-level `_auto_verify_reject_detail()` and stashes a short actionable hint on
`_boot_gate.reject_detail`; `phase_run` surfaces it as `payload["harness_check"]` on gate
rejection (missing test dep -> install it; uncollectable suite / no green test -> call
`run_smoke_test` WITH NO ARGUMENTS to let the harness auto-pick a stable test). STEERING
ONLY: the phase-2 fix gate is a plain lambda without that attribute, so its rejection
payload is byte-identical to pre-patch — no scoring path changes (unit-verified). No answer
leakage (auto_verify_env excludes the instance's FAIL_TO_PASS tests; only env diagnostics
surface). Inert for the already-running runner (module imported once); active on relaunch /
future runs — NOT relaunched for a steering change.

General lesson (also in engineering-patterns.json): a gate that auto-runs a check must feed
back the check's SPECIFIC diagnostic, never a generic rejection, or the agent thrashes
blind until budget death.

## 13. scikit-learn importorskip("pandas") skip false-negative family (telemetry: "N skipped in Xs")

SIGNATURE: a scikit-learn miss whose score_tail is "N skipped in 0.0Xs" (not "failed"/"error").
Many sklearn FAIL_TO_PASS tests begin with `pd = pytest.importorskip("pandas")` (also
matplotlib for plotting tests). If pandas is NOT installed in the scored env, pytest reports
the target tests as SKIPPED, never as pass/fail. A skipped F2P test is not a pass, so the
instance is scored UNRESOLVED even when the model's patch is correct -> FALSE NEGATIVE.
This is a different mechanism from the matplotlib warnings-as-errors "found no collectors"
family (a collection error) but the same outcome: the F2P tests never actually execute.

GENERAL FIX (already shipped, active): install_spec_extras() in swe_agent_v2.py, called in
run_one right after phase-1 env is ready (~line 514), installs the SWE-bench spec-declared
optional TEST deps from ~/swe/spec_extras.json (e.g. sklearn -> pandas<2.0.0, matplotlib<3.9.0),
version-matched, so importorskip-gated tests RUN instead of silently skipping. Env-layer only;
never touches the answer. This makes scoring MORE accurate in BOTH directions: a correct patch
now PASSES (was skip->miss); a wrong patch still FAILS. So it cannot manufacture false positives.

TRIAGE / RECLAIM: results scored BEFORE install_spec_extras existed (e.g. Jul-12/13 sklearn
traces) can carry stale "N skipped" misses. To check one: in its ~/swe/work/<id> venv (which
retains model_patch + test_patch), confirm pandas is present, then run its FAIL_TO_PASS with
-rs. If all F2P now PASS -> home-verified false negative (add to swe_false_negatives.json as
docker_confirmed=false, PENDING); if any FAIL -> real miss (patch wrong), record in
confirmed_real_misses so future cycles skip it. Docker eval remains authoritative before any
score is flipped. Verified example: scikit-learn-25570 (3 pandas_output F2P tests: skip->3
passed) = PENDING FN; 25500 (wrong file; y_pred DataFrame not ndarray) and 25638 (ValueError
mix of unknown/binary targets) FAIL with pandas present = real misses.


## 14. pytest `--no-header` breaks scoring on pytest<6 (auto-miss family; harness bug, not env)

`test_runner.run_tests` (THE single test path — used by `score()`, the model's
fix-verify tools in swe_fix_tools.py, and phase-1 sanity/smoke in
repo_bootstrap_tools.py) appended `--no-header` starting Jul-10 (commit b4272ea).
`--no-header` is a **pytest>=6.0** flag. On pytest 4.x/5.x it is an
`error: unrecognized arguments: --no-header` **usage error -> exit 4, ZERO tests
run**, so the instance is auto-scored UNRESOLVED no matter how good the patch is.

**Signature in results:** `score_tail` is a bare header line
`"  rootdir: /home/bard/swe/work/<id>"` (no `N passed`/`N failed`), and the repo
is one pinning old pytest. In SWE-bench Lite full-300 the ONLY pytest<6
instances are in `pytest-dev/pytest` (other repos pin pytest>=6).

**Triage of all 9 completed pytest<6 misses** (home-verified by running the exact
FAIL_TO_PASS in the retained work-dir, model+test patch applied, WITHOUT the bad flag):
- FALSE NEGATIVE (reclaimable): `pytest-dev__pytest-5227` -> 3 passed (now ok=True via shipped path). Recorded docker_confirmed=false (PENDING) in swe_false_negatives.json.
- REAL misses (patch genuinely wrong; do NOT re-investigate): 5103, 5221, 5413, 5495, 5692, 6116, 7168, 7220 — all still FAIL their F2P once the tests actually run.

**Fix (commit 86bdca4):** version-gate the flag via cached `_pytest_major(py, repo_dir, env)`;
`hdr = '--no-header' if _pytest_major(...) >= 6 else ''`. **Cache key = os.path.join(repo_dir, py)**
— the relative py path `.venv/bin/python` is identical across instances, so keying on it alone
would return a stale major in the long-running benchmark process and could RE-ADD the flag to a
later old-pytest instance (piece-check caught exactly this: 8906 wrongly cached as major 4).

**General lesson -> engineering-patterns.json:** version-gate test-runner flags to the era of the
code under test; a flag valid for your newest instances can be fatal for the oldest.

The running benchmark imported the buggy module once (change inert until relaunch), but there are
**ZERO remaining pytest<6 instances in the to-do set** (all 17 pytest-dev already scored), so
**no relaunch is warranted** — the fix protects future/fresh runs and the reclaim of 5227 is handled
by the end-of-run manifest path.

## 15. Sphinx <5 test collection dies on a too-new support ecosystem (env FN + solve-blinding)

**Signature in results:** a `sphinx-doc/sphinx` instance with `env_ok:true` but
`score_tail` = `"N warnings, 1 error in 0.Xs"` (a *collection* error, not `N passed/failed`),
usually with `phase2_reason: budget` (a few `declared`).

**Root cause (env, general — same family as the matplotlib/pyparsing NO_COLLECTORS bug):**
a fresh Sphinx <5 env pulls the LATEST support ecosystem, which hard-requires a newer Sphinx and
aborts collection when Sphinx loads its default extensions at app setup:
- `sphinxcontrib-applehelp/devhelp/htmlhelp/qthelp/serializinghtml` 2.x **and even 1.0.8** call
  `require_sphinx("5.0")` -> `VersionRequirementError: 5.0`.
- `alabaster` 0.7.16 requires Sphinx >=3.4 -> `VersionRequirementError: 3.4` on Sphinx 3.1/3.3.
- `roman` (imported by `sphinx.builders.latex`) is simply MISSING ->
  `ExtensionError: Could not import extension sphinx.builders.latex (No module named roman)`.
This breaks the authoritative scorer (F2P cannot collect -> auto-miss) AND blinds the model in
phase 1/2 (it can never get a green sphinx test) -> the DEEPER cause behind the sec.12
sphinx bootstrap-budget deaths (steering alone could not fix an env the model cannot make green).

**Fix (data-only, no harness code change):** `~/swe/spec_extras.json` entries for the 13 Sphinx <5
instances (7686, 7738, 7975, 8273, 8282, 8435, 8474, 8506, 8595, 8627, 8713, 8721, 8801):
`roman`, `alabaster==0.7.12`, `sphinxcontrib-applehelp==1.0.2`, `sphinxcontrib-devhelp==1.0.2`,
`sphinxcontrib-htmlhelp==1.0.3`, `sphinxcontrib-jsmath==1.0.1`, `sphinxcontrib-qthelp==1.0.3`,
`sphinxcontrib-serializinghtml==1.1.5`. `install_spec_extras` re-reads this file live per instance
and runs pre-phase-1, so future/fresh runs get a working env for both the model and the scorer.
Sphinx >=5 (10325, 10451, 11445) is CORRECT with the 2.x line and is deliberately NOT pinned.

**Verification (this cycle):** end-to-end through the shipped `install_spec_extras` on untouched
`sphinx-8627` (Sphinx 3.5): before = `ModuleNotFoundError: roman`; after = `import-ok`
(`sphinx.builders.latex` + `roman` + `Sphinx` all import). Collection restored on all 5
collection-error instances (7686/8273/8435/8506/8595); once collecting they RUN and FAIL their F2P
-> **all REAL misses, no false-negative reclaim** (the models solved blind and produced wrong
patches). Already-resolved 7738/8713 stay `1 passed` with the pins -> no regression.

**No relaunch / no reclaim this cycle:** the full-300 run is resumable and skips-done, and all 16
sphinx instances are already scored, so this fix yields nothing for the current run. Realize the
gain via a **targeted re-run of the 13 Sphinx <5 instances** (fresh env, so the model can actually
verify) or on the next fresh full run. General lesson promoted to engineering-patterns.json
(collection/version errors = env, not code) and to knowledge/sphinx-doc__sphinx.md (the pin recipe).


## 16. matplotlib NO_COLLECTORS — the SECOND cause: setuptools_scm/vcs-versioning entry-point hijack

Section 8/the WARN_AS_ERROR_DEP_PINS fix pinned `pyparsing<3.1` to stop matplotlib's
`filterwarnings=error` from turning a `PyparsingDeprecationWarning` into a fatal collection
error. But some matplotlib **dev builds** (e.g. 3.6.0.dev / 23314, 23476) compute
`matplotlib.__version__` via `setuptools_scm.get_version()` **at import**
(`lib/matplotlib/__init__.py::_get_version`). With `setuptools-scm` 8+ installed, the
sibling package **`vcs-versioning`** registers the `release-branch-semver` version-scheme
entry point as a *deprecation shim* (`vcs_versioning/_version_schemes/_standard.py`:
"Version scheme 'release-branch-semver' has been renamed…"). matplotlib's `setup.py` still
requests the old name, so the shim raises a `DeprecationWarning` -> fatal -> **collection
dies even after pyparsing is pinned**. Same `found no collectors` symptom, different dep.

Key subtlety: downgrading `setuptools_scm` to 7.x alone does **not** help while
`vcs-versioning` is still installed — its entry point is still resolved. You must
**uninstall** `vcs-versioning` (and use the self-contained 7.x line). It is also
**version-dependent**: instances whose build reads a cached `_version.py` (e.g. 23987 with
pyparsing already pinned) collect fine, so the extra pins are a harmless no-op there.

Fix (commit d5c3348): `WARN_AS_ERROR_DEP_PINS["matplotlib/matplotlib"] =
["pyparsing<3.1", "setuptools_scm<8", "-vcs_versioning"]`, and `pin_warn_as_error_deps`
now parses `-pkg` specs as `pip uninstall -y pkg` (installs first, then removals). Verified
end-to-end through the shipped function on a reset-to-broken 23314 venv: F2P
`test_invisible_axes[png]` flips collection-error -> **1 passed**; no-op for django. Inert
for the running process (module imported once) — active on relaunch/fresh runs, where it
makes the LIVE scorer correct for this family without needing the reclaim manifest.

Triage of the two newly-surfaced NO_COLLECTORS instances (both `found no collectors`):
- **23314** = home-verified FALSE NEGATIVE (model self-verified; F2P passes once env cleared)
  -> added to swe_false_negatives.json as `docker_confirmed=false` (PENDING, per the
  23913/25570/5227 precedent; promote via Docker eval).
- **23476** = REAL MISS (did not self-verify; once env cleared, F2P fails with a genuine
  `AssertionError` on `Figure.dpi`) -> recorded in `confirmed_real_misses`.


## 17. Scorer scope: PASS_TO_PASS is NOT enforced at home (false-positive analysis)

`swe_agent_v2.score()` runs ONLY `inst['FAIL_TO_PASS']` (via `test_runner.run_tests`); `PASS_TO_PASS` is used for phase-1 smoke-test hints but is never run at scoring time. Official SWE-bench requires BOTH: every F2P transitions to pass AND every P2P stays green (no regression). So the home `resolved` flag has a THEORETICAL false-positive exposure: a patch that fixes F2P but breaks a previously-passing test scores resolved=True at home but UNRESOLVED under the authoritative Docker scorer.

DETECTION (tools: `p2p_audit.py` fast pre-filter, `p2p_fp_audit.py` rigorous confirm). Do NOT trust a bare P2P re-run: many P2P tests fail in the home env even at base commit (home/gold env discrepancy). The valid discriminator is a TWO-STATE CAUSATION test per instance in its surviving work-dir (keeping .venv): A) base+test_patch (no model) run P2P; B) base+test_patch+model_patch run P2P. Genuine regression (suspected FP) = A green AND B fails. A fails = env discrepancy (NOT a scorer bug). A green AND B green = true positive. Confirm any genuine regression with the Docker eval before reclassifying; never edit results in place (clobber-safe; runner rewrites results_full300.json).

EMPIRICAL RESULT (2026-07-15, 140 done / 58 resolved): audited all 56 resolved instances with surviving work-dirs -> 47 fully-green P2P, 9 P2P-fail, 2 no-workdir. Two-state causation test on the 9 -> ALL env discrepancy (8) or corrupted/missing venv (1, sympy-24152); ZERO genuine regressions. So the current resolved count is NOT inflated by the P2P skip, AND naively enforcing P2P at home would have created 8+ NEW false negatives. Conclusion: keep home scorer F2P-only; enforce P2P in the end-of-run Docker audit. Reports: ~/swe/p2p_audit_report.json, ~/swe/p2p_fp_audit_report.json.

## 18. psf/requests httpbin-network false-negative family (external-service tests in a network-isolated scorer)

Signature: psf/requests misses whose FAIL_TO_PASS are httpbin endpoint tests
(test_HTTP_200_OK_GET*, test_BASICAUTH_*, test_POSTBIN_*, test_unicode_multipart_post,
test_manual_redirect_*, test_HTTP_302_ALLOW_REDIRECT_GET, ...). score_tail is either
"N failed, M passed in <tens of seconds>" (DNS/connect timeouts to httpbin.org) or a fast
"N failed in 0.Xs" (connection refused).

Root cause: the requests test suite defines `HTTPBIN = os.environ.get('HTTPBIN_URL',
'http://httpbin.org/')` and helper `httpbin(*suffix)`. On this host httpbin.org is UNREACHABLE,
`pytest-httpbin` is NOT installed, and there is no conftest wiring a local server, so every
httpbin-dependent F2P raises ConnectionError -- independent of whether the model's patch is
correct. Same class as the matplotlib/sphinx/sklearn env-FN families (the test can't even run),
just over the network instead of at collection.

General fix (SHIPPED 2026-07-15, commit 9bab933, ensure_local_httpbin in swe_agent_v2.py): for psf/requests, install pytest-httpbin,
start its bundled local httpbin Flask app on 127.0.0.1 (threaded), and export HTTPBIN_URL to
that URL BEFORE pytest starts (the module reads HTTPBIN at import time, so a fixture is too late).
Tests that scheme-swap http->https on the SAME netloc (e.g. test_mixed_case_scheme_acceptable)
additionally need HTTPS on the same host -- use pytest-httpbin's dual http+https serving (or real
httpbin.org connectivity).

SHIPPED mechanism (general, reusable): the harness writes a repo-root conftest.py whose MODULE
BODY (not a fixture) starts the bundled local httpbin and sets HTTPBIN_URL. pytest imports
conftest.py BEFORE it collects/imports the test modules, so this is the earliest hook that still
beats the test's import-time os.environ.get('HTTPBIN_URL') read -- a session-scoped fixture runs
too late. The server thread is daemon (+atexit stop) so a lingering serve_forever can never hang
pytest at exit. The conftest is UNTRACKED, so it never enters the model's git-diff patch and
carries no instance data. ensure_local_httpbin() runs in score() (AFTER the test patch, so a
suite-provided conftest is never clobbered) and in run_one() after install_spec_extras (phase-2
self-verify). Verified through the shipped path: requests-2148 F2P 3-fail -> 10 passed; real-miss
requests-2317 stays 8 failed (no false positive); no-op for non-httpbin repos. Transferable lesson:
when a suite reads config from the ENVIRONMENT at import time, inject via a repo-root conftest
module body, not a fixture.

Triage / discrimination method (leakage-safe -- no gold/test-patch content to the model):
in the retained work-dir (model+test patch already applied), `pip install pytest-httpbin`, run
`python -c "from httpbin import app; app.run(host='127.0.0.1', port=P, threaded=True)"` in the
background, `export HTTPBIN_URL=http://127.0.0.1:P/`, and re-run the exact F2P node ids.
env-FNs flip to all-pass; genuine misses still fail with real assertions/TypeErrors.

Findings (140-run, all 6 requests instances): requests-2148 = home-verified FALSE NEGATIVE
(10/10 F2P pass locally); requests-2674 = FALSE NEGATIVE (11/12 pass; the 12th is the
local-HTTPS-only artifact above); requests-2317 = REAL MISS (all 8 F2P fail with TypeError from
the model's patched requests/compat.py, even with httpbin up). 2148/2674 recorded in
swe_false_negatives.json as docker_confirmed=false (PENDING Docker confirm); 2317 in
confirmed_real_misses. Resolved requests instances (3362/1963/863) have non-httpbin F2P and are
unaffected.
