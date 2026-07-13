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
- **pytest** — import-machinery bugs (e.g. importlib double-import) are *Python-version sensitive*; run on the canonical interpreter or the bug won't reproduce as intended. 8.x era → 3.9.
- *(add rows as you meet new repos)*

---

## 7. Meta

- SWE-bench publishes the exact environment per instance (`MAP_REPO_VERSION_TO_SPECS`: Python + packages + install command). Using it is **legitimate** (the environment is given; only the gold patch / test patch / FAIL_TO_PASS are off-limits). But the *better* system **derives** the env from the repo's own evidence so it generalizes to any repo — use the spec only as the answer key to grade the deriver against, never as its input.
