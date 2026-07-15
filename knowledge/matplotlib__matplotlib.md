# Knowledge: matplotlib/matplotlib

Accumulated notes for working on `matplotlib/matplotlib`. Loaded whenever this repo is the 
target. Append anything learned. Keep every entry GENERAL to the package —
never an instance-specific fix (that would leak the answer).

_Seeded from 3 resolved run(s)._

## Environment (what has worked)

- Python seen working: 3.11, 3.9
- Backend: uv
- Common installs: pybind11, pyparsing, numpy, oldest-supported-numpy, certifi, meson-python, setuptools, pytest, setuptools_scm

### Build answers found via web search

- **Q:** matplotlib test_pickle.py test functions defined
  **A (snippet):** {"query": "matplotlib test_pickle.py test functions defined", "hits": [{"title": "Testing \u2014 Matplotlib 3.12.0.dev340+gdcac4b84e documentation", "snippet": "To run a single test from the command l

## Fix landscape (orientation, NOT answers)

Resolved fixes in this package have touched:

- `lib/mpl_toolkits/axes_grid1/axes_grid.py`
- `lib/matplotlib/axis.py`
- `lib/matplotlib/offsetbox.py`

## Gotchas

- _(add as you hit them)_


### Gotcha: NO_COLLECTORS under warnings-as-errors has THREE dep causes
1. `pyparsing>=3.1` -> `PyparsingDeprecationWarning` on matplotlib's camelCase API -> fatal.
2. Dev builds call `setuptools_scm.get_version()` at import; `setuptools-scm` 8+ + the
   `vcs-versioning` fork make the old `release-branch-semver` scheme name a fatal
   `DeprecationWarning`. Downgrading setuptools-scm is not enough — **uninstall
   vcs-versioning**. The harness clears both via `WARN_AS_ERROR_DEP_PINS`
   (`pyparsing<3.1`, `setuptools_scm<8`, `-vcs_versioning`). Effect is version-dependent
   (no-op where a cached `_version.py` is used).

3. The resolver pulls the **latest pytest** (9.x / >=8.4), which raises
   `PytestRemovedIn10Warning` for an unrelated same-file test that passes a
   generator to `@parametrize` (e.g. `test_rcparams.py::test_validator_valid`)
   -> fatal under `filterwarnings=error` -> the whole module fails to collect.
   Pin the **runner to its era**: `pytest<8` (7.x still supports `--no-header`).
   `WARN_AS_ERROR_DEP_PINS` now carries all three: `pyparsing<3.1`,
   `setuptools_scm<8`, `-vcs_versioning`, `pytest<8`. Restoring collection can
   REVEAL a real miss (masked verdict may still FAIL); it fixes scorer correctness,
   not the reclaim count. Verified no-regression: resolved 26011 stays green under 7.x.
