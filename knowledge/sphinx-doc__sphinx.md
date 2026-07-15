# Knowledge: sphinx-doc/sphinx

Accumulated notes for working on `sphinx-doc/sphinx`. Loaded whenever this repo is the 
target. Append anything learned. Keep every entry GENERAL to the package —
never an instance-specific fix (that would leak the answer).

_Seeded from 2 resolved run(s)._

## Environment (what has worked)

- Python seen working: 3.11
- Backend: uv
- Common installs: (repo editable only)

### Build answers found via web search

- **Q:** sphinx-doc sphinx tests directory structure test_*.py pytest collect
  **A (snippet):** {"query": "sphinx-doc sphinx tests directory structure test_*.py pytest collect", "hits": [{"title": "Testing API \u2014 Sphinx documentation", "snippet": "Utility functions and pytest fixtures for te

## Fix landscape (orientation, NOT answers)

Resolved fixes in this package have touched:

- `sphinx/util/rst.py`
- `sphinx/ext/autodoc/__init__.py`

## Gotchas

- **Env (general, package-level — NOT an answer):** Sphinx <5 must be tested with its support ecosystem pinned to that era, or default extensions abort test *collection* before any test runs. Install `roman` (the latex builder imports it) and pin `alabaster==0.7.12`, `sphinxcontrib-applehelp==1.0.2`, `sphinxcontrib-devhelp==1.0.2`, `sphinxcontrib-htmlhelp==1.0.3`, `sphinxcontrib-jsmath==1.0.1`, `sphinxcontrib-qthelp==1.0.3`, `sphinxcontrib-serializinghtml==1.1.5`. Symptoms if unpinned: `VersionRequirementError: 5.0` (sphinxcontrib 2.x, and even 1.0.8), `VersionRequirementError: 3.4` (alabaster 0.7.16 on Sphinx 3.1/3.3), or `Could not import extension sphinx.builders.latex (No module named roman)`. Sphinx >=5 works with the current 2.x sphinxcontrib line — do NOT pin it. Wired via ~/swe/spec_extras.json (per-instance, applied pre-phase-1 and at score).
