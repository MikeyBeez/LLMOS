# Knowledge: astropy/astropy

Accumulated notes for working on `astropy/astropy`. Loaded whenever this repo is the 
target. Append anything learned. Keep every entry GENERAL to the package —
never an instance-specific fix (that would leak the answer).

_Seeded from 2 resolved run(s)._

## Environment (what has worked)

- Python seen working: 3.10
- Backend: uv
- Common installs: setuptools, numpy, Cython, extension-helpers, setuptools-scm, wheel, packaging

### Build answers found via web search

- **Q:** astropy astropy/wcs/tests/test_wcs.py::test_fixes pytest collection
  **A (snippet):** {"query": "astropy astropy/wcs/tests/test_wcs.py::test_fixes pytest collection", "hits": [{"title": "astropy/astropy/wcs/tests/test_wcs.py at main - GitHub", "snippet": "2612 lines (2143 loc) \u00b7 8
- **Q:** astropy repo .hypothesis directory pytest collection fix norecursedirs
  **A (snippet):** {"query": "astropy repo .hypothesis directory pytest collection fix norecursedirs", "hits": [{"title": "Pytest: &quot;found no collectors&quot;, cannot test my changes #18062", "snippet": "This only o
- **Q:** astropy repo ".hypothesis" pytest collection delete fix
  **A (snippet):** {"query": "astropy repo \".hypothesis\" pytest collection delete fix", "hits": [], "answer": "", "goal_stack": "no active subgoals"}
- **Q:** pytest ".hypothesis" directory "found no collectors" skip
  **A (snippet):** {"query": "pytest \".hypothesis\" directory \"found no collectors\" skip", "hits": [{"title": "Pytest: &quot;found no collectors&quot;, cannot test my changes #18062", "snippet": "An empty .hypothesis
- **Q:** delete .hypothesis directory pytest astropy fix collection
  **A (snippet):** {"query": "delete .hypothesis directory pytest astropy fix collection", "hits": [{"title": "Pytest: &quot;found no collectors&quot;, cannot test my changes #18062", "snippet": "An empty .hypothesis/ d

## Fix landscape (orientation, NOT answers)

Resolved fixes in this package have touched:

- `astropy/nddata/mixins/ndarithmetic.py`
- `astropy/modeling/separable.py`

## Gotchas

- Treats warnings as ERRORS. A benign DeprecationWarning from a *dependency* becomes a fatal collection error and masks any fix. Real case: a too-new matplotlib pulled a pyparsing whose `oneOf` is deprecated; astropy's conftest imports matplotlib when present -> the whole session died at collection. Don't install deps astropy doesn't need; version-match the ones it does.
- `.hypothesis/` dirs left by earlier test runs break collection ('found no collectors'). Remove them before collecting/scoring (the scorer already `rmtree`s `.hypothesis`).
- Canonical Python: 3.9 for the 5.x era; older eras (1.x) need 3.6 -> build the interpreter with micromamba (uv can't).
