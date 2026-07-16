# Knowledge: astropy/astropy

Accumulated notes for working on `astropy/astropy`. Loaded whenever this repo is the 
target. Append anything learned. Keep every entry GENERAL to the package —
never an instance-specific fix (that would leak the answer).

_Seeded from 2 resolved run(s)._

## Environment (what has worked)

- Python seen working: 3.10
- Backend: uv
- Common installs: setuptools, numpy, Cython, extension-helpers, setuptools-scm, wheel, packaging


## Fix landscape (orientation, NOT answers)

Resolved fixes in this package have touched:

- `astropy/nddata/mixins/ndarithmetic.py`
- `astropy/modeling/separable.py`

## Gotchas

- Treats warnings as ERRORS. A benign DeprecationWarning from a *dependency* becomes a fatal collection error and masks any fix. Real case: a too-new matplotlib pulled a pyparsing whose `oneOf` is deprecated; astropy's conftest imports matplotlib when present -> the whole session died at collection. Don't install deps astropy doesn't need; version-match the ones it does.
- `.hypothesis/` dirs left by earlier test runs break collection ('found no collectors'). Remove them before collecting/scoring (the scorer already `rmtree`s `.hypothesis`).
- Canonical Python: 3.9 for the 5.x era; older eras (1.x) need 3.6 -> build the interpreter with micromamba (uv can't).
