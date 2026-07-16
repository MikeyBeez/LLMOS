# Knowledge: pytest-dev/pytest

Accumulated notes for working on `pytest-dev/pytest`. Loaded whenever this repo is the 
target. Append anything learned. Keep every entry GENERAL to the package —
never an instance-specific fix (that would leak the answer).

_Seeded from 1 resolved run(s)._

## Environment (what has worked)

- Python seen working: 3.11
- Backend: uv
- Common installs: attrs, xmlschema, hypothesis


## Fix landscape (orientation, NOT answers)

Resolved fixes in this package have touched:

- `src/_pytest/assertion/rewrite.py`

## Gotchas

- Import-machinery bugs (importlib import-mode) are PYTHON-VERSION SENSITIVE. Reproduce on the canonical interpreter (3.9 for the 8.x era) or the bug won't behave as reported.
- Double-import under importlib mode -> two distinct module objects (`sys.modules[name] is mod` -> False), so state set on one copy is invisible on the other. The fix is a `sys.modules` cache-check in `import_path()`. COMMON MISS: patching a neighbor in `src/_pytest/pathlib.py` (e.g. `insert_missing_modules`) instead of `import_path` — right file, wrong function.
