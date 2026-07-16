# Knowledge: django/django

Accumulated notes for working on `django/django`. Loaded whenever this repo is the 
target. Append anything learned. Keep every entry GENERAL to the package —
never an instance-specific fix (that would leak the answer).

_Seeded from 3 resolved run(s)._

## Environment (what has worked)

- Python seen working: 3.11
- Backend: uv
- Common installs: setuptools, wheel


## Fix landscape (orientation, NOT answers)

Resolved fixes in this package have touched:

- `django/db/migrations/serializer.py`
- `django/db/models/sql/query.py`
- `django/template/defaultfilters.py`

## Gotchas

- SCORING: run `tests/runtests.py` with `--parallel 1`. Its default parallel worker pool cannot pickle a failing test's traceback -> crashes the pool on teardown, drops the `OK`/`FAILED (failures=N)` summary (score_tail becomes a bare `ResourceWarning: unclosed running multiprocessing pool`), and can perturb the exit code so a passing patch scores as a miss. Serial matches the authoritative SWE-bench harness. (test_runner._django_supports_parallel gates the flag; ENV_KNOWLEDGE sec.19.)
