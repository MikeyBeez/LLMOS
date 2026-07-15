# Knowledge: psf/requests

Accumulated notes for working on `psf/requests`. Loaded whenever this repo is the 
target. Append anything learned. Keep every entry GENERAL to the package —
never an instance-specific fix (that would leak the answer).

_Seeded from 2 resolved run(s)._

## Environment (what has worked)

- Python seen working: 3.9
- Backend: uv
- Common installs: setuptools, pytest

### Build answers found via web search

- none needed so far.

## Fix landscape (orientation, NOT answers)

Resolved fixes in this package have touched:

- `requests/utils.py`
- `requests/sessions.py`

## Gotchas

- Tests read `HTTPBIN = os.environ.get('HTTPBIN_URL', 'http://httpbin.org/')` at IMPORT. Offline,
  every httpbin endpoint test (test_HTTP_200_OK_*, test_BASICAUTH_*, test_POSTBIN_*, ...) raises
  ConnectionError and a correct patch scores as a miss. The harness auto-wires a local server
  (pytest-httpbin + a repo-root conftest that sets HTTPBIN_URL) via ensure_local_httpbin() -- nothing
  to do manually. Do NOT "fix" a requests miss by editing network code before confirming the F2P
  actually reach a live server; the failure is usually the environment, not the patch.
