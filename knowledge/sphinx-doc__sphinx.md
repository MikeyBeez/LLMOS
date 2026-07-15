# sphinx-doc/sphinx -- env/harness notes

## Bootstrap: uv standalone CPython lacks `_testcapi` (env_ok=False / budget losses)
- Several sphinx test modules import the CPython-internal `_testcapi` (e.g. `tests/test_util_inspect.py`), which the uv-provisioned standalone CPython build does NOT ship. Such modules fail COLLECTION with `ModuleNotFoundError: No module named "_testcapi"`. This is an interpreter-packaging artifact, NOT a broken env.
- If a model-driven bootstrap keeps choosing these as smoke tests it can exhaust BOOTSTRAP_BUDGET and the instance is discarded (`env_ok=False, phase1_reason="budget"`). 5 instances lost this way in full300 (8801/8721/8474/8282/7975).
- Pick a smoke/verify test that does NOT touch `_testcapi`, e.g. `tests/test_util_typing.py` (19 passed in a healthy sphinx 3.5.0 / py3.9 venv).
- Harness safeguard shipped (commit b5b197c): a last-resort `auto_verify_env` runs when the model exhausts the budget with sanity_ok but not smoke_ok, so a healthy sphinx env is no longer discarded just because the model mis-picked smoke tests.
- setup.cfg `[tool:pytest] filterwarnings = all` (show, not error) -> sphinx here is NOT a warnings-as-errors suite; do not pin deps for a warnings-as-errors reason.
