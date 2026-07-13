# Knowledge: scikit-learn/scikit-learn

Accumulated notes for working on `scikit-learn/scikit-learn`. Loaded whenever this
repo is the target. Keep entries GENERAL to the package (no instance-specific fix).

_Seeded from investigation (no resolved trace yet — all 4 in the 7/12 run missed)._

## Environment (what has worked)

- Python: **3.9** for the v1.x era; **3.6** for v0.20-0.22 (build the 3.6 interpreter with micromamba — uv can't).
- Backend: uv for 3.8+; micromamba for 3.6.
- pandas and matplotlib are **optional TEST deps**, NOT core — `pip install scikit-learn` does not pull them, so pandas-gated tests silently SKIP. Install them explicitly (spec: `pandas<2.0.0`, `matplotlib<3.9.0`).
- numpy floor for v1.3 is `1.19.2`, which has **no cp39 wheel** (cp39 numpy wheels start at 1.19.3) -> use `1.19.5`, or build from source with `--no-build-isolation` + cython.

## Fix landscape (orientation, NOT answers)

- v1.3 has a recurring bug cluster around the new pandas-output feature (`set_output` / `transform_output="pandas"`) and pandas nullable dtypes — these fixes live in the set_output / type-handling paths.

## Gotchas

- `pytest.importorskip("pandas")` SILENTLY SKIPS when pandas is absent; a naive scorer counts a skip as a miss with no error. If graded tests are 'all skipped', suspect a missing optional dep, not a bad fix.
