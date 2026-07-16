# Knowledge: sympy/sympy

Accumulated notes for working on `sympy/sympy`. Loaded whenever this repo is the 
target. Append anything learned. Keep every entry GENERAL to the package —
never an instance-specific fix (that would leak the answer).

_Seeded from 2 resolved run(s)._

## Environment (what has worked)

- Python seen working: 3.10
- Backend: uv
- Common installs: mpmath


## Fix landscape (orientation, NOT answers)

Resolved fixes in this package have touched:

- `sympy/physics/units/unitsystem.py`
- `sympy/physics/quantum/tensorproduct.py`

## Gotchas

- Return SymPy SINGLETONS from `_eval_*`, arithmetic and simplification methods:
  `S.One`, `S.Zero`, `S.NegativeOne` -- never bare python literals `1` / `0` / `-1`.
  Downstream code expects a `Basic` instance; a raw python int silently breaks
  `.args`, printing and further simplification. (Public sympy convention.)
- Guard the ZERO/degenerate branch of any rewrite. `sign(x)` style rewrites are
  correct as `x/Abs(x)` for every x EXCEPT 0, so the general form usually needs a
  `Piecewise((0, Eq(arg, 0)), (<general form>, True))`.
- Printer methods (`_print_*`) are called with an optional `exp=` when the object
  is raised to a power. If you add `exp` support, the exponent must wrap the WHOLE
  printed form, not be appended to an inner fragment.
