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

## PROTOCOL: pretty-printing is 2D layout -- look at the block, don't reason about it

Pretty-printing does not build a string. It builds a rectangular BLOCK of text
and then glues blocks together. Every printed piece carries four things, and a
rendering bug is almost always a wrong value in one of them:

- `picture`  the lines of the block, padded to equal width
- `baseline` WHICH ROW is the main line. When two blocks are joined side by
             side they are aligned on their baselines -- not on their tops.
- `binding`  a precedence class (ATOM, FUNC, DIV, POW, MUL, ADD, NEG, OPEN)
             that decides whether this piece gets parenthesised by its parent
- width / height

### How to look (do this BEFORE patching a printer)

You cannot see a layout bug by reading the printer source; the source looks
reasonable and the output is wrong. Render the thing and dump its block:

    from sympy.printing.pretty.pretty import PrettyPrinter
    pp = PrettyPrinter({"use_unicode": True})
    f = pp._print(expr)                      # expr = the object from the issue
    for i, line in enumerate(f.picture):
        print("%2d |%s|%s" % (i, line, "  <- baseline" if i == f.baseline else ""))
    print("w=%d h=%d baseline=%d binding=%s"
          % (f.width(), f.height(), f.baseline, f.binding))

Do this for the WHOLE expression and then for each sub-piece separately
(`pp._print(arg)` for arg in `expr.args`). The bug is usually visible the
moment two blocks are side by side.

### What the four values tell you

- A piece appearing at the WRONG HEIGHT -- floating above or sunk into the
  middle of its neighbours -- is a BASELINE disagreement between the pieces
  being joined, not a problem with either piece's own content.
- MISSING or SPURIOUS parentheses are a `binding` value, not a string bug.
  Check the class the piece reports against what its parent does with it.
- MISALIGNED columns (a fraction bar too short, an exponent over the wrong
  character) are width arithmetic: something measured the block before it was
  padded, or measured a line instead of the block.
- Expressions that render as an entirely DIFFERENT FORM than requested mean the
  printer chose another branch upstream -- the object was rewritten into a
  different type before printing. Compare `type(expr)` and `expr.args` against
  what you expected; the layout is a faithful rendering of the wrong tree.

### Verifying a printer fix

Comparing whole multi-line strings is a poor test: it passes trivially or fails
on trailing whitespace. Assert on the STRUCTURE instead --

  - the row index that holds the main expression (baseline)
  - block width and height
  - that a sub-piece's baseline matches its neighbour's after joining
  - that unrelated expressions still render exactly as before (a printer change
    that fixes one form and shifts another is a regression, and the repo's own
    printing tests are the given evidence for that)
