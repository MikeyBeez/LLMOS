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

## EXTERNAL STANDARDS this package implements

sympy emits LaTeX. Correct LaTeX output is governed by conventions documented by
the AMS, NOT by anything in this repository -- so you cannot recover them by
reading neighbouring code, and the neighbours may not follow them. Where a
typesetting rule and local habit disagree, the typesetting rule is why the
output is judged wrong.

Sources: amsmath User's Guide (ams.org/arc/tex/amsmath/amsldoc.pdf);
Short Math Guide for LaTeX (Downes, AMS).

### Delimiter sizing -- the rule that most often bites

`\left` and `\right` size themselves to their content. AMS states plainly that
this "usually turn[s] out larger than necessary", and the effect COMPOUNDS when
nested: a `\left\langle ... \right\rangle` placed inside a `\left( ... \right)`
forces the outer parentheses to grow around already-enlarged inner delimiters.

So when you wrap an expression that ALREADY CONTAINS auto-sized delimiters:

  - do not leave the inner delimiters auto-sized. Use the plain form
    (`\langle`, `(`, `|`) or an explicit size (`\bigl`, `\Bigl`) inside the
    wrapper. Only the OUTERMOST pair should auto-size.
  - brace the wrapped group -- `{\left( ... \right)}^{n}` -- so an exponent
    binds to the whole group rather than to the closing delimiter.

This applies to any printer method that wraps its own output to attach an
exponent or an index, not to one particular function. A sibling method that
gets away with nesting usually contains no inner delimiters, so it never
exercises the case.

### How to check it

Render both forms and compare, rather than reasoning about the string:

    from sympy import latex
    print(latex(expr))        # plain
    print(latex(expr**3))     # wrapped -- inner delimiters must NOT be
                              # \left/\right, and the group must be braced

If the wrapped form contains a nested `\left` inside another `\left`, it is
wrong regardless of what the surrounding methods do.

### The exp= trap: the crash is the symptom, not the specification

Printer methods are called with `exp=<power>` when their object is raised to a
power. A method that lacks the parameter CRASHES with "unexpected keyword
argument" -- and that crash is only the SYMPTOM. Adding `exp=None` to the
signature makes the crash disappear, which makes a crash-based reproduction go
green, which proves nothing about the OUTPUT.

The real specification is the delimiter-sizing rule above, because attaching an
exponent IS the "wrap an already-delimited expression" case:

  - the wrapped form uses PLAIN delimiters inside the wrapper (the unwrapped
    form keeps its auto-sized ones -- the two forms differ on purpose)
  - the group is braced so the exponent binds to all of it

Do NOT copy the wrapper from neighbouring methods: most of them contain no
inner delimiters, so their pattern never exercised this rule and silently
violates it for methods that do.

Verify by RENDERING, never by absence-of-crash: print `latex(obj)` and
`latex(obj**3)` and check the wrapped form -- no `\left` nested inside another
`\left`, and the exponent attached to a braced group. A reproduction that only
demonstrates the TypeError will pass on a wrong fix.
