"""Composition map: does this layout code have the MEANS to place a block?

Found by solving sympy-23191 by hand. The layout map I built showed the symptom
(basis vector on the wrong row) and not the cause. The cause was static and
visible without running anything:

    arg_str = self._print(v).parens()[0]        # block -> STRING, block discarded
    o1.append(arg_str + ' ' + k._pretty_form)   # string concatenation
    ...
    tempstr = tempstr.replace(vectstrs[i], '')  # remove it again
    tempstr.replace(RIGHT_PAREN_GLYPH, GLYPH + ' ' + vectstrs[i])  # re-insert by glyph

Code that assembles 2D output by concatenating and searching STRINGS cannot
preserve baselines -- there is no baseline in a string. The glyph it searches
for may belong to a nested sub-expression, which is exactly how a unit vector
ends up mid-expression. No amount of reasoning about the symptom reveals this;
one look at the composition method does.

So the map answers: which of these two things is this function doing?

  BLOCK COMPOSITION  .right() .left() .above() .below() .parens() prettyForm(...)
                     -> baselines are handled for you
  STRING SURGERY     + on rendered text, .replace(), .split(chr(10)), indexing
                     into lines, searching for glyphs
                     -> baselines are lost; multi-line arguments WILL misplace

General beyond sympy: any function that builds multi-line output by string
concatenation has thrown away the alignment information it needs.

Purely static AST -- no model, no execution.
"""
import ast, os, sys

COMPOSE = {"right", "left", "above", "below", "parens", "next", "stack",
           "leftslash", "root", "func_name"}
SURGERY = {"replace", "split", "splitlines", "join", "strip", "lstrip",
           "rstrip", "find", "index", "startswith", "endswith"}


def compose_map(path, func_name):
    """Classify how `func_name` in `path` builds its output."""
    try:
        tree = ast.parse(open(path, encoding="utf-8", errors="ignore").read())
    except Exception as e:
        return {"error": str(e)[:80]}
    fn = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
            fn = node
            break
    if fn is None:
        return {"error": "function %s not found in %s" % (func_name, path)}

    compose_calls, surgery_calls, str_concat, line_index = [], [], 0, 0
    for n in ast.walk(fn):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
            a = n.func.attr
            if a in COMPOSE:
                compose_calls.append((a, getattr(n, "lineno", 0)))
            elif a in SURGERY:
                surgery_calls.append((a, getattr(n, "lineno", 0)))
        # string concatenation with +
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Add):
            for side in (n.left, n.right):
                if isinstance(side, ast.Constant) and isinstance(side.value, str):
                    str_concat += 1
                    break
        # indexing into a list of lines: o1[i] = ...
        if isinstance(n, ast.Subscript) and isinstance(n.ctx, ast.Store):
            line_index += 1

    verdict = ("BLOCK COMPOSITION" if compose_calls and not surgery_calls
               else "STRING SURGERY" if (surgery_calls or str_concat > 2)
               else "unclear")
    note = ""
    if verdict == "STRING SURGERY":
        note = ("This function assembles 2D output by manipulating STRINGS. A "
                "string has no baseline, so any multi-line argument will be "
                "placed by character position rather than by alignment -- and a "
                "glyph it searches for may belong to a NESTED sub-expression "
                "rather than the block being placed. If the reported symptom is "
                "'something appears in the wrong place', this is the mechanism. "
                "Compose blocks instead (parens/right/left/above), which align "
                "on baselines by construction.")
    return {"function": func_name, "verdict": verdict,
            "block_ops": sorted({a for a, _ in compose_calls}),
            "string_ops": sorted({a for a, _ in surgery_calls}),
            "literal_concats": str_concat, "line_assignments": line_index,
            "note": note}


if __name__ == "__main__":
    p, f = sys.argv[1], sys.argv[2]
    r = compose_map(p, f)
    for k, v in r.items():
        if v not in ("", [], 0):
            print("%-18s %s" % (k + ":", v))
