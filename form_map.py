"""Form-selection map: which output FORMS can this function produce, and what
CONDITION selects each? (Mikey: "why don't we have a map for that mechanism?")

The mechanism, stated generally: a function that chooses among several output
representations based on a predicate over its input. Printers choosing radical
vs power form. Formatters choosing long vs short layout. Mode branches choosing
host vs subdomain. The bug class is always the same -- THE PREDICATE ADMITS AN
INPUT IT SHOULD NOT (or misses one it should), so a valid input gets rendered
in the wrong form. The output then looks plausible and is wrong, and reading
the format strings tells you nothing: the fault is in the `if`, not the
template.

We had the RULE twice over (layout KB: "a different form means the printer
chose another branch upstream"; pattern 40: modes must preserve invariants)
and no MAP. This is the map: for a function, every branch condition paired
with the form it selects, extracted from AST. Symbolic, no model, general.

    FORM MAP for _print_Pow:
      IF   expr.exp.is_Rational and expr.exp.q != 1 ...  -> root form (\\sqrt)
      ELSE                                               -> power form (^{})

With that in front of it, "pi**(1/E) renders as a radical" stops being a
mystery: the input's path through the conditions is checkable one predicate at
a time with check().
"""
import ast, os, sys

FORMY = ("return", "format string")


def _src(node, cap=100):
    try:
        s = ast.unparse(node)
    except Exception:
        return "?"
    return " ".join(s.split())[:cap]


def _form_of(body):
    """A short description of what a branch produces: its return expression or
    the format-ish strings it builds."""
    for n in body:
        for w in ast.walk(n):
            if isinstance(w, ast.Return) and w.value is not None:
                return "returns " + _src(w.value, 80)
    # no return: look for string constants being assembled
    frags = []
    for n in body:
        for w in ast.walk(n):
            if isinstance(w, ast.Constant) and isinstance(w.value, str) and len(w.value) > 2:
                frags.append(w.value.replace("\n", "\\n")[:40])
                if len(frags) >= 2:
                    break
    return ("builds " + " ... ".join(repr(f) for f in frags[:2])) if frags else "(no visible form)"


def form_map(path, func_name, limit=10):
    """Every branch condition in `func_name` paired with the form it selects."""
    try:
        tree = ast.parse(open(path, encoding="utf-8", errors="ignore").read())
    except Exception as e:
        return {"error": str(e)[:80]}
    leaf = (func_name or "").split(".")[-1]
    fn = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == leaf:
            fn = node
            break
    if fn is None:
        return {"error": "function %s not found" % leaf}

    branches = []

    def walk_ifs(node, depth=0):
        for ch in getattr(node, "body", []) if not isinstance(node, ast.If) else []:
            pass
        for w in (node.body if hasattr(node, "body") else []):
            if isinstance(w, ast.If):
                branches.append({"condition": _src(w.test),
                                 "selects": _form_of(w.body), "depth": depth})
                walk_ifs_body(w.body, depth + 1)
                # elif chains
                orelse = w.orelse
                while len(orelse) == 1 and isinstance(orelse[0], ast.If):
                    e = orelse[0]
                    branches.append({"condition": _src(e.test),
                                     "selects": _form_of(e.body), "depth": depth})
                    walk_ifs_body(e.body, depth + 1)
                    orelse = e.orelse
                if orelse:
                    branches.append({"condition": "(otherwise)",
                                     "selects": _form_of(orelse), "depth": depth})
                    walk_ifs_body(orelse, depth + 1)
            else:
                walk_ifs(w, depth)

    def walk_ifs_body(body, depth):
        for w in body:
            if isinstance(w, ast.If):
                branches.append({"condition": _src(w.test),
                                 "selects": _form_of(w.body), "depth": depth})
                walk_ifs_body(w.body, depth + 1)
                orelse = w.orelse
                while len(orelse) == 1 and isinstance(orelse[0], ast.If):
                    e = orelse[0]
                    branches.append({"condition": _src(e.test),
                                     "selects": _form_of(e.body), "depth": depth})
                    walk_ifs_body(e.body, depth + 1)
                    orelse = e.orelse
                if orelse:
                    walk_ifs_body(orelse, depth + 1)
            elif hasattr(w, "body"):
                walk_ifs_body(w.body, depth)

    walk_ifs_body(fn.body, 0)
    if not branches:
        return None
    note = ("This function SELECTS among output forms. If the output is in the "
            "WRONG FORM for some input, the fault is in one of these CONDITIONS "
            "-- a predicate admitting an input it should not, or missing one it "
            "should -- not in the form templates. Trace YOUR input through the "
            "conditions with check(): evaluate each predicate on the failing "
            "input and find the branch it takes.")
    return {"function": leaf, "branches": branches[:limit], "note": note}


if __name__ == "__main__":
    r = form_map(sys.argv[1], sys.argv[2])
    if not r:
        print("(no branches)")
    elif r.get("error"):
        print("error:", r["error"])
    else:
        print("FORM MAP for %s:" % r["function"])
        for b in r["branches"]:
            print("  %sIF %s" % ("  " * b["depth"], b["condition"][:88]))
            print("  %s   -> %s" % ("  " * b["depth"], b["selects"][:80]))
