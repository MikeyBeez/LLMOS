"""Knowledge triggers tied to TOOLS (Mikey).

"The triggers have to be tied to the tools. If we use flask, that should
trigger certain things. If we use pytest, that should trigger certain things."

The 21171 failure in one line: the rule that solves it was injected at turn 0
and not applied at turn 40 when the patch was written. Knowledge delivered as a
wall of text at the start decays; knowledge delivered BY THE TOOL, at the
moment it applies, cannot be forgotten because it arrives with the action.

Each trigger is a symbolic match -- repo, tool, file path, enclosing function,
argument content -- plus the rule to say. Matching is pure string/regex work,
no model call. A matched rule rides back in the tool result, scoped to the
exact edit that made it relevant.

LEAKAGE: rules are the same general statements as the KB (audited); conditions
are STRUCTURAL (path patterns, name patterns, argument shapes), never
instance-specific. A trigger may never name an instance or quote expected
output.
"""
import re

# Each: dict(repo=substring|None, tool=name|None, path=regex|None,
#            enclosing=regex|None, args=regex|None, rule=text)
# First two matches fire, at most, per call.

TRIGGERS = [
    # ---- sympy: printing ------------------------------------------------
    dict(repo="sympy", tool="patch", path=r"printing/",
         enclosing=r"_print_", args=r"\bexp\b",
         rule=("EXP= TRAP: adding `exp=None` fixes the crash, which is the "
               "SYMPTOM. Attaching an exponent IS the wrap-an-already-"
               "delimited-expression case: plain delimiters INSIDE the wrapper "
               "(no \\left nested in \\left), and brace the group so the "
               "exponent binds to it. Do NOT copy the wrapper from sibling "
               "methods -- most contain no inner delimiters and never "
               "exercised this rule. Verify by RENDERING obj and obj**3, "
               "never by absence of crash.")),
    dict(repo="sympy", tool="patch", path=r"printing/pretty",
         enclosing=None, args=None,
         rule=("2D LAYOUT: pretty-printing builds BLOCKS (picture/baseline/"
               "binding), not strings. If the composition report above says "
               "STRING SURGERY, that is the mechanism: a string has no "
               "baseline, so multi-line pieces land by character position. "
               "Compose with parens/right/left/above instead. Dump the block "
               "via check() before and after -- a layout bug is invisible in "
               "source and obvious in the rendered block.")),
    # ---- pytest: import machinery ---------------------------------------
    dict(repo="pytest", tool="patch", path=r"(pathlib|import|conftest)",
         enclosing=None, args=None,
         rule=("ONE CANONICAL REGISTRY: don't import/construct the same thing "
               "twice unless there is a reason. Check the place the system "
               "tracks it (sys.modules) under the name everyone else uses "
               "BEFORE building a fresh one. Test by obtaining it TWO WAYS and "
               "asserting identity (`is`, not `==`), running UNDER the "
               "framework (as_pytest) -- a standalone script cannot exercise "
               "framework import machinery.")),
    # ---- flask / any web framework: output vocabulary --------------------
    dict(repo="flask", tool="patch", path=None, enclosing=None,
         args=r"(header|label|column|--sort|echo|print)",
         rule=("OUTPUT VOCABULARY: name displayed fields after the ATTRIBUTE "
               "they show, in the codebase's own vocabulary, not the issue "
               "reporter's wording. If an object exposes ALTERNATIVE fields "
               "for one slot (chosen by a mode/flag), find EVERY such field "
               "and cover each with its own field-derived label.")),
    # ---- any repo: adding a parameter the issue proposed ------------------
    dict(repo=None, tool="patch", path=None, enclosing=None,
         args=r"def \w+\([^)]*=\s*(None|False|True|[\"'])",
         rule=("PROPOSED-API NAMING: if this parameter came from the issue, "
               "the reporter's name is a SUGGESTION. Read the signature you "
               "are extending: match the type, shape and naming style of the "
               "options already there (a family of booleans named for their "
               "property beats a string mode borrowed from a builtin).")),
    # ---- any repo: emitting spec-governed text ----------------------------
    dict(repo=None, tool="patch", path=None, enclosing=None,
         args=r"\\\\(left|right|begin|frac)|<\w+>|SELECT |INSERT ",
         rule=("SPEC-GOVERNED OUTPUT: this string is consumed by a renderer/"
               "parser, so its correctness is defined by that format's "
               "published rules, not by neighbouring code -- siblings often "
               "get away with violations their inputs never exercise. Check "
               "the bare AND the nested/wrapped case: this bug class only "
               "appears when the construct nests inside its own kind.")),
]


def fire(repo, tool, path="", enclosing="", args_text="", limit=2):
    """Return the rules whose conditions match this tool call. Pure symbolic."""
    out = []
    for t in TRIGGERS:
        if t.get("repo") and t["repo"] not in (repo or ""):
            continue
        if t.get("tool") and t["tool"] != tool:
            continue
        if t.get("path") and not re.search(t["path"], path or ""):
            continue
        if t.get("enclosing") and not re.search(t["enclosing"], enclosing or ""):
            continue
        if t.get("args") and not re.search(t["args"], args_text or "", re.I):
            continue
        out.append(t["rule"])
        if len(out) >= limit:
            break
    return out
