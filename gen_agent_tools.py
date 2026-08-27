#!/usr/bin/env python3
"""Generate AGENT_TOOLS.gen.md from the code, so the fix-phase tool surface
cannot drift from what the agent is actually offered.

Same principle as tool-inventory.md on the Mac: a hand-written list of tools is
wrong within a week. Run this after changing FIX_TOOLS.

    python3 gen_agent_tools.py > AGENT_TOOLS.gen.md
"""
import os, sys, textwrap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

GATES = {  # tool name -> env var that must be "1" for it to be advertised
    "edit_line": "EDIT_LINE",
    "insert_lines": "EDIT_SURFACE",
    "rewrite_function": "EDIT_SURFACE",
    "differential": "DIAG_GATE",
    "declare_site": "DIAG_GATE",
}

# Advertise everything so the generated page is the FULL surface, then mark
# which entries are gated off by default.
for _v in set(GATES.values()):
    os.environ[_v] = "1"

import swe_fix_tools as T

def first_sentence(text, limit=400):
    text = " ".join((text or "").split())
    return text[:limit] + ("..." if len(text) > limit else "")

rows = []
for t in T.FIX_TOOLS:
    fn = t.get("function", {})
    name = fn.get("name", "?")
    params = (fn.get("parameters") or {}).get("properties") or {}
    req = set((fn.get("parameters") or {}).get("required") or [])
    sig = ", ".join(("%s*" % p) if p in req else p for p in params)
    rows.append((name, sig, GATES.get(name, ""), first_sentence(fn.get("description"))))

print("# LLMOS fix-phase tool surface — GENERATED, do not hand-edit")
print()
print("Regenerate with `python3 gen_agent_tools.py > AGENT_TOOLS.gen.md` after any")
print("change to `FIX_TOOLS` in `swe_fix_tools.py`. A hand-maintained tool list is")
print("wrong within a week; this one is read out of the code.")
print()
print("`*` marks a required parameter. **Gate** is the env var that must be `1` for the")
print("tool to appear in the advertised menu — the handler is always registered, only")
print("the schema is withheld. The menu is a BEHAVIOUR surface: adding an entry changes")
print("routing for every instance, which is why new tools ship gated off and are turned")
print("on explicitly in the runner script (`~/swe/run_all300.sh`).")
print()
print("| tool | parameters | gate | what it does |")
print("|---|---|---|---|")
for name, sig, gate, desc in rows:
    print("| `%s` | `%s` | %s | %s |" % (name, sig, ("`%s`" % gate) if gate else "always", desc))
print()
print("**Total advertised with all gates on: %d.**" % len(rows))
print()
print(textwrap.dedent("""\
    ## The edit surface (added 2026-08-26)

    Until this date the agent could only REPLACE existing text: `patch` swaps
    `old_snippet` for `new_snippet` and explicitly refuses an empty `old_snippet`;
    `edit_line` swaps a fragment on one line. There was no way to say "add these
    lines here" or "this whole function is wrong".

    That was measured, not guessed. On the never-solved instances the model made
    ZERO patch calls — django-11019's tool histogram over a 48-minute run reads
    `locate 15, read_range 9, patch 0`. It read the right file and the right
    function and never attempted an edit, because nothing it could say fit.

    A survey of the correct fix for all 300 instances says why:

    | set | median +lines | median longest deleted run | >=8-line deletion | >=25 lines added |
    |---|---|---|---|---|
    | never-solved (124) | 6 | 1 | 6% | 14% |
    | solvable (176) | 4 | 1 | 2% | 0% |

    The never-solved fixes are mostly ADDITIONS, not rewrites — 11910 is +4/-0,
    11797 +3/-3, 11564 +30/-1. So two tools, not one:

    - `insert_lines` — adds lines after a 1-based line (0 = top of file), verbatim,
      because indentation is semantic and stays the caller's business.
    - `rewrite_function` — replaces a whole `def` including decorators, resolving
      `Class.method` or an unambiguous bare name.

    Both syntax-check and REVERT on failure, and both refuse test files.

    ### Every edit tool writes atomically

    `open(path, "w")` truncates immediately, so a crash between truncate and write
    leaves the source file EMPTY — the agent destroying the checkout it is editing,
    in a way the syntax check cannot catch because there is nothing left to parse.
    All four edit tools now go through `_atomic_write()`: temp file in the same
    directory, flush, fsync, copy the original's mode (mkstemp makes 0600), then
    `os.replace()`. The undo for a rejected edit lives in memory, not in a `.bak`
    beside the source.

    ### Known gap

    `rewrite_function` cannot add imports, so a structural fix still needs a
    `patch` or `insert_lines` call for the import block.

    ### Adoption

    Read it from `~/swe/research/postmortem/<instance_id>.jsonl` -> `features`.
    NOT from the results JSON, which has no features key, and NOT from the run log,
    which truncates tool results at ~150 chars. Relevant counters:
    `rewrite_function`, `insert_lines`, `no_edit_yet_shown`, `locate_assist_*`,
    `locate_budget_spent`.
    """))
