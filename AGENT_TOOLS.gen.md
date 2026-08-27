# LLMOS fix-phase tool surface — GENERATED, do not hand-edit

Regenerate with `python3 gen_agent_tools.py > AGENT_TOOLS.gen.md` after any
change to `FIX_TOOLS` in `swe_fix_tools.py`. A hand-maintained tool list is
wrong within a week; this one is read out of the code.

`*` marks a required parameter. **Gate** is the env var that must be `1` for the
tool to appear in the advertised menu — the handler is always registered, only
the schema is withheld. The menu is a BEHAVIOUR surface: adding an entry changes
routing for every instance, which is why new tools ship gated off and are turned
on explicitly in the runner script (`~/swe/run_all300.sh`).

| tool | parameters | gate | what it does |
|---|---|---|---|
| `check` | `snippet*` | always | Answer ONE small question about the code, right now, cheaply. Runs a few lines of python in the venv and gives you back what they PRINT. Registers nothing and verifies nothing -- it cannot help or hurt your reproduction, so use it freely. THIS IS THE TOOL FOR CHECKING A MECHANISM: is this object the same object as that one (use `is`, print the ids); which branch does this flag actually take; what ... |
| `reproduce` | `python_script*, as_pytest` | always | Run a reproduction inside the (verified) venv that demonstrates the bug by EXITING NONZERO. The last failing reproduction becomes the registered one that verify_fix reruns after your patch. Do this FIRST. TWO INSTRUMENTS — pick the one that can actually show THIS bug: (a) default: a plain script (python -c) for crashes, wrong return values, exceptions; (b) as_pytest=true: your script is a PYTEST T... |
| `locate` | `pattern*, file_glob` | always | grep across the repo for a symbol/message/pattern. Searches file CONTENTS only, never file names. Returns file:line matches (up to 40) plus an LLM ranking of the likeliest bug site; on 0 matches it reports whether the glob matched any files and lists files whose NAME matches the pattern. |
| `read_range` | `file*, start*, end*` | always | Read lines [start, end] of a specific file. Follow locate — grep gives you the line number, read_range opens the exact window. |
| `insert_lines` | `file*, after_line*, new_lines*` | `EDIT_SURFACE` | ADD new lines to a file after a given line number. This ADDS; it replaces nothing, so use it whenever the fix is new code rather than changed code -- a new branch, a new helper, an extra guard, an import, a new method on a class. patch and edit_line can only REPLACE existing text, so do not contort a replacement into an insertion by retyping a line you did not want to change. after_line is the 1-b... |
| `rewrite_function` | `file*, name*, new_source*` | `EDIT_SURFACE` | Replace an ENTIRE function or method with new source. Use this when the bug is in the ALGORITHM rather than in one condition or literal -- when the code's whole approach to the problem is wrong and no sequence of small edits inside it can be right. Signals you are in that case: you have read the function and cannot point at a single line to change; earlier attempts edited its conditions and none w... |
| `edit_line` | `file*, line*, old*, new*` | `EDIT_LINE` | Change ONE fragment on ONE line. You give the line number and the exact fragment to replace; the harness rewrites the bytes. You do NOT retype the line's indentation or the rest of the line, so whitespace cannot drift. PREFER THIS over patch for in-place edits: changing an argument, an operator, a literal, a name. The fragment must occur EXACTLY ONCE on that line - it need NOT be unique in the fil... |
| `patch` | `file*, start_line, end_line, old_snippet, new_snippet*` | always | Replace source text in a SOURCE file (test files are refused). TWO WAYS TO ANCHOR. (a) old_snippet: must match EXACTLY and be unique. (b) start_line + end_line: replaces those lines outright. PREFER LINE ANCHORING whenever the text contains backslashes, quotes or escape sequences -- reproducing such a line verbatim through a JSON argument usually fails on over-escaping, while two line numbers cann... |
| `verify_fix` | `` | always | Rerun the registered reproduction script. ok=true when it exits 0 (the bug no longer occurs). submit is only accepted after this passes on a script that previously FAILED. |
| `symbol` | `name*` | always | Instant lookup of a function/class/method by NAME from a mechanical map of the whole repo: every definition site as file:line plus signature and first docstring line. Use this FIRST when you know the name -- it replaces the locate('def X')-then-read_range hunt in one call. locate is still right for non-definition text (strings, calls, comments). |
| `run_tests` | `test_id*` | always | Run an existing test file or test id from the repo's suite as a REGRESSION check (did my patch break something nearby?). This is not the verification gate — verify_fix is. |
| `submit` | `summary*` | always | Terminal call. ONLY accepted after: a reproduction failed (RED), you patched source, and verify_fix passed (GREEN) with a non-empty diff. |
| `differential` | `bug_script*, control_script*` | `DIAG_GATE` | DIAGNOSIS: run the SAME operations twice -- once under the condition the issue names (bug_script) and once WITHOUT it (control_script). If the exits differ, that condition is load-bearing: the bug lives in STATE it changes, and the fix site is usually the function that WRITES that state, not the one where the symptom appears. |
| `declare_site` | `file*, function, role, reason*` | `DIAG_GATE` | DIAGNOSIS: declare WHERE the fix will go and WHY, before patching. role='writer' if the function writes the state your diagnosis found, 'reader' if it only consumes it. Recorded as state, shown every turn; edits outside the declared file are challenged once. |
| `recall` | `ref*` | always | See the FULL output of an earlier tool call. Long outputs are shown summarized, ending with a note and an id like out17; pass that id as ref to get the entire original output back. Nothing is ever discarded -- only summarized. recall retrieves the rest. |

**Total advertised with all gates on: 15.**

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

