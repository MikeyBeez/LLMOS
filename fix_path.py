#!/usr/bin/env python3
"""sibling_body got an ABSOLUTE path and silently did nothing.

Observed live at turn 14 of the first SIBLING_BODY run: the model called
patch() with file='/home/bard/swe/work/django__django-16910/django/db/models/
sql/query.py', so result['edited'] was absolute, so check() ran
`git show HEAD:/home/bard/swe/...` which fails, so it returned {} on the exact
patch it exists to look at. The model varies between absolute and repo-relative
paths across calls, so both have to work.
"""
import os
import py_compile

P = "/home/bard/Code/LLMOS/sibling_body.py"
OLD = '''    if not rel_path or not rel_path.endswith(".py") or not written_text:
        return {}
    src = _pristine(repo_dir, rel_path)'''

NEW = '''    if not rel_path or not rel_path.endswith(".py") or not written_text:
        return {}
    # The model calls patch() with an absolute path about as often as a
    # repo-relative one. `git show HEAD:/abs/path` fails silently, which made
    # this whole check a no-op on its first live run. Normalise both shapes.
    if os.path.isabs(rel_path):
        try:
            rel_path = os.path.relpath(rel_path, repo_dir)
        except Exception:
            return {}
    if rel_path.startswith("./"):
        rel_path = rel_path[2:]
    if rel_path.startswith(".."):
        return {}                      # outside the repo; nothing to say
    src = _pristine(repo_dir, rel_path)'''

with open(P, encoding="utf-8") as f:
    s = f.read()
n = s.count(OLD)
if n != 1:
    raise SystemExit("ABORT: anchor count %d != 1" % n)
if "import os" not in s.split("def ")[0]:
    s = s.replace("import ast\nimport re", "import ast\nimport os\nimport re", 1)
s = s.replace(OLD, NEW, 1)
with open(P, "w", encoding="utf-8") as f:
    f.write(s)
py_compile.compile(P, doraise=True)
print("patched + compiled: sibling_body.py")
