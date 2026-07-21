"""Generate the architecture atlas: a navigable call graph, built from AST.

Mikey asked for a user guide and a function index. I argued for a GENERATED
graph instead of prose, because prose rots (the README still describes the
project from before workshop mode existed) and because prose would not have
caught any of today's failures.

Today's failures were all the SAME SHAPE: a thing built correctly and wired to
nothing.

    h_submit      rewritten with an advisory gate -- but `submit` maps to
                  RETURN and the handler is never dispatched. Dead.
    compose_map   built, tested, and referenced by no caller for an hour.
    _research_maps failed silently on a missing import; nothing noticed.

Every one is visible in a single query: WHICH FUNCTIONS ARE DEFINED AND NEVER
CALLED? That question has an exact answer from the AST, and no answer at all
from documentation.

So this emits:
  - every module, every def/class, with location
  - who calls it, and what it calls
  - NEVER-CALLED functions, first, in red -- the wiring check
  - string-dispatched names (handlers["swe.patch"]) resolved so table-driven
    dispatch does not read as dead

Output: docs/atlas.html -- one self-contained file, no dependencies.
"""
import ast, os, sys, glob, html, collections, time

ROOT = os.path.expanduser("~/Code/LLMOS")
OUT = os.path.join(ROOT, "docs", "atlas.html")

defs = {}          # qualified -> {leaf, kind, module, line, doc}
calls = collections.defaultdict(set)      # caller qual -> set(callee leaf)
called_by = collections.defaultdict(set)  # callee leaf -> set(caller qual)
str_refs = collections.Counter()          # leaf -> times seen as a string literal


def index(path, mod):
    try:
        tree = ast.parse(open(path, encoding="utf-8", errors="ignore").read())
    except Exception:
        return

    def first_line(node):
        d = ast.get_docstring(node) or ""
        return d.strip().splitlines()[0][:110] if d.strip() else ""

    def walk(node, prefix, enclosing):
        for ch in getattr(node, "body", []):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                qual = (prefix + "." + ch.name) if prefix else ch.name
                key = mod + "::" + qual
                defs[key] = {"leaf": ch.name, "module": mod, "line": ch.lineno,
                             "kind": ("class" if isinstance(ch, ast.ClassDef)
                                      else "method" if prefix else "function"),
                             "doc": first_line(ch), "qual": qual}
                walk(ch, qual, key)
            else:
                scan(ch, enclosing)

    def scan(node, enclosing):
        for n in ast.walk(node):
            if isinstance(n, ast.Call):
                f = n.func
                name = (f.id if isinstance(f, ast.Name)
                        else f.attr if isinstance(f, ast.Attribute) else None)
                if name and enclosing:
                    calls[enclosing].add(name)
                    called_by[name].add(enclosing)
            elif isinstance(n, (ast.Name, ast.Attribute)):
                # a bare reference: passed as a callback, stored in a dispatch
                # table, aliased. That is wiring, even though it is not a call.
                nm = (n.id if isinstance(n, ast.Name) else n.attr)
                if nm and enclosing:
                    called_by[nm].add(enclosing)
            elif isinstance(n, ast.Constant) and isinstance(n.value, str):
                v = n.value
                if v and (v.startswith("swe.") or v.startswith("_")
                          or v.replace("_", "").isalnum()):
                    str_refs[v.split(".")[-1]] += 1

    walk(tree, "", None)
    # module-level calls
    scan(tree, mod + "::<module>")


SKIP = ("eval_hard", "eval_", "mmlu_", "math_agent", "atlas_doc")
for p in sorted(glob.glob(os.path.join(ROOT, "*.py"))):
    b = os.path.basename(p)
    if any(b.startswith(x) for x in SKIP):
        continue
    index(p, b)

# ---- the wiring check ----------------------------------------------------
ENTRY = ("main", "__init__", "__repr__", "__str__", "__enter__", "__exit__")
never = []
for key, d in defs.items():
    if d["kind"] == "class":
        continue
    leaf = d["leaf"]
    if leaf in ENTRY or leaf.startswith("test_"):
        continue
    if called_by.get(leaf):
        continue
    if str_refs.get(leaf):       # dispatched by string, e.g. handlers["swe.x"]
        continue
    never.append((key, d))
never.sort(key=lambda kv: kv[1]["module"])

# ---- render --------------------------------------------------------------
def esc(s):
    return html.escape(str(s or ""))


rows = []
for key, d in never:
    rows.append(
        "<tr><td class=n>%s</td><td>%s</td><td class=m>%s:%d</td><td>%s</td></tr>"
        % (esc(d["qual"]), esc(d["kind"]), esc(d["module"]), d["line"], esc(d["doc"])))

mods = collections.defaultdict(list)
for key, d in sorted(defs.items(), key=lambda kv: (kv[1]["module"], kv[1]["line"])):
    mods[d["module"]].append((key, d))

sections = []
for mod, items in sorted(mods.items()):
    lis = []
    for key, d in items:
        cb = sorted(called_by.get(d["leaf"], ()))[:6]
        cl = sorted(x for x in calls.get(key, ()) if any(
            v["leaf"] == x for v in defs.values()))[:8]
        dead = " dead" if any(k == key for k, _ in never) else ""
        lis.append(
            "<li class='f%s' id='%s'><b>%s</b> <span class=k>%s</span> "
            "<span class=m>:%d</span>%s"
            "<div class=d>%s</div>"
            "<div class=r><span>called by:</span> %s</div>"
            "<div class=r><span>calls:</span> %s</div></li>"
            % (dead, esc(key), esc(d["qual"]), esc(d["kind"]), d["line"],
               " <span class=warn>NEVER CALLED</span>" if dead else "",
               esc(d["doc"]),
               ", ".join(esc(c.split("::")[-1]) for c in cb) or "<i>nothing</i>",
               ", ".join(esc(c) for c in cl) or "<i>nothing</i>"))
    sections.append("<section><h2>%s <span class=c>%d</span></h2><ul>%s</ul></section>"
                    % (esc(mod), len(items), "".join(lis)))

doc = """<!doctype html><meta charset=utf-8>
<title>LLMOS architecture atlas</title>
<style>
body{font:14px/1.5 -apple-system,Segoe UI,sans-serif;margin:0;background:#0e1116;color:#d7dde5}
header{padding:20px 28px;border-bottom:1px solid #263040;position:sticky;top:0;background:#0e1116;z-index:5}
h1{margin:0 0 4px;font-size:19px}
.sub{color:#7e8b9d;font-size:12.5px}
main{padding:20px 28px;max-width:1100px}
h2{font-size:15px;margin:26px 0 8px;color:#8ab4f8;border-bottom:1px solid #1e2836;padding-bottom:5px}
.c{color:#5c6a7d;font-weight:400;font-size:12px}
ul{list-style:none;padding:0;margin:0}
li.f{padding:8px 12px;margin:5px 0;background:#141a22;border-left:3px solid #263040;border-radius:0 4px 4px 0}
li.f.dead{border-left-color:#e5534b;background:#1d1416}
b{color:#e6edf3}
.k{color:#7e8b9d;font-size:11.5px}
.m{color:#5c6a7d;font-size:11.5px}
.d{color:#9aa7b8;font-size:12.5px;margin:3px 0}
.r{font-size:12px;color:#7e8b9d}
.r span:first-child{color:#5c6a7d;display:inline-block;min-width:72px}
.warn{color:#e5534b;font-weight:600;font-size:11px;letter-spacing:.4px}
table{border-collapse:collapse;width:100%%;margin:6px 0 4px}
td{padding:5px 10px;border-bottom:1px solid #1e2836;font-size:12.5px}
td.n{color:#e5534b;font-weight:600}
.lead{background:#1d1416;border:1px solid #3d2226;border-radius:6px;padding:14px 18px;margin:8px 0 18px}
.lead h2{margin-top:0;color:#e5534b;border:0}
i{color:#4d596b}
</style>
<header><h1>LLMOS architecture atlas</h1>
<div class=sub>generated from AST — %s — %d definitions across %d modules.
Never hand-edited: regenerate with <code>python3 atlas_doc.py</code></div></header>
<main>
<div class=lead><h2>Defined and never called (%d)</h2>
<div class=sub>The wiring check. Every silent failure this project has had was a
thing built correctly and connected to nothing — a rewritten handler that was
never dispatched, a map with no caller, a collector that failed on a missing
import. Names reached only through string dispatch are excluded.</div>
<table>%s</table></div>
%s
</main>""" % (time.strftime("%Y-%m-%d %H:%M"), len(defs), len(mods),
              len(never), "".join(rows) or "<tr><td><i>none</i></td></tr>",
              "".join(sections))

os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w", encoding="utf-8").write(doc)
print("wrote %s" % OUT)
print("  %d definitions, %d modules" % (len(defs), len(mods)))
print("  NEVER CALLED: %d" % len(never))
for key, d in never[:12]:
    print("     %-42s %s:%d" % (d["qual"], d["module"], d["line"]))
