"""Magic-string checker (Mikey): verify agreement-at-a-distance, BEFORE the maps.

Measured: when a gold fix touches a string-as-discriminator, we fail 75% vs 51%
-- and the class is 2.5x over-represented in the never-resolved tail. The
disease: a literal here must agree with a literal there, and nothing checks.

This is the checker. When a patch WRITES a discriminator string (a dict key, a
== comparison, a getattr name, a prefix check, a membership tuple), verify the
value against the package:

  AGREES     -> show the agreement set (where else this exact value is used as
                a discriminator -- these are the places that must stay in sync)
  NEAR-MISS  -> the exact value appears NOWHERE, but a close variant does
                (case, underscore/dash, singular/plural). Almost always a typo
                or the reporter's word instead of the codebase's.
  ORPHAN     -> the value appears nowhere else in the package at all. A brand
                new label only your code knows. Sometimes right (a genuinely
                new key); usually the bug.

Index built once per checkout from AST, cached OUTSIDE the repo tree.
Ubiquitous strings (>40 uses) and trivia (len<2) are skipped -- no signal.
"""
import ast, os, re, json, time, sys, collections

CACHE = os.path.expanduser("~/swe/magicmaps")

# discriminator positions, for classifying occurrences
def _kind_of(parent, node):
    if isinstance(parent, ast.Subscript) and parent.slice is node:
        return "key"
    if isinstance(parent, ast.Compare):
        return "comparison"
    if isinstance(parent, ast.Call):
        f = parent.func
        nm = (f.attr if isinstance(f, ast.Attribute)
              else f.id if isinstance(f, ast.Name) else "")
        if nm in ("getattr", "hasattr", "setattr", "get", "pop", "setdefault",
                  "startswith", "endswith"):
            return nm
    if isinstance(parent, (ast.Tuple, ast.List, ast.Set)):
        return "membership"
    return None


def build(repo_dir, pkg=None):
    t0 = time.time()
    root = os.path.join(repo_dir, pkg) if pkg else repo_dir
    disc = collections.defaultdict(list)   # value -> [{loc, kind}]
    every = collections.Counter()          # value -> total occurrences anywhere
    files = 0
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d not in
                  (".git", ".venv", ".condaenv", "__pycache__", "node_modules")
                  and not d.startswith("test")]
        for fn in fns:
            if not fn.endswith(".py") or fn.startswith("test_"):
                continue
            p = os.path.join(dp, fn)
            rel = os.path.relpath(p, repo_dir)
            try:
                tree = ast.parse(open(p, encoding="utf-8", errors="ignore").read())
            except Exception:
                continue
            files += 1
            parent_of = {}
            for parent in ast.walk(tree):
                for ch in ast.iter_child_nodes(parent):
                    parent_of[ch] = parent
            for n in ast.walk(tree):
                if isinstance(n, ast.Constant) and isinstance(n.value, str):
                    v = n.value
                    if not (2 <= len(v) <= 60) or "\n" in v:
                        continue
                    every[v] += 1
                    k = _kind_of(parent_of.get(n), n)
                    if k:
                        disc[v].append({"loc": "%s:%d" % (rel, n.lineno), "kind": k})
    os.makedirs(CACHE, exist_ok=True)
    blob = {"built": int(time.time()), "files": files,
            "disc": {k: v[:12] for k, v in disc.items()},
            "every": dict(every)}
    json.dump(blob, open(os.path.join(
        CACHE, os.path.basename(os.path.abspath(repo_dir)) + ".json"), "w"))
    print("magic index: %d files, %d discriminator values in %.1fs"
          % (files, len(disc), time.time() - t0))
    return blob


def _load(repo_dir):
    p = os.path.join(CACHE, os.path.basename(os.path.abspath(repo_dir)) + ".json")
    if os.path.isfile(p):
        try:
            return json.load(open(p))
        except Exception:
            pass
    pkg = "src" if os.path.isdir(os.path.join(repo_dir, "src")) else None
    return build(repo_dir, pkg)


def _variants(v):
    out = {v.lower(), v.upper(), v.replace("-", "_"), v.replace("_", "-"),
           v.capitalize(), v.title()}
    out |= {v + "s", v[:-1]} if not v.endswith("s") else {v[:-1]}
    out.discard(v)
    return out


# strings written in DISCRIMINATOR position inside a patch snippet (regex on
# the fragment -- fragments rarely parse as full AST)
SNIPPET_PATS = [
    r'\[\s*[\'"]([A-Za-z_][\w .-]{1,50})[\'"]\s*\]',
    r'[=!]=\s*[\'"]([^\'"\n]{2,50})[\'"]',
    r'(?:getattr|hasattr|setattr)\s*\(\s*[^,]+,\s*[\'"](\w{2,50})[\'"]',
    r'\.(?:get|pop|setdefault|startswith|endswith)\s*\(\s*[\'"]([^\'"\n]{2,50})[\'"]',
    r'\bin\s*\(\s*[\'"]([^\'"\n]{2,50})[\'"]',
]


def check_snippet(repo_dir, snippet, limit=4):
    """Verdicts for every discriminator string the snippet writes."""
    found = []
    seen = set()
    for pat in SNIPPET_PATS:
        for m in re.finditer(pat, snippet or ""):
            v = m.group(1)
            if v not in seen:
                seen.add(v)
                found.append(v)
    if not found:
        return None
    m = _load(repo_dir)
    disc, every = m.get("disc") or {}, m.get("every") or {}
    out = []
    for v in found[:limit]:
        total = every.get(v, 0)
        if total > 40:
            continue                      # ubiquitous; no signal
        sites = disc.get(v) or []
        if sites:
            out.append({"value": v, "verdict": "AGREES",
                        "sites": ["%s (%s)" % (s["loc"], s["kind"])
                                  for s in sites[:5]],
                        "note": ("used as a discriminator at %d other site(s) "
                                 "-- these must stay in agreement with your "
                                 "edit" % len(sites))})
        else:
            near = [w for w in _variants(v) if w in every]
            if near:
                out.append({"value": v, "verdict": "NEAR-MISS",
                            "existing_variants": near[:4],
                            "note": ("this EXACT value appears nowhere in the "
                                     "package, but close variants do. Almost "
                                     "always a typo or the issue reporter's "
                                     "word where the codebase uses its own -- "
                                     "match the existing variant unless you "
                                     "are deliberately introducing a new "
                                     "name.")})
            elif total == 0:
                out.append({"value": v, "verdict": "ORPHAN",
                            "note": ("appears NOWHERE else in the package. A "
                                     "label only your code knows. If another "
                                     "component must recognise it (a caller, "
                                     "a registry, a test), nothing will -- "
                                     "verify the exact expected value with "
                                     "check() before relying on it.")})
    return out or None


if __name__ == "__main__":
    r = check_snippet(sys.argv[1], sys.argv[2])
    print(json.dumps(r, indent=1) if r else "(no discriminator strings)")
