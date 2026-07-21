"""The name expert: one canonical registry for the FUNCTION space.

Mikey: "Have you built a canonical name expert for the function space? Once
functions are created they have to be entered into the name expert. Then when
you want to use a function, you look it up and get the correct name."

schema.py did this for the DATA space (field names). This does it for functions,
because the same failure kept happening:

    _clip_result(...)                      invented; NameError, swallowed
    compose_map(f, "Cls._print_X")         qualified name; AST holds the LEAF,
                                           lookup failed, feature silently dead
    handlers["swe.check"]                  string key into a dispatch table
    getattr-by-string across modules       no check at all

Every one is a name asserted rather than resolved, failing silently or late.

WHAT IT DOES
  index(roots)      walk .py files, record every def/class with its qualified
                    name, leaf name, file and line -- mechanically, from AST
  resolve(name)     accept ANY form (leaf, Class.method, module.Class.method)
                    and return the canonical record, or raise KeyError naming
                    close matches
  exists(name)      cheap boolean for guards

It resolves ACROSS the qualified/leaf boundary, which is the specific crossing
that silently disabled the composition map. Callers stop writing
`name.split(".")[-1]` in five different places, each slightly differently.

USE
    from names import HARNESS
    HARNESS.resolve("compose_map")                    -> record
    HARNESS.resolve("PrettyPrinter._print_Integral")  -> resolves via the leaf
    HARNESS.resolve("_clip_result")                   -> KeyError: did you mean...

    from names import NameExpert
    repo = NameExpert.for_path("/home/bard/swe/work/sympy__sympy-18057")
"""
import ast, os, difflib, glob


class NameExpert:
    def __init__(self, name="index"):
        self.name = name
        self.by_qual = {}      # "Class.method" / "func" -> record
        self.by_leaf = {}      # "method" -> [records]

    # ---- building -------------------------------------------------------
    def add(self, qual, leaf, kind, loc):
        rec = {"canonical": qual, "leaf": leaf, "kind": kind, "loc": loc}
        self.by_qual[qual] = rec
        self.by_leaf.setdefault(leaf, []).append(rec)
        return rec

    def index_file(self, path, rel=None):
        rel = rel or os.path.basename(path)
        try:
            tree = ast.parse(open(path, encoding="utf-8", errors="ignore").read())
        except Exception:
            return 0
        n = 0

        def walk(node, prefix):
            nonlocal n
            for child in getattr(node, "body", []):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    qual = (prefix + "." + child.name) if prefix else child.name
                    self.add(qual, child.name,
                             "method" if prefix else "function",
                             "%s:%d" % (rel, child.lineno))
                    n += 1
                    walk(child, qual)          # nested defs
                elif isinstance(child, ast.ClassDef):
                    qual = (prefix + "." + child.name) if prefix else child.name
                    self.add(qual, child.name, "class", "%s:%d" % (rel, child.lineno))
                    n += 1
                    walk(child, qual)

        walk(tree, "")
        return n

    def index_tree(self, root, pattern="**/*.py", skip=("test", ".venv", ".git")):
        total = 0
        for p in glob.glob(os.path.join(root, pattern), recursive=True):
            if any(s in p for s in skip):
                continue
            total += self.index_file(p, os.path.relpath(p, root))
        return total

    @classmethod
    def for_path(cls, root, **kw):
        e = cls(name=os.path.basename(root.rstrip("/")))
        e.index_tree(root, **kw)
        return e

    # ---- using ----------------------------------------------------------
    def resolve(self, name, strict=True):
        """Any form in, canonical record out. Crosses qualified<->leaf."""
        if not name:
            raise KeyError("empty name")
        if name in self.by_qual:
            return self.by_qual[name]
        leaf = name.split(".")[-1]
        hits = self.by_leaf.get(leaf)
        if hits:
            if len(hits) == 1:
                return hits[0]
            # ambiguous leaf: prefer one whose qualified name ends with `name`
            for h in hits:
                if h["canonical"].endswith(name):
                    return h
            if not strict:
                return hits[0]
            raise KeyError("%r is ambiguous in %s -- %d definitions: %s"
                           % (name, self.name, len(hits),
                              ", ".join("%s (%s)" % (h["canonical"], h["loc"])
                                        for h in hits[:5])))
        near = difflib.get_close_matches(leaf, list(self.by_leaf), n=3, cutoff=0.6)
        raise KeyError("%r is not a known name in %s.%s"
                       % (name, self.name,
                          (" Did you mean %s?" % ", ".join(repr(n) for n in near))
                          if near else ""))

    def exists(self, name):
        try:
            self.resolve(name)
            return True
        except KeyError:
            return False

    def leaf_of(self, name):
        """The AST-level name, resolved rather than string-split."""
        return self.resolve(name)["leaf"]

    def __len__(self):
        return len(self.by_qual)


# the harness's own function space, built on import
HARNESS = NameExpert.for_path(os.path.expanduser("~/Code/LLMOS"),
                              pattern="*.py")
