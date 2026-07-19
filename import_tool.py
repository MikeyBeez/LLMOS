"""import_tool: the one correct way to import, symbolically.

Importing a module by name is a solved problem with a single correct form:
return the already-loaded module if it exists, otherwise load it exactly once
and cache it. Doing it any other way (constructing a fresh module object while
one already lives in sys.modules) creates TWO objects for one logical module --
state set on one is invisible on the other. That is the pytest-11148 bug, and
the whole double-import / namespace-package / importlib-mode family.

Use this in reproductions instead of hand-rolling sys.path + __import__, and
MIRROR it in the fix: any import path that builds a module must first do the
sys.modules check this function does.
"""
import importlib
import importlib.util
import sys


def cached_import(module_name):
    """Return sys.modules[module_name] if present, else import once and cache.
    Idempotent: cached_import(x) is cached_import(x) is import_module(x)."""
    mod = sys.modules.get(module_name)
    if mod is not None:
        return mod
    return importlib.import_module(module_name)


def import_from_path(module_name, path):
    """Load a module from an explicit file path -- but honor the cache FIRST,
    the way a correct importer must. Returns the SAME object on repeat calls."""
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module          # cache BEFORE exec (import contract)
    spec.loader.exec_module(module)
    return module


def imports_same_object(module_name, path):
    """A ready-made reproduction for double-import bugs: import the same module
    two ways and report whether they are the SAME object. Correct machinery ->
    True; buggy build-instead-of-reuse machinery -> False."""
    a = cached_import(module_name)
    b = import_from_path(module_name, path)
    return a is b
