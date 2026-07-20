# pytest-dev/pytest -- repo knowledge

## PROTOCOL: import-machinery bugs have ONE canonical fix (use the import tool)

pytest bugs cluster in its IMPORT MACHINERY (`src/_pytest/pathlib.py`,
`import_path`, importlib mode, namespace packages, rootdir/conftest
collection). When the symptom is "the same module imported twice yields two
different objects" / "state set during import is lost" / "module imported once
under importlib appears twice":

The fix is ALWAYS the same symbolic shape -- BEFORE constructing a fresh module
object, check the cache and return the existing one:

    if mode is ImportMode.importlib:
        module_name = module_name_from_path(path, root)
        with contextlib.suppress(KeyError):        # <-- the fix
            return sys.modules[module_name]
        # ... only now build the module via meta_path / spec

Equivalently `if module_name in sys.modules: return sys.modules[module_name]`.
This is exactly what `import_tool.import_from_path` does (cache check first).
Do not reason it out fresh -- MIRROR the import tool's contract: return the
cached module; build once; cache before exec.

VERIFICATION: this bug is INSIDE pytest's own collector -- it cannot be shown
by a standalone `python -c` script. Reproduce with as_pytest=true (a pytest
test that imports the same module two ways and asserts identity), so the
reproduction runs through the framework that actually exhibits the bug. A
standalone repro will stay red forever and hide a correct fix.

Reference specimen: pytest-11148 (module imported twice under import-mode=
importlib; two `pmxbot.logging` objects; class var `store` lost).

## Environment
- pytest 7.x/8.x era -> Python 3.9+. Tests: `pytest testing/`.
