# pytest-dev/pytest -- repo knowledge

## PROTOCOL: don't import the same thing twice unless there is a reason

If it has already been loaded and is still valid, reuse it. Only build a
fresh one when something genuinely requires that, and be able to say why.
Two copies of one logical thing means state set on one is invisible on the
other -- a bug that looks like magic. Before constructing, check whether it
already exists, in the place the system tracks it, under the same name
everyone else uses.

### How to test it

Get the thing TWO WAYS and assert they are the SAME OBJECT -- identity
(`is`), not equality (`==`); two fresh copies usually compare equal, which
is exactly how this bug hides.

  1. obtain it the way the subsystem under test obtains it
  2. obtain it the way ordinary code obtains it
  3. assert step 1 `is` step 2

Both paths are required. A test that obtains it twice through the SAME
path only proves that path agrees with itself: it goes green while the
real bug survives, which is the classic false verification here.

Stronger form: set a marker (an attribute, a class variable) on the first
copy, then read it back through the second. Missing marker means two
objects, whatever the names say.

Run the test where the duplication actually happens. If the duplicating
machinery belongs to the framework, the test must run UNDER that framework
(reproduce with as_pytest=true); a standalone script cannot exercise it and
will stay red forever even after a correct fix.

pytest bugs cluster in its IMPORT MACHINERY (`src/_pytest/pathlib.py`,
`import_path`, the `--import-mode` variants, namespace packages,
rootdir/conftest collection).

When the symptom is of the form "the same thing is loaded twice" / "state set
during load is lost" / "two objects exist where there should be one":

- There is already a CANONICAL REGISTRY for this kind of object. Find it. Do
  not build a second one.
- The defect is a code path that WRITES to the canonical registry but never
  READS it before constructing.
- The fix shape: consult the canonical registry BEFORE constructing, and look
  it up under the SAME KEY the rest of the system uses. A self-consistent
  internal key can satisfy a narrow unit test and still fail an integration
  test, because other code stored the object under the canonical key.
- Work out, from the code, the identity-agreement table: canonical store /
  canonical key / which function computes that key / which path bypasses it.

MODES: enumerate the modes/flags of the subsystem and ask, for each, which
invariant it bypasses. Modes may differ in HOW they locate a thing, never in
WHETHER it is unique. Reverting the user to a "safe" mode is a retreat, not a
fix -- they chose the mode for a capability.

VERIFICATION: bugs inside pytest's own collector/importer cannot be shown by a
standalone script. Reproduce with as_pytest=true so the reproduction runs
through the framework that exhibits the bug; also exercise BOTH paths that are
supposed to agree (the framework's and the ordinary one), or a fix that only
makes the framework self-consistent will look verified and still be wrong.

## Environment
- pytest 7.x/8.x era -> Python 3.9+. Tests: `pytest testing/`.
