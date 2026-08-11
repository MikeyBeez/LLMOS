"""REPRO_CONTRACT tests (2026-08-11).

The idiom retest measured that advice adoption is proportional to how
mechanical the advice is: "use Agg" (copyable) was adopted from the first
reproduction, "render before asserting" (structural) was ignored -- one
instance wrote eight non-rendering reproductions with the rule in capitals
in its context.  So the structure is enforced at the tool, one-shot:

  - gated off by default;
  - matplotlib only;
  - the FIRST pyplot reproduction with no render is refused with the
    reason, and nothing runs;
  - a resubmission runs unchanged (constructor-raise bugs need no draw);
  - scripts that already render are never touched.

Run: cd ~/Code/LLMOS && python3 -m pytest tests/test_repro_contract.py -q
"""
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import swe_fix_tools as F

PLAIN = "import matplotlib.pyplot as plt\nfig, ax = plt.subplots()\nassert ax.get_xlim()\n"
RENDERED = ("import io, matplotlib.pyplot as plt\nfig, ax = plt.subplots()\n"
            "fig.savefig(io.BytesIO(), format='png')\nassert ax.get_xlim()\n")


class Base(unittest.TestCase):
    repo_name = "matplotlib/matplotlib"

    def setUp(self):
        self.repo = tempfile.mkdtemp()
        self._old = os.environ.get("REPRO_CONTRACT")
        self.handlers, self.state = F.make_fix_handlers(
            self.repo, repo=self.repo_name)

    def tearDown(self):
        if self._old is None:
            os.environ.pop("REPRO_CONTRACT", None)
        else:
            os.environ["REPRO_CONTRACT"] = self._old
        shutil.rmtree(self.repo, ignore_errors=True)

    def repro(self, script):
        return self.handlers["swe.reproduce"](None, {"python_script": script})


class ContractTest(Base):

    def test_off_by_default(self):
        os.environ.pop("REPRO_CONTRACT", None)
        res = self.repro(PLAIN)
        self.assertNotIn("never RENDERS", str(res.get("error", "")))

    def test_first_nonrendering_script_is_refused_not_run(self):
        os.environ["REPRO_CONTRACT"] = "1"
        res = self.repro(PLAIN)
        self.assertIn("error", res)
        self.assertIn("never RENDERS", res["error"])
        self.assertNotIn("exit", res)          # nothing executed

    def test_resubmission_runs(self):
        os.environ["REPRO_CONTRACT"] = "1"
        self.repro(PLAIN)                       # consumes the one hint
        res = self.repro(PLAIN)
        self.assertNotIn("never RENDERS", str(res.get("error", "")))

    def test_rendering_script_is_never_touched(self):
        os.environ["REPRO_CONTRACT"] = "1"
        res = self.repro(RENDERED)
        self.assertNotIn("never RENDERS", str(res.get("error", "")))
        # and the one-shot hint is still unspent for a later plain script
        res2 = self.repro(PLAIN)
        self.assertIn("never RENDERS", str(res2.get("error", "")))


class OtherRepoTest(Base):
    repo_name = "sympy/sympy"

    def test_contract_is_matplotlib_only(self):
        os.environ["REPRO_CONTRACT"] = "1"
        res = self.repro(PLAIN)
        self.assertNotIn("never RENDERS", str(res.get("error", "")))


class ForceDrawTest(unittest.TestCase):
    """REPRO_FORCE_DRAW: the draw is symbolic, not advisory.

    Mikey, 2026-08-11: "You can't just have a rule for drawing. You have to
    make it deterministic. That means you have to do it with symbolic code.
    Force it to draw."  The harness wraps every matplotlib script-mode
    reproduction: Agg pinned in a prologue, every open figure drawn in an
    epilogue.  A bug invisible before the draw becomes a nonzero exit with
    repo frames in the traceback, with zero model cooperation.
    """

    def setUp(self):
        self._old = os.environ.get("REPRO_FORCE_DRAW")
        os.environ["REPRO_FORCE_DRAW"] = "1"

    def tearDown(self):
        if self._old is None:
            os.environ.pop("REPRO_FORCE_DRAW", None)
        else:
            os.environ["REPRO_FORCE_DRAW"] = self._old

    PLAIN = ("import matplotlib.pyplot as plt\n"
             "fig, ax = plt.subplots()\n"
             "print('pre-draw ok')\n")

    def test_off_by_default(self):
        os.environ.pop("REPRO_FORCE_DRAW", None)
        self.assertEqual(F._mpl_force_draw(self.PLAIN, "matplotlib/matplotlib"),
                         self.PLAIN)

    def test_other_repos_untouched(self):
        self.assertEqual(F._mpl_force_draw(self.PLAIN, "sympy/sympy"),
                         self.PLAIN)

    def test_wrap_shape(self):
        w = F._mpl_force_draw(self.PLAIN, "matplotlib/matplotlib")
        self.assertIn("use('Agg', force=True)", w)
        self.assertIn("harness epilogue", w)
        self.assertLess(w.index("Agg"), w.index("import matplotlib.pyplot as plt"))
        compile(w, "w", "exec")                 # wrapped text must still parse

    def test_future_imports_stay_first(self):
        s = "from __future__ import annotations\n" + self.PLAIN
        w = F._mpl_force_draw(s, "matplotlib/matplotlib")
        self.assertTrue(w.startswith("from __future__"))
        compile(w, "w", "exec")

    def test_draw_crash_becomes_nonzero_exit(self):
        """Stub matplotlib whose canvas.draw() raises: plain script exits 0,
        wrapped script exits 3 with the crash in stderr."""
        import subprocess, sys, tempfile, textwrap
        w = F._mpl_force_draw(self.PLAIN, "matplotlib/matplotlib")
        stub = tempfile.mkdtemp()
        os.makedirs(os.path.join(stub, "matplotlib"))
        open(os.path.join(stub, "matplotlib", "__init__.py"), "w").write(
            "def use(*a, **k):\n    pass\n")
        open(os.path.join(stub, "matplotlib", "pyplot.py"), "w").write(
            textwrap.dedent("""
            class _Canvas:
                def draw(self):
                    raise RuntimeError("draw-time crash")
            class _Fig:
                canvas = _Canvas()
            def subplots():
                return _Fig(), object()
            def get_fignums():
                return [1]
            def figure(n):
                return _Fig()
            """))
        env = dict(os.environ, PYTHONPATH=stub)
        r0 = subprocess.run([sys.executable, "-c", self.PLAIN], env=env,
                            capture_output=True, text=True)
        r1 = subprocess.run([sys.executable, "-c", w], env=env,
                            capture_output=True, text=True)
        shutil.rmtree(stub, ignore_errors=True)
        self.assertEqual(r0.returncode, 0)
        self.assertEqual(r1.returncode, 3)
        self.assertIn("draw-time crash", r1.stderr)


if __name__ == "__main__":
    unittest.main()
