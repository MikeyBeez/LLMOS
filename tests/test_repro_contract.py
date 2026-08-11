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


if __name__ == "__main__":
    unittest.main()
