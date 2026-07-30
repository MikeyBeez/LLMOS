"""Extract the reproduction the REPORTER already wrote, instead of inventing one.

WHY THIS EXISTS -- measured on regress18_v4, 2026-07-29/30.

Seven instances walked the entire 6-operation repertoire. Every one of them ended
up using segment 1's candidate patch, and not one ever reached a green
reproduction. django-16873 is the clean case: the correct patch existed at turn 7,
the model then wrote four scripts that PASSED (turns 23, 27, 28, 29), the harness
refused all four -- a script can only be registered by FAILING -- and PAUSED the
model with "change the PATCH". It then spent 3504s and five more segments before
the fallback reached back and used the turn-7 patch, which graded correct.

Meanwhile the problem statement for that instance contained this, verbatim:

    class RegressionTests(SimpleTestCase):
        @setup({"join02": '{% autoescape off %}{{ some_list|join:some_var }}{% endautoescape %}'})
        def test_join02(self):
            some_list = ["<p>Hello World!</p>", "beta & me", "<script>Hi!</script>"]
            some_var = "<br/>"
            output = self.engine.render_to_string("join02", {...})
            self.assertEqual(output, some_var.join(some_list))

A complete test, on the repo's own harness, with the correct assertion, plus the
actual-vs-expected diff and the directory to run it in. We asked the model to
invent one instead.

Measured over all 300 Lite statements: 60% contain a code snippet or traceback,
49% state expected-vs-actual, 32% have both, 11% contain literal test code.

LEAK SAFETY -- this reads ONLY problem_statement, which is the agent's own input;
every SWE-bench agent is given it. It never touches `patch`, `test_patch` or
FAIL_TO_PASS. Two further guards:
  * An extracted reproduction is admitted ONLY if it FAILS on the unmodified base
    tree. A script that passes at base is not exercising the bug, whatever it
    claims, and is discarded.
  * Provenance is recorded and logged on every admission, so anyone reading the
    numbers can see the reproduction came from the reporter and not from us.

Worth stating plainly: for some instances the reporter's snippet resembles what
the maintainer later turned into the graded test. That is a property of the
dataset, identical for anyone who reads the issue, and the reporter says what
SHOULD happen -- never how to make it happen. The fix still has to be found.
"""
import os
import re

FENCE = re.compile(r"```(?:python|py|pycon|console|text)?\s*\n(.*?)```", re.S)
DOCTEST = re.compile(r"(?:^|\n)((?:\s*>>>[^\n]*\n(?:[^\n>][^\n]*\n)*)+)")
TB = re.compile(r"Traceback \(most recent call last\):")
TESTCLASS = re.compile(r"^\s*class\s+\w+\((?:[\w.]*(?:SimpleTestCase|TestCase|"
                       r"TransactionTestCase|unittest\.TestCase))\s*\)\s*:", re.M)
TESTFUNC = re.compile(r"^\s*def\s+test_\w+\s*\(", re.M)
FOLDER = re.compile(r"(?:run(?:ning)?\s+(?:this\s+|it\s+)?inside\s+(?:the\s+)?|"
                    r"place\s+(?:this\s+|it\s+)?in\s+(?:the\s+)?)"
                    r"([\w./-]+?)\s*(?:folder|directory|dir)\b", re.I)

_PY_HINT = re.compile(
    r"^\s*(?:import\s+\w|from\s+[\w.]+\s+import\b|def\s+\w+\s*\(|class\s+\w+"
    r"|@\w|assert\b|print\s*\(|\w+\s*=\s*\S)", re.M)


def _dedent(s):
    lines = [l for l in s.splitlines() if l.strip()]
    if not lines:
        return s
    pad = min(len(l) - len(l.lstrip()) for l in lines)
    return "\n".join(l[pad:] if len(l) >= pad else l for l in s.splitlines())


def _looks_python(s):
    """Is this plausibly runnable python, rather than prose or a diff or HTML?"""
    if not s.strip():
        return False
    if s.lstrip().startswith(("diff --git", "---", "+++", "<", "{%", "$ ", "#!/bin/")):
        return False
    if TB.search(s) and not _PY_HINT.search(s):
        return False        # a bare traceback is a fault locator, not a script
    return bool(_PY_HINT.search(s))


def _doctest_to_script(block):
    """>>> session -> asserts. An expected value under a >>> line is a claim."""
    out, pend = [], None
    for raw in block.splitlines():
        s = raw.strip()
        if s.startswith(">>>") or s.startswith("..."):
            if pend is not None:
                out.append(pend)
                pend = None
            code = s[3:].strip()
            if not code:
                continue
            # `expr` alone followed by an expected value becomes an assertion
            pend = code
        elif s and pend is not None:
            if re.match(r"^[A-Za-z_][\w.]*\s*[=(]", pend) or pend.startswith(
                    ("import ", "from ", "def ", "class ", "for ", "if ")):
                out.append(pend)
            else:
                out.append("assert repr(%s) == %r, (%r, repr(%s))"
                           % (pend, s, s, pend))
            pend = None
        elif not s and pend is not None:
            out.append(pend)
            pend = None
    if pend is not None:
        out.append(pend)
    return "\n".join(out)


def code_blocks(problem_statement):
    """Candidate code blocks from the issue text, best first.

    Each item: {"source": str, "kind": one of testcase|script|doctest,
                "why": short provenance string}
    """
    ps = problem_statement or ""
    found, seen = [], set()

    def add(src, kind, why):
        src = _dedent(src).strip()
        if not src or src in seen or not _looks_python(src):
            return
        seen.add(src)
        found.append({"source": src, "kind": kind, "why": why})

    for m in FENCE.finditer(ps):
        body = m.group(1)
        kind = "testcase" if (TESTCLASS.search(body) or TESTFUNC.search(body)) \
            else "script"
        add(body, kind, "fenced code block in the issue")

    # unfenced test class: walk BACK over the contiguous import run above it --
    # the reporter puts imports on their own lines and the class body needs them.
    # django-16873 needs escape, SimpleTestCase, and crucially the relative
    # `from ..utils import setup`, which is what decides where the file must
    # live -- then forward to the end of the class indent run.
    for m in TESTCLASS.finditer(ps):
        pre = []
        for l in reversed(ps[:m.start()].splitlines()):
            if re.match(r"^\s*(?:from\s+[.\w]+\s+import\b|import\s+[\w.]+)", l):
                pre.insert(0, l)
            elif l.strip() == "" and pre:
                continue
            else:
                break
        lines, keep = ps[m.start():].splitlines(), []
        for i, l in enumerate(lines):
            if i and l.strip() and not l[:1].isspace() and not TESTCLASS.match(l):
                break
            keep.append(l)
        add("\n".join(pre + keep), "testcase", "unfenced test class in the issue")

    for m in DOCTEST.finditer(ps):
        conv = _doctest_to_script(m.group(1))
        if conv.strip():
            add(conv, "doctest", "doctest session in the issue, expected values "
                                 "turned into assertions")

    # unfenced import-led block, last resort
    if not found:
        for m in re.finditer(r"(?:^|\n)((?:(?:from|import)\s+[\w.]+[^\n]*\n)"
                             r"(?:[^\n]*\n){0,40})", ps):
            add(m.group(1), "script", "unfenced code in the issue")
            break

    order = {"testcase": 0, "doctest": 1, "script": 2}
    found.sort(key=lambda b: (order[b["kind"]], -len(b["source"])))
    return found


def named_folder(problem_statement, repo_dir):
    """A directory the reporter told us to run in, if it exists in the repo."""
    for m in FOLDER.finditer(problem_statement or ""):
        frag = m.group(1).strip("/")
        for base in ("", "tests", "test"):
            cand = os.path.join(repo_dir, base, frag) if base else \
                os.path.join(repo_dir, frag)
            if os.path.isdir(cand):
                return os.path.relpath(cand, repo_dir)
        # a trailing fragment like template_tests/filter_tests
        for root, dirs, _f in os.walk(os.path.join(repo_dir, "tests")):
            if root.replace("\\", "/").endswith(frag):
                return os.path.relpath(root, repo_dir)
    return None


def needs_package_home(source):
    """Relative imports (from ..utils import setup) only work inside a package."""
    return bool(re.search(r"^\s*from\s+\.+\w", source, re.M))
