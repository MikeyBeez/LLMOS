"""Unit test for the STUCK circuit-breaker (env STUCK_ESCALATE). Drives the real
h_patch / h_check from make_fix_handlers so it tests the shipped code path, not a
re-implementation."""
import os, sys, tempfile, textwrap
sys.path.insert(0, os.path.expanduser("~/Code/LLMOS"))
import swe_fix_tools as T

def _setup():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "mod.py")
    open(p, "w").write("a = 1\nb = 2\nc = 3\nd = 4\n")
    handlers, state = T.make_fix_handlers(d)
    return d, handlers, state

def test_off_applies_even_when_stuck():
    os.environ.pop("STUCK_ESCALATE", None)
    d, h, state = _setup()
    state["stuck"] = 9
    r = h["swe.patch"](None, {"file": "mod.py", "start_line": 2, "end_line": 2,
                              "new_snippet": "b = 20"})
    assert "edited" in r, ("gate OFF must apply patch", r)
    assert state["stuck"] == 0, ("successful write resets stuck", state["stuck"])
    print("PASS off_applies_even_when_stuck")

def test_on_redirects_at_3():
    os.environ["STUCK_ESCALATE"] = "1"
    d, h, state = _setup()
    state["stuck"] = 3
    r = h["swe.patch"](None, {"file": "mod.py", "start_line": 2, "end_line": 2,
                              "new_snippet": "b = 20"})
    assert "edited" not in r, ("gate ON at 3 must NOT apply", r)
    assert "NO PROGRESS" in r.get("error", ""), r
    assert open(os.path.join(d, "mod.py")).read().count("b = 2\n") == 1, "file untouched"
    print("PASS on_redirects_at_3")

def test_on_hard_latches_at_5():
    os.environ["STUCK_ESCALATE"] = "1"
    d, h, state = _setup()
    state["stuck"] = 5
    r = h["swe.patch"](None, {"file": "mod.py", "start_line": 2, "end_line": 2,
                              "new_snippet": "b = 20"})
    assert "PATCH LOCKED" in r.get("error", ""), r
    print("PASS on_hard_latches_at_5")

def test_on_below_threshold_applies():
    os.environ["STUCK_ESCALATE"] = "1"
    d, h, state = _setup()
    state["stuck"] = 2
    r = h["swe.patch"](None, {"file": "mod.py", "start_line": 2, "end_line": 2,
                              "new_snippet": "b = 20"})
    assert "edited" in r, ("stuck=2 is below threshold, must apply", r)
    assert state["stuck"] == 0, state["stuck"]
    print("PASS on_below_threshold_applies")

def test_check_resets_stuck():
    os.environ["STUCK_ESCALATE"] = "1"
    d, h, state = _setup()
    state["stuck"] = 4
    try:
        h["swe.check"](None, {"snippet": "print(1)"})
    except Exception:
        pass
    assert state["stuck"] == 0, ("check() must reset stuck", state["stuck"])
    print("PASS check_resets_stuck")

if __name__ == "__main__":
    test_off_applies_even_when_stuck()
    test_on_redirects_at_3()
    test_on_hard_latches_at_5()
    test_on_below_threshold_applies()
    test_check_resets_stuck()
    os.environ.pop("STUCK_ESCALATE", None)
    print("ALL STUCK TESTS PASS")
