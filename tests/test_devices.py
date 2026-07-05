"""World-touching device tests: fs.write, fs.list, shell.exec — each sandboxed to
the allowed roots, capability-gated, and (for the privileged ones) revoked on taint.

Run from the repo root:  PYTHONPATH=. python3 tests/test_devices.py
"""
import os
import shutil
import tempfile

from llmos.store import Store
from llmos.kernel import Kernel
from llmos.cpu import MockCPU
from llmos.syscall import SyscallTable, CapabilityError


class P:
    def __init__(self, caps):
        self.capabilities = set(caps)
        self.pid = 1
        self.tainted = False


def main():
    work = tempfile.mkdtemp()
    db = tempfile.mktemp(suffix=".db")
    store = Store(db)
    pol = {"allowed": [work], "writable": [work], "untrusted": []}
    st = SyscallTable(store, fs_policy=pol)

    # fs.write inside the root works
    r = st.dispatch(P({"fs.write"}), "fs.write", {"path": os.path.join(work, "a.txt"), "content": "hello"})
    assert r["bytes"] == 5 and os.path.exists(os.path.join(work, "a.txt"))

    # fs.write outside the root is denied (path-traversal safe)
    for bad in ("/etc/evil.txt", os.path.join(work, "../escape.txt")):
        try:
            st.dispatch(P({"fs.write"}), "fs.write", {"path": bad, "content": "x"})
            assert False, f"should have denied {bad}"
        except CapabilityError:
            pass

    # fs.list
    r = st.dispatch(P({"fs.read"}), "fs.list", {"path": work})
    assert "a.txt" in r["entries"], r

    # fs.edit: targeted unique replace (own file, so it doesn't disturb a.txt)
    open(os.path.join(work, "e.txt"), "w").write("hello world")
    r = st.dispatch(P({"fs.write"}), "fs.edit", {"path": os.path.join(work, "e.txt"), "old": "hello", "new": "HELLO"})
    assert r.get("replaced") == 1 and open(os.path.join(work, "e.txt")).read() == "HELLO world", r
    # fs.edit: snippet not found -> error (no write)
    r = st.dispatch(P({"fs.write"}), "fs.edit", {"path": os.path.join(work, "e.txt"), "old": "zzz", "new": "x"})
    assert "error" in r, r
    # fs.edit: not unique -> error
    open(os.path.join(work, "d.txt"), "w").write("x x")
    r = st.dispatch(P({"fs.write"}), "fs.edit", {"path": os.path.join(work, "d.txt"), "old": "x", "new": "y"})
    assert "error" in r and "unique" in r["error"], r
    # fs.edit requires the fs.write capability
    try:
        st.dispatch(P(set()), "fs.edit", {"path": os.path.join(work, "a.txt"), "old": "H", "new": "h"})
        assert False
    except CapabilityError:
        pass

    # shell.exec runs, cwd sandboxed, output captured
    r = st.dispatch(P({"shell.exec"}), "shell.exec", {"cmd": "echo hi && cat a.txt", "cwd": work})
    assert r["exit_code"] == 0 and "hi" in r["stdout"] and "hello" in r["stdout"], r

    # shell.exec cwd outside roots denied
    try:
        st.dispatch(P({"shell.exec"}), "shell.exec", {"cmd": "ls", "cwd": "/etc"})
        assert False
    except CapabilityError:
        pass

    # capability enforced: a process without the cap cannot exec / write
    for name, args in [("shell.exec", {"cmd": "echo x"}), ("fs.write", {"path": os.path.join(work, "b"), "content": "x"})]:
        try:
            st.dispatch(P(set()), name, args)
            assert False, f"{name} should require its capability"
        except CapabilityError:
            pass

    # through the kernel: a tainted process loses the privileged world devices
    kernel = Kernel(store, MockCPU({}), log=lambda *a: None, fs_policy=pol)
    pid = kernel.spawn("t", capabilities={"fs.write", "shell.exec", "fs.read", "mem.write"}, budget=4)
    pcb = kernel.procs[pid]
    kernel._apply_taint(pcb)   # simulate untrusted data entering the window
    assert "fs.write" not in pcb.capabilities, "taint should revoke fs.write"
    assert "shell.exec" not in pcb.capabilities, "taint should revoke shell.exec"
    assert "fs.read" in pcb.capabilities, "taint should NOT revoke read-only caps"

    store.close()
    shutil.rmtree(work, ignore_errors=True)
    if os.path.exists(db):
        os.unlink(db)
    print("ALL DEVICE TESTS PASSED")


if __name__ == "__main__":
    main()
