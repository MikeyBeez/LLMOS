"""Edit-contract test: with contract={"require_edit": True}, the kernel refuses a
RETURN until the process has made at least one successful fs.edit. A process that
tries to 'read and quit' gets trapped and forced to actually edit.

Run from the repo root:  PYTHONPATH=. python3 tests/test_edit_contract.py
"""
import os
import shutil
import tempfile

from llmos.store import Store
from llmos.kernel import Kernel
from llmos.cpu import MockCPU
from llmos.isa import Instruction, Op

WORKFILE = None   # set in main()


def program(pcb):
    """Mimics ornith's failure mode: tries to RETURN without editing. After the
    kernel traps that, it makes the edit, then returns."""
    made_edit = any(c["op"] == "CALL" and (c["args"] or {}).get("name") == "fs.edit"
                    and isinstance(c["result"], dict) and c["result"].get("edited")
                    for c in pcb.context)
    trapped = any(isinstance(c["result"], dict) and c["result"].get("trap") for c in pcb.context)
    if made_edit:
        return Instruction(Op.RETURN, {"result": "fixed"})
    if trapped:
        return Instruction(Op.CALL, {"name": "fs.edit", "args": {"path": WORKFILE, "old": "bug", "new": "fix"}})
    return Instruction(Op.RETURN, {"result": "done (but I never edited anything!)"})


def main():
    global WORKFILE
    work = tempfile.mkdtemp()
    WORKFILE = os.path.join(work, "code.py")
    open(WORKFILE, "w").write("this line has a bug in it\n")
    db = tempfile.mktemp(suffix=".db")
    store = Store(db)
    pol = {"allowed": [work], "writable": [work], "untrusted": []}
    kernel = Kernel(store, MockCPU({"fix": program}), log=lambda *a: None, fs_policy=pol)
    kernel.boot()
    pid = kernel.spawn("fix", capabilities={"fs.write", "fs.read"}, budget=8,
                       contract={"require_edit": True})
    kernel.run()
    pcb = kernel.procs[pid]

    traps = [c for c in pcb.context if isinstance(c["result"], dict) and c["result"].get("trap")]
    assert traps, "a RETURN with no edit should have been trapped"
    assert "fix" in open(WORKFILE).read(), "the edit should have been forced through"
    assert pcb.status.value == "DONE" and pcb.result == "fixed", pcb.result

    store.close()
    shutil.rmtree(work, ignore_errors=True)
    if os.path.exists(db):
        os.unlink(db)
    print("ALL EDIT-CONTRACT TESTS PASSED")


if __name__ == "__main__":
    main()
