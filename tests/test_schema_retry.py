"""Schema-gate tests: the model gets one corrective retry, then the CPU fails closed.

Run from the repo root:  PYTHONPATH=. python3 tests/test_schema_retry.py

Uses a scripted subclass of OllamaCPU (overriding _generate) so we can drive exact
model outputs without a live model.
"""
from llmos.cpu import OllamaCPU
from llmos.isa import Op


class ScriptedCPU(OllamaCPU):
    def __init__(self, outputs):
        super().__init__(max_retries=1, log=lambda *a: None)
        self._outputs = list(outputs)
        self.calls = 0

    def _generate(self, pcb, correction=None):
        self.calls += 1
        return self._outputs.pop(0)


class FakePCB:
    goal = "x"
    pc = 0
    context = []


def main():
    pcb = FakePCB()

    # 1. invalid (missing required 'key') then valid -> retries once, returns the valid instruction
    cpu = ScriptedCPU([
        '{"op":"WRITE_MEM","args":{}}',
        '{"op":"WRITE_MEM","args":{"key":"k","value":1}}',
    ])
    instr = cpu.step(pcb)
    assert cpu.calls == 2, f"expected one retry (2 calls), got {cpu.calls}"
    assert instr.op == Op.WRITE_MEM and instr.args["key"] == "k"

    # 2. invalid twice (bad JSON, then unknown op) -> one chance used, fails closed
    cpu2 = ScriptedCPU([
        'not json at all',
        '{"op":"FROB","args":{}}',
    ])
    instr2 = cpu2.step(pcb)
    assert cpu2.calls == 2, f"expected exactly 2 calls, got {cpu2.calls}"
    assert instr2.op == Op.RETURN and instr2.args.get("result") == "SCHEMA VALIDATION FAILED"
    assert "error" in instr2.args

    # 3. valid on the first try -> no retry
    cpu3 = ScriptedCPU(['{"op":"YIELD","args":{}}'])
    instr3 = cpu3.step(pcb)
    assert cpu3.calls == 1 and instr3.op == Op.YIELD

    print("ALL SCHEMA-RETRY TESTS PASSED")


if __name__ == "__main__":
    main()
