#!/usr/bin/env python3
"""LLMOS on MMLU (subset), NATIVE tool-calling.

Ornith is trained to emit <tool_call> XML blocks that ollama's qwen3 parser
surfaces as OpenAI-style tool_calls. Using the same protocol the model was
trained on beats trying to squeeze it into our interpretive JSON-ISA.

MMLU is single-turn: one `answer(letter)` tool. When the model emits it, we
RETURN the letter and score.

    PYTHONPATH=~/Code/LLMOS python3 mmlu_agent.py [N]
"""
import json, os, re, sys, time

sys.path.insert(0, os.path.expanduser("~/Code/LLMOS"))
from llmos.store import Store
from llmos.kernel import Kernel
from tool_call_cpu import ToolCallCPU

HOST   = "http://127.0.0.1:11434"
MODEL  = "ornith:35b"
NUMCTX = 65536
NUMPRED = 2048        # ornith opens with <think> block; give it real room
TEMP   = 0.6          # Ornith model card: temp=0.6 top_p=0.95 top_k=20
BUDGET = 4
INST   = os.path.expanduser("~/mmlu/instances.json")
OUT    = os.path.expanduser("~/mmlu/results.json")
STORE  = os.path.expanduser("~/mmlu/store.db")

os.makedirs(os.path.dirname(STORE), exist_ok=True)

# Native tool-calling schema. `answer` is the terminal tool — when the model
# calls it, we RETURN the letter.
TOOLS = [
    {"type": "function", "function": {
        "name": "answer",
        "description": "Commit to your final answer as one letter: A, B, C, or D.",
        "parameters": {"type": "object",
                       "properties": {"letter": {"type": "string", "enum": ["A", "B", "C", "D"]}},
                       "required": ["letter"]}}},
]
TOOL2SYS = {"answer": "RETURN"}   # 'RETURN' target → Instruction(RETURN, ...)

SYSTEM = (
    "You are an expert answering multiple-choice questions. Read the question, "
    "reason briefly, then call the `answer` tool with exactly one letter "
    "(A, B, C, or D)."
)


def prompt_for(inst):
    labels = "ABCD"
    body = "\n".join(f"{labels[i]}. {c}" for i, c in enumerate(inst["choices"]))
    return f"Question: {inst['question']}\n\n{body}"


_LETTER = re.compile(r"\b([ABCD])\b")

def extract_letter(result):
    if not result:
        return None
    if isinstance(result, dict):
        v = result.get("letter") or result.get("answer") or result.get("result")
        if v: return extract_letter(v)
    s = str(result).strip().upper()
    if s in "ABCD":
        return s
    m = _LETTER.search(s)
    return m.group(1) if m else None


def run_one(kernel, inst):
    goal = prompt_for(inst)
    pid = kernel.spawn(goal, budget=BUDGET)
    t0 = time.time()
    kernel.run()
    pcb = kernel.procs[pid]
    letter = extract_letter(pcb.result)
    return {
        "id": inst["id"], "subject": inst["subject"],
        "gold": inst["answer"], "raw": pcb.result, "letter": letter,
        "correct": letter == inst["answer"],
        "budget_used": BUDGET - pcb.budget,
        "seconds": round(time.time() - t0, 1),
        "status": pcb.status.value,
    }


def main():
    with open(INST) as f:
        instances = json.load(f)
    if len(sys.argv) > 1:
        instances = instances[: int(sys.argv[1])]

    store = Store(STORE)
    cpu = ToolCallCPU(tools=TOOLS, tool2sys=TOOL2SYS, system_prompt=SYSTEM,
                     model=MODEL, host=HOST, temperature=TEMP,
                     num_predict=NUMPRED, num_ctx=NUMCTX, keep_alive="24h")
    kernel = Kernel(store, cpu, project="mmlu")
    kernel.boot()

    results, correct = [], 0
    for i, inst in enumerate(instances, 1):
        r = run_one(kernel, inst)
        results.append(r); correct += int(r["correct"])
        print(f"[{i:>3}/{len(instances)}] {inst['subject']:<28} gold={r['gold']} "
              f"pred={r['letter']} {'OK' if r['correct'] else '.'}  {r['seconds']}s",
              flush=True)
        with open(OUT, "w") as f:
            json.dump({"n": i, "correct": correct, "score": correct / i,
                       "results": results}, f, indent=1)
    print(f"\nMMLU subset ({len(instances)}): {correct}/{len(instances)} = {correct/len(instances):.1%}")


if __name__ == "__main__":
    main()
