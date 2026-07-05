import os, sys, tempfile
sys.path.insert(0, os.path.expanduser("~/Code/LLMOS"))
from llmos.store import Store
from llmos.kernel import Kernel
from llmos.cpu import OllamaCPU

goal = ("Get the current time and store it under key t. Look at the hour of that time, and "
        "store 'even' or 'odd' under key parity depending on whether the hour number is even "
        "or odd. Return the parity.")
db = tempfile.mktemp(suffix=".db")
store = Store(db)
cpu = OllamaCPU(num_predict=1024, log=print)   # ornith:35b default
k = Kernel(store, cpu, log=print)
k.boot()
pid = k.spawn(goal, budget=16)
k.run()
pcb = k.procs[pid]
print("\nCONTRACT :", pcb.contract)
print("RESULT   :", pcb.result)
print("MEM      :", {kk: store.mem_read('mem', kk) for kk in store.mem_list('mem')})
print("### DEMO DONE ###")
