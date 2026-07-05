import os, sys, tempfile
sys.path.insert(0, os.path.expanduser("~/Code/LLMOS"))
from llmos.store import Store
from llmos.kernel import Kernel
from llmos.cpu import OllamaCPU

# Heterogeneous CPUs: the powerful model foreground, the cheap fast model for idle work.
fg = OllamaCPU(model="ornith:35b", host="http://127.0.0.1:11435", num_predict=1024, log=print)
bg = OllamaCPU(model="llama3.1:8b", host="http://127.0.0.1:11434", num_predict=256, log=print)

db = tempfile.mktemp(suffix=".db")
store = Store(db)
k = Kernel(store, fg, log=print, bg_cpu=bg)
k.boot()

# Foreground: real work on ornith
fpid = k.spawn("store the number 42 under key answer and return it", budget=12)
# Background: idle-time reflection on llama (runs only when the foreground is idle)
bpid = k.spawn("In one short sentence, store under key summary a note that the answer 42 "
               "was saved to memory, then return the summary.", budget=8, background=True)

k.run()

print("\nFG (ornith) result :", k.procs[fpid].result)
print("BG (llama) result  :", k.procs[bpid].result)
print("MEM     :", {kk: store.mem_read('mem', kk) for kk in store.mem_list('mem')})
print("CATALOG :", {kk: store.mem_read('catalog', kk) for kk in store.mem_list('catalog')})
print("### DEMO DONE ###")
