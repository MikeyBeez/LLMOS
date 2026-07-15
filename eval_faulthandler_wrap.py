import sys, signal, faulthandler, runpy
faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)
sys.argv = ["swebench.harness.run_evaluation"] + sys.argv[1:]
runpy.run_module("swebench.harness.run_evaluation", run_name="__main__")
