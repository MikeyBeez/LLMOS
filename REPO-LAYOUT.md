# Repository layout — two machines, one truth

The canonical repository is git@github.com:MikeyBeez/LLMOS.git (branch main).

There are exactly two checkouts:

1. Mac mini ~/Code/LLMOS — a normal clone. Review, docs, backup.
2. pop (192.168.12.174) ~/Code/LLMOS — a normal clone. THIS IS WHERE CODE
   RUNS. The SWE agent imports from here (PYTHONPATH), and llama-server
   launch scripts live here.

Rules (learned the hard way, 2026-07-09/10):

- Edit on pop, commit on pop, push from pop. The Mac PULLS. Never sync by
  scp/rsync again — that produced three divergent copies of the agent and
  a pilot that silently ran old code.
- ~/swe on pop is RUNTIME STATE ONLY: instances.json, work/, mirrors/,
  traces_v2/, remedies.json, training/, logs. No source code. The agent
  entrypoint ~/swe/swe_agent_v2.py is a SYMLINK into this repo.
- A running agent holds the code it imported in memory. Editing files
  during a run affects the NEXT run, not the current one. If a fix must
  apply now, restart the run.
- Before editing on either machine: git pull. After editing: commit+push
  immediately (standing preference: always update git, no asking).
