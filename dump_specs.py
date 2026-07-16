import json, os
from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS as M
out = {}
for repo, vers in M.items():
    out[repo] = {}
    for v, spec in vers.items():
        out[repo][str(v)] = {k: spec.get(k) for k in
                             ("python", "packages", "pip_packages", "install", "pre_install", "test_cmd")}
p = os.path.expanduser("~/swe/swebench_specs.json")
json.dump(out, open(p, "w"), indent=0, default=str)
n = sum(len(v) for v in out.values())
print("wrote", p, "|", len(out), "repos,", n, "repo-version specs")
