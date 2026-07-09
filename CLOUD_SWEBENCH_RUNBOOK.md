# SWE-bench Verified on a Spot A100 — LLMOS + Ornith-1.0-35B

Goal: get the actual, publication-comparable number for LLMOS-with-ornith:35b on
the full 500-instance SWE-bench Verified set. Published Ornith-1.0-35B is 75.6%
on Verified. We currently have 34.2% on a 76-instance SWE-bench Lite sample.
Different benchmarks + subset noise mean the comparison is meaningless. This run
fixes that.

## Cost estimate

RunPod / Vast.ai spot A100 40GB: ~$0.60–1.20/hr depending on availability. Full
Verified with our 40-turn budget at ~90s/instance = ~12.5 hrs. Round to $10–15
for one run. Do NOT use non-spot on-demand ($1.89–3.00/hr) — spot preemption is
tolerable because we checkpoint per-instance and can resume.

## Choice of provider

- **RunPod**: friendlier UI, Python SDK, pods keep state, good for iterative
  debugging. Slightly pricier.
- **Vast.ai**: cheaper spot, more variance in machines, need to hunt for a good
  host. Fine once we know it works.

Recommend RunPod for the first run, Vast for follow-ups.

## Instance choice

- A100 40GB is enough. Ornith:35b Q4 (21GB) fits fully in VRAM. Full 65K
  context KV-cache adds ~4GB. Total ~25GB — comfortable in 40GB.
- H100 works but is 2–3x the price and only ~1.5x throughput for this workload.
- L40S also fine (48GB, cheaper on Vast).

Skip: A6000 (48GB, PCIe, no NVLink) is fine on paper but often only available
on-demand at $0.99/hr — not much cheaper than A100 spot.

## Pre-launch: verify locally

Before spending money, make sure our swe_agent.py + kernel + cpu code doesn't
have a fresh bug. Rerun a 3-instance Verified batch **locally on pop** first:

```bash
ssh pop-os
cd ~/Code/LLMOS
~/swebench-venv/bin/python swe_verified_select.py 3
PYTHONPATH=~/Code/LLMOS python3 swe_agent.py 3 verified_instances.json
```

If the 3 instances complete cleanly (any resolved count is fine), proceed.

## RunPod launch

1. Sign in to https://runpod.io. Add ~$20 credits.
2. Community Cloud → GPU Pod → Filter: A100 40GB, Spot, region US.
3. Template: **RunPod PyTorch 2.4 CUDA 12.4** (Ubuntu 22.04, Python, git, curl).
4. Volume: 100GB (need room for ornith:35b 21GB + repos + venv).
5. Deploy. Note the SSH command they give you.

## First-boot setup script

SSH in, then run:

```bash
#!/bin/bash
set -euo pipefail

# 1) Ollama for the model server (llama.cpp direct also works — ollama is simpler)
curl -fsSL https://ollama.com/install.sh | sh
sudo systemctl start ollama || (nohup ollama serve > /root/ollama.log 2>&1 &)
sleep 5

# 2) Pull ornith:35b (21GB download; takes 10–20 min on RunPod's link)
ollama pull ornith:35b

# 3) Warm-load so first inference isn't a 60s cold start
curl -s http://127.0.0.1:11434/api/generate \
  -d '{"model":"ornith:35b","prompt":"hi","stream":false,"keep_alive":"24h"}' \
  > /dev/null

# 4) LLMOS
cd /root
git clone https://github.com/MikeyBeez/LLMOS.git
cd LLMOS

# 5) Python deps (uv is faster than pip)
python3 -m venv /root/swe-venv
source /root/swe-venv/bin/activate
pip install --upgrade pip
pip install datasets pandas requests

# 6) SWE-bench harness deps for scoring (per-instance venv creation happens in swe_agent.py)
pip install uv

# 7) Verified dataset
python3 swe_verified_select.py 500   # writes ~/swe/verified_instances.json
```

## Launch the benchmark

```bash
cd /root/LLMOS
export PYTHONPATH=/root/LLMOS
mkdir -p /root/swe/traces
# Runs one-at-a-time, deletes each repo after scoring, keeps only results.json
setsid nohup python3 swe_agent.py 500 verified_instances.json \
    > /root/swe/verified.log 2>&1 &
echo "pid=$!"
tail -f /root/swe/verified.log
```

Monitor GPU:

```bash
watch -n 5 nvidia-smi
```

## Handling spot preemption

Spot instances can be killed with a 2-minute warning. swe_agent.py already
writes results.json incrementally (one line per resolved instance). If the pod
dies mid-run, redeploy, re-mount the volume, and restart — swe_agent.py's
resume logic will skip already-completed instances.

Snapshot results periodically to your own storage:

```bash
# Every 30 min (via cron or watch loop)
scp /root/swe/results.json bard@your-mac:~/Code/LLMOS-cloud-results/verified-$(date -u +%Y%m%dT%H%M%SZ).json
```

Or use `runpodctl` to push to RunPod persistent volume.

## Copy results back

When done (12–15 hours):

```bash
# On the pod
tar czf /root/swe_verified_out.tar.gz /root/swe/results.json /root/swe/traces/
scp /root/swe_verified_out.tar.gz bard@your-mac:~/Downloads/
```

Then locally:

```bash
python3 -c '
import json, collections
d = json.load(open("results.json"))
d = d if isinstance(d, list) else d.get("results", [])
r = sum(1 for x in d if x.get("resolved"))
print(f"SWE-bench Verified: {r}/{len(d)} = {r/len(d):.1%}")
by_repo = collections.Counter((x["repo"], "OK" if x.get("resolved") else "..") for x in d)
for (repo, tag), n in by_repo.items(): print(f"  {repo:<24} {tag}: {n}")
'
```

## What to compare

Publish:
- ornith:35b + LLMOS scaffold: **X/500** on SWE-bench Verified
- Ornith-1.0-35B published: **75.6%** on SWE-bench Verified (per commit 928c16d)
- Delta = LLMOS scaffold contribution (positive = we help; negative = we fight)

For the delta to be meaningful the eval quantization has to match (Q4_K_M
ollama vs. whatever DeepReinforce ran — probably FP16 or BF16). If we come out
5–10 pts lower, that's likely the quantization gap, not scaffold hurt.

## Optional: Vast.ai variant

Vast.ai spot A100 is often 30–50% cheaper. Same procedure, different provisioning
UX. Their CLI (`vastai search offers`, `vastai create instance`) can be scripted
if we want to spin up nightly.

## When NOT to do this

- If we haven't first fixed the calc device gaps (n!, C(n,k), trig) — those
  same limitations will bite on any coding task involving combinatorics or
  numerical methods. Fix locally, then spend the $10.
- If swe_agent.py's edit interface hasn't been tested on non-sympy repos —
  Verified spans 12 repos with different code styles. Do a 5-instance local
  Verified run across ≥3 repos first.
