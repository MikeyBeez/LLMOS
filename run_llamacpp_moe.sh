#!/bin/bash
# Stand up llama-server with MoE expert offloading tuned for qwen35moe
# (Ornith-1.0-35B) on a 16GB GPU. The trick: ornith:35b has ~34.7B params
# but only ~3B active per token (8/128 experts routed). Ollama defaults
# spill whole layers to CPU; we can do better by keeping ALL layers'
# ATTENTION + ROUTER + NORM on the GPU (fast, always used) and pushing only
# the big EXPERT MATRICES for some layers to CPU (large, sparsely accessed).
#
# The empirical rule most people converge on: keep attention/norm/router on
# GPU across all layers, keep experts for as many layers as fit, spill the
# rest to CPU. Which layers to spill is workload-dependent — try middle
# layers first (start/end layers are the hot ones for most tasks).
#
# Tuning knobs at the top. Start with the defaults, watch nvidia-smi + the
# server log's `n_gpu_layers = ... offloaded = ...` line, adjust
# EXPERT_CPU_LAYERS until VRAM sits at ~90-95% during inference.

set -euo pipefail

BIN="${BIN:-$HOME/llama.cpp/build/bin/llama-server}"
GGUF="${GGUF:-$HOME/ornith_35b.gguf}"      # symlink created by run_llamacpp.sh
PORT="${PORT:-8080}"
HOST="${HOST:-127.0.0.1}"
CTX="${CTX:-65536}"
PARALLEL="${PARALLEL:-1}"
LOG="${LOG:-/tmp/llamacpp_moe.log}"

# --- MoE offload tuning knobs -------------------------------------------
# `-ot` (override-tensor) takes a regex + backend. Match qwen35moe's expert
# tensor names for a range of layers, force them to CPU. Try WIDER ranges
# if we run out of VRAM (log shows OOM), NARROWER ranges if VRAM sits idle.
#
# qwen35moe naming (verified via `llama-server --dump-metadata`):
#   blk.N.ffn_gate_exps.weight   <- routed expert gate  (big)
#   blk.N.ffn_up_exps.weight     <- routed expert up    (big)
#   blk.N.ffn_down_exps.weight   <- routed expert down  (big)
#   blk.N.ffn_gate_inp.weight    <- ROUTER              (small, keep on GPU)
#   blk.N.attn_*                 <- attention           (keep on GPU)
#   blk.N.ffn_norm.weight        <- layer norm          (keep on GPU)
#
# ORNITH-1.0-35B has 48 layers (0-47). Default here spills the MIDDLE
# 20 layers' experts to CPU: 14-33. Keep first 14 and last 14 on GPU
# (heuristic: early attention + late projection is often hottest).
EXPERT_CPU_LAYERS="${EXPERT_CPU_LAYERS:-blk\\.(1[4-9]|2[0-9]|3[0-3])\\.ffn_.*_exps=CPU}"

# --- verify prereqs -----------------------------------------------------
[ -x "$BIN" ]  || { echo "no llama-server at $BIN"; exit 1; }
[ -e "$GGUF" ] || { echo "no GGUF at $GGUF — run run_llamacpp.sh once to create the symlink"; exit 1; }

# Make sure ollama isn't holding ornith:35b (21GB > 16GB VRAM; two copies won't fit)
if pgrep -f "llama-server.*sha256-6540411d9e4c" > /dev/null; then
    echo "ollama's llama-server is still hosting ornith:35b."
    echo "  run:  ollama stop ornith:35b"
    echo "then re-run this script."
    exit 2
fi

# Ollama's llama-server binary loads its CUDA backend dynamically. The bare
# binary doesn't know where to find libggml-cuda.so — we have to add the
# runner's directory to LD_LIBRARY_PATH. Pop has driver 580 (CUDA 13), so
# cuda_v13 is the correct runner; fall back to cuda_v12 if not present.
if [ -d /usr/local/lib/ollama/cuda_v13 ]; then
    CUDA_RUNNER=/usr/local/lib/ollama/cuda_v13
else
    CUDA_RUNNER=/usr/local/lib/ollama/cuda_v12
fi
export LD_LIBRARY_PATH="$CUDA_RUNNER:/usr/local/lib/ollama:${LD_LIBRARY_PATH:-}"

echo "starting llama-server with MoE expert offload:"
echo "  model:   $GGUF"
echo "  port:    $PORT"
echo "  ctx:     $CTX"
echo "  cuda:    $CUDA_RUNNER"
echo "  offload: -ot $EXPERT_CPU_LAYERS  (middle 20 layers -> CPU)"
echo "  log:     $LOG"

setsid nohup env LD_LIBRARY_PATH="$LD_LIBRARY_PATH" "$BIN" \
    --model "$GGUF" \
    --host "$HOST" --port "$PORT" \
    --n-gpu-layers 99 \
    --override-tensor "$EXPERT_CPU_LAYERS" \
    --ctx-size "$CTX" \
    --parallel "$PARALLEL" \
    --flash-attn auto \
    --cache-reuse 256 \
    --batch-size 512 --ubatch-size 512 \
    --no-webui \
    --log-verbosity 1 \
    > "$LOG" 2>&1 &

PID=$!
echo "llama-server pid=$PID"
sleep 3
if ! kill -0 "$PID" 2>/dev/null; then
    echo "llama-server exited early; tail of log:"
    tail -80 "$LOG"
    exit 1
fi
echo "waiting for /health (model load can take ~30-90s at 21GB with offload)..."
for i in $(seq 1 90); do
    if curl -sf "http://$HOST:$PORT/health" > /dev/null 2>&1; then
        echo "server up at http://$HOST:$PORT"
        # print the offload summary from the log
        grep -E "n_gpu_layers|offloaded|VRAM|tensor.*CPU" "$LOG" | tail -10 || true
        exit 0
    fi
    sleep 2
done
echo "timeout waiting for /health; log tail:"
tail -80 "$LOG"
exit 1
