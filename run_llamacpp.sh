#!/bin/bash
# Stand up llama-server on pop for LLMOS benchmarks.
#
# Ollama's own llama-server binary + the ornith:35b GGUF blob it already
# downloaded are on disk — this script points a fresh server at them so we
# can pass args ollama's REST API doesn't expose (grammar, cache_prompt,
# --parallel, --draft-model).
#
# IMPORTANT: only run this when ollama's own llama-server is NOT resident
# with ornith:35b (21GB > 16GB VRAM; two copies won't fit). Simplest way:
#   `ollama stop ornith:35b` before starting this,
#   or wait for ollama's keep_alive to expire.

set -euo pipefail

BIN="${BIN:-/usr/local/lib/ollama/llama-server}"
GGUF="${GGUF:-/var/lib/ollama-models/blobs/sha256-6540411d9e4ca638983e905959188e1be74db7ff62151c69d71a6f2748584f68}"
PORT="${PORT:-8080}"
HOST="${HOST:-127.0.0.1}"
CTX="${CTX:-65536}"
NGL="${NGL:-40}"                 # gpu layers (16GB card + 21GB weights -> partial offload)
PARALLEL="${PARALLEL:-1}"        # single slot for one sequential benchmark client
LOG="${LOG:-/tmp/llamacpp_server.log}"

sudo -n true 2>/dev/null || { echo "will need sudo for reading ollama's blobs"; }

# Verify inputs
[ -x "$BIN" ]   || { echo "no llama-server at $BIN"; exit 1; }
sudo test -f "$GGUF" || { echo "no GGUF at $GGUF"; exit 1; }

# The blobs are owned by user 'ollama'. Make a readable symlink under our home
# so llama-server can open it without sudo.
LINK="$HOME/ornith_35b.gguf"
if [ ! -e "$LINK" ]; then
    sudo ln -s "$GGUF" "$LINK"
    sudo chown -h "$USER" "$LINK" || true
fi

echo "starting llama-server: model=$LINK port=$PORT ngl=$NGL ctx=$CTX log=$LOG"
setsid nohup "$BIN" \
    --model "$LINK" \
    --host "$HOST" --port "$PORT" \
    --n-gpu-layers "$NGL" \
    --ctx-size "$CTX" \
    --parallel "$PARALLEL" \
    --flash-attn auto \
    --cache-reuse 256 \
    --no-webui \
    --log-verbosity 1 \
    > "$LOG" 2>&1 &

PID=$!
echo "llama-server pid=$PID"
sleep 2
if ! kill -0 "$PID" 2>/dev/null; then
    echo "llama-server exited early; tail of log:"
    tail -40 "$LOG"
    exit 1
fi
echo "waiting for /health to be ready..."
for i in $(seq 1 60); do
    if curl -sf "http://$HOST:$PORT/health" > /dev/null 2>&1; then
        echo "server up at http://$HOST:$PORT"
        exit 0
    fi
    sleep 2
done
echo "timeout waiting for /health"
tail -40 "$LOG"
exit 1
