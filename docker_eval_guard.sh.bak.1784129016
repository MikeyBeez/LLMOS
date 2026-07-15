#!/usr/bin/env bash
# docker_eval_guard.sh -- safe wrapper around the SWE-bench Docker eval for overnight audits.
#
# WHY: the AUTHORITATIVE SWE-bench Docker eval (swebench.harness.run_evaluation) can
# DEADLOCK -- observed 2026-07-15 (multiple cycles): all Python threads parked in
# futex_do_wait, ~1% CPU, 0 docker images / 0 build cache, NO progress. On 2026-07-15
# ~09:53 CDT this was REPRODUCED even with a WARM registry and a SUCCESSFUL HuggingFace
# dataset load (docker-py 7.1.0 handshake = 0.02s; CLI + low-level + high-level SDK builds
# of a trivial image all succeed in <0.1s). So the hang is INSIDE run_evaluation's build
# orchestration, BEFORE any build step dispatches -- not the network, not the SDK, not a
# cold cache. An unguarded eval silently consumes an entire overnight cycle.
#
# This wrapper makes a hang FAIL FAST and VISIBLY:
#   * validates the predictions file (exists, non-empty JSON list)
#   * preflights the docker image cache and warns when it is cold
#   * EARLY-STALL WATCHDOG (new): samples a strictly-growing progress token
#     (docker image count | running containers | bytes in the run logs); if it does not
#     move for --grace seconds (default 300) it declares a deadlock and aborts in ~grace
#     seconds instead of waiting out the full --timeout. A genuinely slow cold build streams
#     into run_instance.log and grows images/build cache, so this does NOT false-positive on
#     "slow but progressing".
#   * enforces a hard timeout (SIGTERM then SIGKILL) as a backstop
#   * on stall/timeout, force-removes any leftover containers for this run_id
#   * distinguishes a stall/deadlock (exit 2) from a genuine eval error (exit 1)
#   * --selftest verifies the watchdog decision logic WITHOUT docker (stalled child killed
#     within grace; progressing child left alone)
#
# ANSWER-LEAKAGE: this only runs the authoritative scorer on the model's OWN prediction.
# It writes no results and injects nothing into any instance. General/scoring-layer only.
#
# Usage:
#   docker_eval_guard.sh --preds PATH --run-id ID --instances "id1 id2 ..." \
#                        [--timeout SECS] [--grace SECS] [--venv PATH] [--dataset NAME] \
#                        [--workers N] [--dry-run]
#   docker_eval_guard.sh --selftest
# Defaults: timeout=1800  grace=300  sample=20  venv=~/swebench-venv  dataset=SWE-bench/SWE-bench_Lite  workers=1
set -euo pipefail

PREDS="" RUN_ID="" INSTANCES="" TIMEOUT=1800
VENV="$HOME/swebench-venv" DATASET="SWE-bench/SWE-bench_Lite" WORKERS=1 DRY=0
GRACE="${GRACE:-300}" SAMPLE="${SAMPLE:-20}" SELFTEST=0

while [ $# -gt 0 ]; do
  case "$1" in
    --preds)     PREDS="$2"; shift 2 ;;
    --run-id)    RUN_ID="$2"; shift 2 ;;
    --instances) INSTANCES="$2"; shift 2 ;;
    --timeout)   TIMEOUT="$2"; shift 2 ;;
    --grace)     GRACE="$2"; shift 2 ;;
    --sample)    SAMPLE="$2"; shift 2 ;;
    --venv)      VENV="$2"; shift 2 ;;
    --dataset)   DATASET="$2"; shift 2 ;;
    --workers)   WORKERS="$2"; shift 2 ;;
    --dry-run)   DRY=1; shift ;;
    --selftest)  SELFTEST=1; shift ;;
    *) echo "guard: unknown arg: $1" >&2; exit 3 ;;
  esac
done

die() { echo "guard: $*" >&2; exit 3; }

# --- progress probe: a token that STRICTLY GROWS while the eval does real work ---
# During the observed deadlock NONE of these move; a real (even slow) build grows at least
# one, so an unchanged token for GRACE seconds == deadlock.
_progress_token() {
  local imgs conts logb
  imgs=$(docker images -q 2>/dev/null | wc -l | tr -d ' ')
  conts=$(docker ps -q 2>/dev/null | wc -l | tr -d ' ')
  logb=$(find "$HOME/swe/logs/run_evaluation/$RUN_ID" -type f -printf '%s\n' 2>/dev/null | awk '{s+=$1} END{print s+0}')
  echo "${imgs}|${conts}|${logb}"
}

# --- watchdog: kill CHILD process group if PROBE output is unchanged for GRACE secs ---
# args: <child_pgid> <grace> <sample> <marker_path> <probe_fn_name>
# Relies on job control (set -m) so the child is its own process-group leader (pgid==pid).
_watchdog() {
  local pgid="$1" grace="$2" sample="$3" marker="$4" probe="$5"
  local last cur stalled=0
  last="$($probe)"
  while kill -0 "-$pgid" 2>/dev/null; do
    sleep "$sample"
    kill -0 "-$pgid" 2>/dev/null || break
    cur="$($probe)"
    if [ "$cur" = "$last" ]; then
      stalled=$(( stalled + sample ))
      if [ "$stalled" -ge "$grace" ]; then
        echo "guard: STALL -- no progress for ${stalled}s (probe frozen at: $cur) -- killing eval" >&2
        : > "$marker"
        kill -TERM "-$pgid" 2>/dev/null || true
        sleep 5
        kill -KILL "-$pgid" 2>/dev/null || true
        return 0
      fi
    else
      stalled=0; last="$cur"
    fi
  done
  return 0
}

# --- selftest: verify the watchdog DECISION LOGIC without docker ---
if [ "$SELFTEST" = "1" ]; then
  set +e
  set -m
  tmp="$(mktemp -d)"
  probe_file() { cat "$tmp/p" 2>/dev/null; }

  # Case A: STALLED child (probe never changes) -> watchdog must kill within ~grace.
  echo 0 > "$tmp/p"
  sleep 60 & child=$!
  t0=$(date +%s)
  _watchdog "$child" 6 2 "$tmp/markerA" probe_file
  wait "$child" 2>/dev/null
  dt=$(( $(date +%s) - t0 ))
  if [ ! -f "$tmp/markerA" ]; then echo "SELFTEST FAIL: stalled child was NOT flagged"; rm -rf "$tmp"; exit 1; fi
  if [ "$dt" -gt 15 ]; then echo "SELFTEST FAIL: stall kill too slow (${dt}s > 15)"; rm -rf "$tmp"; exit 1; fi
  echo "selftest A ok: stalled child killed in ${dt}s (<=15) and flagged"

  # Case B: PROGRESSING child (probe grows each second) -> watchdog must NOT kill.
  echo 0 > "$tmp/p"
  ( for i in $(seq 1 8); do echo "$i" > "$tmp/p"; sleep 1; done ) & writer=$!
  sleep 8 & child=$!
  _watchdog "$child" 4 1 "$tmp/markerB" probe_file
  wait "$child" 2>/dev/null
  kill "$writer" 2>/dev/null || true
  if [ -f "$tmp/markerB" ]; then echo "SELFTEST FAIL: progressing child was wrongly killed"; rm -rf "$tmp"; exit 1; fi
  echo "selftest B ok: progressing child NOT killed"

  rm -rf "$tmp"
  echo "SELFTEST PASS"
  exit 0
fi

[ -n "$PREDS" ]     || die "missing --preds"
[ -n "$RUN_ID" ]    || die "missing --run-id"
[ -n "$INSTANCES" ] || die "missing --instances"
[ -f "$PREDS" ]     || die "predictions file not found: $PREDS"

# predictions must be a non-empty JSON list
python3 - "$PREDS" <<'PY' || die "predictions file is not a non-empty JSON list"
import json,sys
d=json.load(open(sys.argv[1]))
assert isinstance(d,list) and len(d)>0, "not a non-empty list"
for e in d:
    assert e.get("instance_id") and e.get("model_patch"), "entry missing instance_id/model_patch"
PY

[ -x "$VENV/bin/python" ] || die "venv python not found: $VENV/bin/python"

IMG_COUNT=$(docker images -q 2>/dev/null | wc -l | tr -d ' ')
if [ "$IMG_COUNT" = "0" ]; then
  echo "guard: COLD docker cache (0 images) -- expect a long cold rebuild; timeout=${TIMEOUT}s grace=${GRACE}s"
else
  echo "guard: docker cache has ${IMG_COUNT} images"
fi

CMD=( timeout --signal=TERM --kill-after=30 "$TIMEOUT"
      "$VENV/bin/python" -m swebench.harness.run_evaluation
      --dataset_name "$DATASET"
      --predictions_path "$PREDS"
      --run_id "$RUN_ID"
      --instance_ids $INSTANCES
      --max_workers "$WORKERS" )

echo "guard: command:"
printf '  %q' "${CMD[@]}"; echo
if [ "$DRY" = "1" ]; then
  echo "guard: [dry-run] not executing."
  exit 0
fi

set +e
set -m                       # job control: the backgrounded eval gets its own process group
"${CMD[@]}" &
EVAL_PID=$!                  # == process-group leader (pgid) under set -m
MARKER="$(mktemp -u)"
echo "guard: early-stall watchdog armed (grace=${GRACE}s sample=${SAMPLE}s; hard timeout=${TIMEOUT}s backstop)"
_watchdog "$EVAL_PID" "$GRACE" "$SAMPLE" "$MARKER" _progress_token &
WD_PID=$!
wait "$EVAL_PID"; rc=$?
kill "$WD_PID" 2>/dev/null || true
wait "$WD_PID" 2>/dev/null
set +m
set -e

if [ -f "$MARKER" ]; then
  rm -f "$MARKER"
  echo "guard: EARLY DEADLOCK-STALL abort (no progress for >=${GRACE}s) -- cleaning up run_id containers"
  docker ps -a --format '{{.Names}}' 2>/dev/null | grep -F ".$RUN_ID" | xargs -r docker rm -f 2>/dev/null || true
  exit 2
fi

if [ $rc -eq 124 ] || [ $rc -eq 137 ]; then
  echo "guard: TIMEOUT/DEADLOCK after ${TIMEOUT}s (rc=$rc) -- cleaning up run_id containers"
  docker ps -a --format '{{.Names}}' 2>/dev/null | grep -F ".$RUN_ID" | xargs -r docker rm -f 2>/dev/null || true
  exit 2
fi
exit $rc
