#!/usr/bin/env bash
# docker_eval_guard.sh -- safe wrapper around the SWE-bench Docker eval for overnight audits.
#
# WHY: the AUTHORITATIVE SWE-bench Docker eval (swebench.harness.run_evaluation) can
# DEADLOCK -- observed 2026-07-15: all Python threads parked in futex_do_wait, ~1% CPU,
# 0 docker images / 0 build cache, stale CLOSE-WAIT sockets to HuggingFace -- while
# emitting NO progress. On a COLD docker cache (image store pruned to 0) a cold rebuild is
# also very slow. An unguarded eval can silently consume an entire overnight cycle.
#
# This wrapper makes a hang FAIL FAST and VISIBLY:
#   * validates the predictions file (exists, non-empty JSON list)
#   * preflights the docker image cache and warns when it is cold
#   * enforces a hard timeout (SIGTERM then SIGKILL)
#   * on timeout, force-removes any leftover containers for this run_id
#   * distinguishes a timeout/deadlock (exit 2) from a genuine eval error (exit 1)
#
# ANSWER-LEAKAGE: this only runs the authoritative scorer on the model's OWN prediction.
# It writes no results and injects nothing into any instance. General/scoring-layer only.
#
# Usage:
#   docker_eval_guard.sh --preds PATH --run-id ID --instances "id1 id2 ..." \
#                        [--timeout SECS] [--venv PATH] [--dataset NAME] [--workers N] \
#                        [--dry-run]
# Defaults: timeout=1800  venv=~/swebench-venv  dataset=SWE-bench/SWE-bench_Lite  workers=1
set -euo pipefail

PREDS="" RUN_ID="" INSTANCES="" TIMEOUT=1800
VENV="$HOME/swebench-venv" DATASET="SWE-bench/SWE-bench_Lite" WORKERS=1 DRY=0

while [ $# -gt 0 ]; do
  case "$1" in
    --preds)     PREDS="$2"; shift 2 ;;
    --run-id)    RUN_ID="$2"; shift 2 ;;
    --instances) INSTANCES="$2"; shift 2 ;;
    --timeout)   TIMEOUT="$2"; shift 2 ;;
    --venv)      VENV="$2"; shift 2 ;;
    --dataset)   DATASET="$2"; shift 2 ;;
    --workers)   WORKERS="$2"; shift 2 ;;
    --dry-run)   DRY=1; shift ;;
    *) echo "guard: unknown arg: $1" >&2; exit 3 ;;
  esac
done

die() { echo "guard: $*" >&2; exit 3; }
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
  echo "guard: COLD docker cache (0 images) -- expect a long cold rebuild; timeout=${TIMEOUT}s"
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
"${CMD[@]}"
rc=$?
set -e

if [ $rc -eq 124 ] || [ $rc -eq 137 ]; then
  echo "guard: TIMEOUT/DEADLOCK after ${TIMEOUT}s (rc=$rc) -- cleaning up run_id containers"
  docker ps -a --format '{{.Names}}' 2>/dev/null | grep -F ".$RUN_ID" | xargs -r docker rm -f 2>/dev/null || true
  exit 2
fi
exit $rc
