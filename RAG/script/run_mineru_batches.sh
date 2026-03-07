#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BATCH_ROOT="$BASE_DIR/batches"
OUTPUT_ROOT="$BASE_DIR/output"
LOG_ROOT="$BASE_DIR/logs"

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"

if [ ! -d "$BATCH_ROOT" ]; then
  echo "[ERROR] batches directory not found"
  exit 1
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

for batch in $(find "$BATCH_ROOT" -maxdepth 1 -type d -name "batch_*" | sort); do
  batch_name=$(basename "$batch")
  log_file="$LOG_ROOT/${batch_name}.log"
  done_flag="$LOG_ROOT/${batch_name}.done"

  if [ -f "$done_flag" ]; then
    if grep -Eiq "traceback|error|exception|aborted|cannot import name" "$log_file" 2>/dev/null; then
      echo "[WARN] $batch_name has stale done flag with error log, rerunning"
      rm -f "$done_flag"
    else
      echo "[SKIP] $batch_name already processed"
      continue
    fi
  fi

  echo "===================================="
  echo "[RUN] Processing $batch_name"
  echo "===================================="

  before_count=$(find "$OUTPUT_ROOT" -mindepth 1 | wc -l | tr -d ' ')
  mineru -p "$batch" -o "$OUTPUT_ROOT" --backend pipeline > "$log_file" 2>&1

  status=$?
  after_count=$(find "$OUTPUT_ROOT" -mindepth 1 | wc -l | tr -d ' ')

  if [ $status -eq 0 ] && [ "$after_count" -gt "$before_count" ] && ! grep -Eiq "traceback|error|exception|aborted|cannot import name" "$log_file"; then
    touch "$done_flag"
    echo "[OK] $batch_name finished"
  else
    echo "[FAIL] $batch_name failed" | tee -a "$LOG_ROOT/mineru_failed.log"
  fi

  echo "[INFO] Batch $batch_name done, memory should be released"
  sleep 2
done
