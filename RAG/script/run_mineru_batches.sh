#!/usr/bin/env bash
set -e

BATCH_ROOT="batches"
OUTPUT_ROOT="output"
LOG_ROOT="logs"

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
    echo "[SKIP] $batch_name already processed"
    continue
  fi

  echo "===================================="
  echo "[RUN] Processing $batch_name"
  echo "===================================="

  (
    mineru -p "$batch" -o "$OUTPUT_ROOT" --backend pipeline > "$log_file" 2>&1
  )

  status=$?

  if [ $status -eq 0 ]; then
    touch "$done_flag"
    echo "[OK] $batch_name finished"
  else
    echo "[FAIL] $batch_name failed" | tee -a "$LOG_ROOT/mineru_failed.log"
  fi

  echo "[INFO] Batch $batch_name done, memory should be released"
  sleep 2
done
