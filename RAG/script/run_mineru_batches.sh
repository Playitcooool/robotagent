#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PDF_ROOT="$BASE_DIR/arxiv_pdfs_filtered"
OUTPUT_ROOT="$BASE_DIR/output"
LOG_ROOT="$BASE_DIR/logs"

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"

if [ ! -d "$PDF_ROOT" ]; then
  echo "[ERROR] arxiv_pdfs_filtered directory not found"
  exit 1
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

for pdf in "$PDF_ROOT"/*.pdf; do
  if [ ! -f "$pdf" ]; then
    echo "[WARN] arxiv_pdfs_filtered has no pdf files"
    break
  fi

  pdf_name=$(basename "$pdf")
  stem="${pdf_name%.*}"
  log_file="$LOG_ROOT/${stem}.log"
  done_flag="$LOG_ROOT/${stem}.done"

  if [ -f "$done_flag" ]; then
    if grep -Eiq "traceback|error|exception|aborted|cannot import name" "$log_file" 2>/dev/null; then
      echo "[WARN] $pdf_name has stale done flag with error log, rerunning"
      rm -f "$done_flag"
    else
      echo "[SKIP] $pdf_name already processed"
      continue
    fi
  fi

  echo "===================================="
  echo "[RUN] Processing $pdf_name"
  echo "===================================="

  before_count=$(find "$OUTPUT_ROOT" -mindepth 1 | wc -l | tr -d ' ')
  mineru -p "$pdf" -o "$OUTPUT_ROOT" --backend pipeline > "$log_file" 2>&1

  status=$?
  after_count=$(find "$OUTPUT_ROOT" -mindepth 1 | wc -l | tr -d ' ')

  if [ $status -eq 0 ] && [ "$after_count" -gt "$before_count" ] && ! grep -Eiq "traceback|error|exception|aborted|cannot import name" "$log_file"; then
    touch "$done_flag"
    echo "[OK] $pdf_name finished"
  else
    echo "[FAIL] $pdf_name failed" | tee -a "$LOG_ROOT/mineru_failed.log"
  fi

  echo "[INFO] $pdf_name done, memory should be released"
  sleep 2
done
