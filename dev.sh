#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_PYTHON="${BACKEND_PYTHON:-python}"

cleanup() {
  local code=$?
  if [[ -n "${BACK_PID:-}" ]]; then kill "$BACK_PID" >/dev/null 2>&1 || true; fi
  if [[ -n "${FRONT_PID:-}" ]]; then kill "$FRONT_PID" >/dev/null 2>&1 || true; fi
  wait >/dev/null 2>&1 || true
  exit $code
}
trap cleanup INT TERM EXIT

echo "[dev] starting backend on http://127.0.0.1:8000"
(
  cd "$ROOT_DIR"
  "$BACKEND_PYTHON" server.py
) &
BACK_PID=$!

echo "[dev] starting frontend on http://127.0.0.1:5173"
(
  cd "$ROOT_DIR/frontend"
  npm run dev
) &
FRONT_PID=$!

wait -n "$BACK_PID" "$FRONT_PID"
