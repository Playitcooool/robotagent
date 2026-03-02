#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_PYTHON="${BACKEND_PYTHON:-python}"

kill_pid_and_children() {
  local pid="$1"
  if [[ -z "$pid" ]]; then
    return 0
  fi
  if kill -0 "$pid" >/dev/null 2>&1; then
    pkill -TERM -P "$pid" >/dev/null 2>&1 || true
    kill -TERM "$pid" >/dev/null 2>&1 || true
  fi
}

kill_listeners_on_port() {
  local port="$1"
  local pids
  pids="$(lsof -ti tcp:"$port" -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -n "$pids" ]]; then
    echo "[dev] stopping existing process on :$port -> $pids"
    for pid in $pids; do
      kill_pid_and_children "$pid"
    done
    sleep 0.3
    # force kill if still alive
    for pid in $pids; do
      if kill -0 "$pid" >/dev/null 2>&1; then
        kill -KILL "$pid" >/dev/null 2>&1 || true
      fi
    done
  fi
}

cleanup() {
  local code=$?
  kill_pid_and_children "${BACK_PID:-}"
  kill_pid_and_children "${FRONT_PID:-}"
  wait >/dev/null 2>&1 || true
  exit "$code"
}
trap cleanup INT TERM EXIT

# Re-run safe: always clean old listeners before starting new ones.
kill_listeners_on_port 8000
kill_listeners_on_port 5173

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

# macOS default bash (3.2) has no `wait -n`, so poll both processes.
while true; do
  if ! kill -0 "$BACK_PID" >/dev/null 2>&1; then
    wait "$BACK_PID" || true
    exit 1
  fi
  if ! kill -0 "$FRONT_PID" >/dev/null 2>&1; then
    wait "$FRONT_PID" || true
    exit 1
  fi
  sleep 1
done
