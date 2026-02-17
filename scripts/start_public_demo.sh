#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT_DIR/logs"
INSTANCE_NAME="${INSTANCE_NAME:-default}"
HOST="${HOST:-127.0.0.1}"
API_PORT="${API_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"
API_LOG=""
FE_LOG=""

usage() {
  cat <<EOF
Usage: $0 [--instance NAME] [--api-port PORT] [--frontend-port PORT] [--host HOST]

Examples:
  $0
  $0 --instance colleague --api-port 8001 --frontend-port 3001
  API_PORT=8002 FRONTEND_PORT=3002 INSTANCE_NAME=third $0
EOF
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --instance)
        INSTANCE_NAME="$2"
        shift 2
        ;;
      --api-port)
        API_PORT="$2"
        shift 2
        ;;
      --frontend-port)
        FRONTEND_PORT="$2"
        shift 2
        ;;
      --host)
        HOST="$2"
        shift 2
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        echo "Unknown argument: $1" >&2
        usage
        exit 1
        ;;
    esac
  done
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

require_cmd poetry
require_cmd npm
require_cmd cloudflared

is_valid_port() {
  [[ "$1" =~ ^[0-9]+$ ]] && (( "$1" >= 1 && "$1" <= 65535 ))
}

port_in_use() {
  local port="$1"
  if command -v lsof >/dev/null 2>&1; then
    lsof -iTCP:"$port" -sTCP:LISTEN -n -P >/dev/null 2>&1
    return $?
  fi
  if command -v ss >/dev/null 2>&1; then
    ss -ltn "( sport = :$port )" 2>/dev/null | grep -q LISTEN
    return $?
  fi
  return 1
}

parse_args "$@"

if ! is_valid_port "$API_PORT"; then
  echo "Invalid API port: $API_PORT" >&2
  exit 1
fi
if ! is_valid_port "$FRONTEND_PORT"; then
  echo "Invalid frontend port: $FRONTEND_PORT" >&2
  exit 1
fi
if [[ "$API_PORT" == "$FRONTEND_PORT" ]]; then
  echo "API and frontend ports must be different" >&2
  exit 1
fi
if port_in_use "$API_PORT"; then
  echo "Port $API_PORT is already in use (API)." >&2
  exit 1
fi
if port_in_use "$FRONTEND_PORT"; then
  echo "Port $FRONTEND_PORT is already in use (frontend)." >&2
  exit 1
fi

API_LOG="$LOG_DIR/public_demo_${INSTANCE_NAME}_api.log"
FE_LOG="$LOG_DIR/public_demo_${INSTANCE_NAME}_frontend.log"

mkdir -p "$LOG_DIR"
cd "$ROOT_DIR"

API_PID=""
FE_PID=""

cleanup() {
  echo
  echo "Stopping public demo processes..."
  if [[ -n "$FE_PID" ]] && kill -0 "$FE_PID" >/dev/null 2>&1; then
    kill "$FE_PID" >/dev/null 2>&1 || true
  fi
  if [[ -n "$API_PID" ]] && kill -0 "$API_PID" >/dev/null 2>&1; then
    kill "$API_PID" >/dev/null 2>&1 || true
  fi
}

trap cleanup EXIT INT TERM

echo "Starting instance: $INSTANCE_NAME"
echo "Backend:  http://$HOST:$API_PORT"
echo "Frontend: http://$HOST:$FRONTEND_PORT"
echo

echo "Starting API server on http://$HOST:$API_PORT ..."
DEBUG=false API_HOST="$HOST" API_PORT="$API_PORT" poetry run python src/api_server.py >"$API_LOG" 2>&1 &
API_PID=$!
sleep 3
if ! kill -0 "$API_PID" >/dev/null 2>&1; then
  echo "API startup failed. Last logs:"
  tail -n 50 "$API_LOG" || true
  exit 1
fi

echo "Starting frontend on http://$HOST:$FRONTEND_PORT ..."
(
  cd src/frontend
  VITE_API_PROXY_TARGET="http://$HOST:$API_PORT" npm run dev -- --host "$HOST" --port "$FRONTEND_PORT"
) >"$FE_LOG" 2>&1 &
FE_PID=$!
sleep 4
if ! kill -0 "$FE_PID" >/dev/null 2>&1; then
  echo "Frontend startup failed. Last logs:"
  tail -n 50 "$FE_LOG" || true
  exit 1
fi

echo "API log: $API_LOG"
echo "Frontend log: $FE_LOG"
echo "Opening Cloudflare tunnel..."
echo "Share the https://...trycloudflare.com URL shown below."
echo

TUNNEL_HOST_HEADER="${TUNNEL_HOST_HEADER:-127.0.0.1:$FRONTEND_PORT}"
cloudflared tunnel --url "http://$HOST:$FRONTEND_PORT" --http-host-header "$TUNNEL_HOST_HEADER"
