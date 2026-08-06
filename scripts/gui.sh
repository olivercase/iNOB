#!/usr/bin/env bash
#
# Start the iNOB GUI: API backend, web frontend, browser.
#
# The point of this script is that starting the GUI should be one action, not
# a checklist. It picks interpreters and ports itself, waits until each service
# actually answers before moving on, and cleans both up on exit — so closing
# the window leaves nothing running behind it.
#
#   scripts/gui.sh            start everything and open the browser
#   scripts/gui.sh --no-open  same, without opening a browser
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BACKEND_PORT="${INOB_BACKEND_PORT:-8000}"
WEB_PORT="${INOB_WEB_PORT:-}"
OPEN_BROWSER=1
[[ "${1:-}" == "--no-open" ]] && OPEN_BROWSER=0

say()  { printf "\033[38;5;42m▸\033[0m %s\n" "$*"; }
warn() { printf "\033[38;5;214m!\033[0m %s\n" "$*"; }
die()  { printf "\033[38;5;203m✗\033[0m %s\n" "$*" >&2; exit 1; }

port_free() { ! lsof -ti "tcp:$1" -sTCP:LISTEN >/dev/null 2>&1; }

# The backend needs fastapi + uvicorn. The *solve* needs duneuropy, which is
# usually a different interpreter — the backend picks that one per run itself
# (see _pipeline_python in gui/backend/app.py), so all we need here is a
# Python that can serve.
pick_backend_python() {
  local candidates=("${INOB_PYTHON:-}" "$ROOT/.venv/bin/python"
                    python3 python3.14 python3.13 python3.12 python3.11)
  for py in "${candidates[@]}"; do
    [[ -z "$py" ]] && continue
    if command -v "$py" >/dev/null 2>&1 &&
       "$py" -c "import fastapi, uvicorn" >/dev/null 2>&1; then
      command -v "$py"
      return 0
    fi
  done
  return 1
}

# ── backend ──────────────────────────────────────────────────────────────────

BACKEND_PID=""
WEB_PID=""

cleanup() {
  trap - EXIT INT TERM
  [[ -n "$WEB_PID" ]]     && kill "$WEB_PID" 2>/dev/null || true
  [[ -n "$BACKEND_PID" ]] && kill "$BACKEND_PID" 2>/dev/null || true
  wait 2>/dev/null || true
  say "stopped"
}
trap cleanup EXIT INT TERM

if port_free "$BACKEND_PORT"; then
  PY="$(pick_backend_python)" || die \
    "no Python with fastapi + uvicorn. Install them: python3 -m pip install fastapi uvicorn"
  say "backend  $("$PY" -V 2>&1) on :$BACKEND_PORT"
  # Agg keeps anything drawing in-process off the macOS GUI path; the pipeline
  # itself runs in a child process, which is where the real rendering happens.
  MPLBACKEND=Agg "$PY" -m uvicorn gui.backend.app:app \
    --host 127.0.0.1 --port "$BACKEND_PORT" \
    >"$ROOT/outputs/logs/gui-backend.log" 2>&1 &
  BACKEND_PID=$!
else
  say "backend  already running on :$BACKEND_PORT"
fi

for _ in $(seq 1 60); do
  curl -sf "http://127.0.0.1:$BACKEND_PORT/api/health" >/dev/null 2>&1 && break
  sleep 0.5
done
curl -sf "http://127.0.0.1:$BACKEND_PORT/api/health" >/dev/null 2>&1 ||
  die "backend did not come up — see outputs/logs/gui-backend.log"

# ── frontend ─────────────────────────────────────────────────────────────────

WEB_DIR="$ROOT/gui/web"
[[ -d "$WEB_DIR/node_modules" ]] || {
  say "installing web dependencies (first run only)"
  (cd "$WEB_DIR" && npm install --silent)
}

# Take 3000 if it's free, otherwise the next open port — someone else's dev
# server should never stop this one from starting.
if [[ -z "$WEB_PORT" ]]; then
  for candidate in 3000 3001 3002 3003; do
    if port_free "$candidate"; then WEB_PORT="$candidate"; break; fi
  done
  [[ -z "$WEB_PORT" ]] && die "ports 3000-3003 are all in use; set INOB_WEB_PORT"
fi

URL="http://localhost:$WEB_PORT"

if port_free "$WEB_PORT"; then
  say "frontend on :$WEB_PORT"
  (cd "$WEB_DIR" && INOB_BACKEND_HTTP="http://127.0.0.1:$BACKEND_PORT" \
     NEXT_PUBLIC_INOB_WS="ws://127.0.0.1:$BACKEND_PORT" \
     npx next dev -p "$WEB_PORT" >"$ROOT/outputs/logs/gui-web.log" 2>&1) &
  WEB_PID=$!
else
  say "frontend already running on :$WEB_PORT"
fi

for _ in $(seq 1 120); do
  curl -sf -o /dev/null "$URL" && break
  sleep 0.5
done
curl -sf -o /dev/null "$URL" || die "frontend did not come up — see outputs/logs/gui-web.log"

say "ready → $URL"
[[ "$OPEN_BROWSER" == 1 ]] && (command -v open >/dev/null && open "$URL" || true)
printf "\n  logs: outputs/logs/gui-backend.log · outputs/logs/gui-web.log\n"
printf "  press Ctrl-C to stop both\n\n"

wait
