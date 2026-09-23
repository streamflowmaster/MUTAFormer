#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_PYTHON="$PROJECT_DIR/.venv-panel/bin/python"
if [[ -x "$PROJECT_DIR/.venv-panel-20260922/bin/python" ]]; then
  DEFAULT_PYTHON="$PROJECT_DIR/.venv-panel-20260922/bin/python"
fi
PYTHON_BIN="${PANEL_PYTHON:-$DEFAULT_PYTHON}"
PID_FILE="$PROJECT_DIR/web/server.pid"
LOG_FILE="$PROJECT_DIR/web/server.log"

cd "$PROJECT_DIR"
if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "Panel server is already running with PID $(cat "$PID_FILE")"
  exit 0
fi

# Use one installed environment. Do not splice another Conda environment into it.
"$PYTHON_BIN" -c 'import torch, pandas, flask, sklearn; import sys; sys.path.insert(0,"web/vendor/python"); import captum; print("torch",torch.__version__,"captum",captum.__version__,"sklearn",sklearn.__version__)'
export PANEL_HOST="${PANEL_HOST:-0.0.0.0}"
export PANEL_PORT="${PANEL_PORT:-8000}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
nohup "$PYTHON_BIN" "$PROJECT_DIR/web/server.py" >"$LOG_FILE" 2>&1 &
echo $! >"$PID_FILE"
echo "Panel server started with PID $(cat "$PID_FILE")"
