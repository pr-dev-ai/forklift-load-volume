#!/usr/bin/env bash
# Linux / macOS: creates a virtual environment, installs requirements, starts the app.
set -e
cd "$(dirname "$0")"
PY=python3
command -v python3 >/dev/null 2>&1 || PY=python
if [ ! -x .venv/bin/python ]; then
  echo "Creating virtual environment..."
  "$PY" -m venv .venv
fi
echo "Installing requirements..."
.venv/bin/python -m pip install --quiet --disable-pip-version-check -r requirements.txt
exec .venv/bin/python app.py "$@"
