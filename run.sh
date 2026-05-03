#!/bin/bash
set -e
echo "========================================"
echo "  VetAI — Flask ML API (backend/)      "
echo "========================================"
echo ""

if ! command -v python3 &> /dev/null; then
    echo "Python 3 is required."
    exit 1
fi

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT/backend"

if [ ! -d "$ROOT/venv" ]; then
    echo "Creating venv at repo root..."
    python3 -m venv "$ROOT/venv"
fi

# shellcheck source=/dev/null
source "$ROOT/venv/bin/activate" 2>/dev/null || source "$ROOT/venv/Scripts/activate"

echo "Installing backend dependencies..."
pip install -r requirements.txt -q

if [ "${1:-}" = "train" ]; then
    echo "Training models..."
    python train_models.py
fi

echo ""
echo "Starting API on http://0.0.0.0:5000"
echo "Press Ctrl+C to stop"
exec python ml_server.py
