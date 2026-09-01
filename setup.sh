#!/usr/bin/env bash
# FaceChain bootstrap script
# Usage: bash setup.sh
# Requires: Python 3.10+, Node 18+, npm
set -euo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO"

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║       FaceChain — Bootstrap Setup               ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# ---- Python env ----------------------------------------------------------
echo "[1/5] Installing Python dependencies..."
python -m pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
pip install \
  fastapi==0.115.12 \
  "uvicorn[standard]==0.35.0" \
  python-multipart==0.0.20 \
  slowapi==0.1.9 \
  sse-starlette==2.3.6 \
  aiofiles==25.1.0 \
  bleach==6.2.0 \
  --quiet
echo "    ✓ Python deps installed"

# ---- Playwright browser --------------------------------------------------
echo "[2/5] Installing Playwright Chromium browser..."
python -m playwright install chromium --quiet 2>/dev/null || \
  python -m playwright install chromium
echo "    ✓ Chromium installed"

# ---- Frontend ------------------------------------------------------------
echo "[3/5] Installing frontend dependencies..."
if [ -d "frontend" ]; then
  cd frontend
  npm install --silent
  cd "$REPO"
  echo "    ✓ npm packages installed"
else
  echo "    ✗ frontend/ directory not found — skipping"
fi

# ---- .env ----------------------------------------------------------------
echo "[4/5] Checking .env..."
if [ ! -f ".env" ]; then
  cp .env.example .env
  echo "    ✓ Created .env from .env.example"
  echo "    → Fill in PRIVATE_KEY (and API keys) before running the full pipeline"
else
  echo "    ✓ .env already exists"
fi

# ---- Self-check ----------------------------------------------------------
echo "[5/5] Running offline test suite..."
python -m pytest tests/ -q --tb=short 2>&1
echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  Setup complete — start with:                   ║"
echo "║                                                  ║"
echo "║  Backend:   uvicorn server:app --reload          ║"
echo "║  Frontend:  cd frontend && npm run dev           ║"
echo "║  UI:        http://localhost:5173                ║"
echo "║                                                  ║"
echo "║  CLI:       python pipeline.py --image samples/sundar_pichai.jpg --no-chain  ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""
