# FaceChain — development shortcuts
# Requires: Python 3.10+, Node 18+, npm

.PHONY: help install backend ui dev test test-backend test-frontend lint setup

help:
	@echo ""
	@echo "FaceChain Makefile targets:"
	@echo "  make setup          Bootstrap everything (venv + pip + npm + .env)"
	@echo "  make install        Install Python + frontend deps (no .env copy)"
	@echo "  make backend        Start FastAPI server on :8000 (reload)"
	@echo "  make ui             Start Vite dev server on :5173"
	@echo "  make dev            Print instructions to run both (use two terminals)"
	@echo "  make test           Run all tests (backend + frontend)"
	@echo "  make test-backend   Run Python test suite"
	@echo "  make test-frontend  Run Vitest suite"
	@echo "  make lint           Run ruff + eslint"
	@echo ""

setup:
	@echo "==> Setting up FaceChain..."
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	pip install fastapi==0.115.12 uvicorn[standard]==0.35.0 python-multipart==0.0.20 \
	            slowapi==0.1.9 sse-starlette==2.3.6 aiofiles==25.1.0 bleach==6.2.0
	python -m playwright install chromium
	@echo "==> Installing frontend deps..."
	cd frontend && npm install
	@if not exist .env ( copy .env.example .env && echo "==> Created .env from .env.example — fill in keys" ) else ( echo "==> .env already exists" )
	@echo "==> Running offline self-check..."
	python -m pytest tests/ -q --tb=short
	@echo ""
	@echo "==> Setup complete."
	@echo "    Backend:  uvicorn server:app --reload"
	@echo "    Frontend: cd frontend && npm run dev"

install:
	pip install -r requirements.txt
	pip install fastapi==0.115.12 uvicorn[standard]==0.35.0 python-multipart==0.0.20 \
	            slowapi==0.1.9 sse-starlette==2.3.6 aiofiles==25.1.0 bleach==6.2.0
	cd frontend && npm install

backend:
	uvicorn server:app --reload --host 0.0.0.0 --port 8000

ui:
	cd frontend && npm run dev

dev:
	@echo ""
	@echo "Run these in two separate terminals:"
	@echo ""
	@echo "  Terminal 1 (backend):"
	@echo "    uvicorn server:app --reload --host 0.0.0.0 --port 8000"
	@echo ""
	@echo "  Terminal 2 (frontend):"
	@echo "    cd frontend && npm run dev"
	@echo ""
	@echo "  Then open: http://localhost:5173"
	@echo ""

test: test-backend test-frontend

test-backend:
	python -m pytest tests/ -q --tb=short

test-frontend:
	cd frontend && npm run test

lint:
	-python -m ruff check src/ tests/ server.py pipeline.py scripts/ 2>&1 || true
	-cd frontend && npm run lint 2>&1 || true
