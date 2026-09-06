FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    cmake \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir \
    fastapi==0.115.12 \
    "uvicorn[standard]==0.35.0" \
    python-multipart==0.0.20 \
    slowapi==0.1.9 \
    sse-starlette==2.3.6 \
    aiofiles==25.1.0 \
    bleach==6.2.0

RUN python -m playwright install --with-deps chromium

COPY . .

CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-10000}"]