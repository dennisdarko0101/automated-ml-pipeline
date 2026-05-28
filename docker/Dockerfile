# Stage 1: Build
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# hatchling needs the readme and the package source present to build the
# project wheel, so copy them before installing.
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir --timeout=300 --retries=10 --upgrade pip \
    && pip install --no-cache-dir --timeout=300 --retries=10 .

# Stage 2: Runtime
FROM python:3.11-slim AS runtime

WORKDIR /app

# XGBoost and LightGBM load the OpenMP runtime (libgomp) at import time; the slim
# base image does not ship it, so install it in the runtime stage.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r appuser && useradd -r -g appuser appuser

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY src/ ./src/
COPY configs/ ./configs/
COPY sample_data/ ./sample_data/
COPY scripts/ ./scripts/

RUN mkdir -p /app/data /app/models && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
