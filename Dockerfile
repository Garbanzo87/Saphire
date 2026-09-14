# Saphire API / worker image (CPU). Add `--build-arg EXTRAS="train,providers,mcp,postgres"` for the full stack.
FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md LICENSE ./
COPY saphire ./saphire
ARG EXTRAS="providers,mcp,postgres"
RUN pip install --upgrade pip && pip install ".[${EXTRAS}]"
RUN useradd -m saphire && mkdir -p /data /artifacts && chown -R saphire /data /artifacts
USER saphire
ENV SAPHIRE_DATABASE_URL=sqlite:////data/saphire.db SAPHIRE_ARTIFACTS_DIR=/artifacts
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s CMD curl -sf http://localhost:8000/health || exit 1
CMD ["saphire", "serve", "--host", "0.0.0.0", "--port", "8000"]
