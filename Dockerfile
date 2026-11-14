# ---------- Stage 1: builder (install packages into .venv) ----------
FROM python:3.12-slim AS builder

# Copy the uv binary from its official image (pinned version).
COPY --from=ghcr.io/astral-sh/uv:0.11.12 /uv /bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# 1) Dependencies only: this layer is cached until pyproject.toml/uv.lock change.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-group train --no-install-project

# 2) Our code: installed as a normal package (not editable), so src/ isn't needed later.
COPY src ./src
RUN uv sync --locked --no-dev --no-group train --no-editable

# ---------- Stage 2: runtime (only what the API needs) ----------
FROM python:3.12-slim

# Run as a normal user, not root.
RUN useradd --create-home app
WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY models/model.joblib /app/models/model.joblib

ENV PATH="/app/.venv/bin:$PATH" \
    MODEL_PATH=/app/models/model.joblib \
    PYTHONUNBUFFERED=1

USER app
EXPOSE 8000

# --host 0.0.0.0: accept requests from outside the container.
CMD ["uvicorn", "churn.api:app", "--host", "0.0.0.0", "--port", "8000"]
