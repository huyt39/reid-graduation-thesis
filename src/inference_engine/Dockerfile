FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

RUN pip install --no-cache-dir uv

# Copy deps first so pip install is cached as a separate layer.
# Using a BuildKit cache mount so uv's wheel cache persists across builds —
# torch (~750MB) is downloaded once and never re-fetched unless the version changes.
COPY pyproject.toml /app/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system .

COPY src /app/src

# Model weights are expected to be mounted or copied here
# COPY assets/models /app/src/assets/models

EXPOSE 8000
CMD ["python", "-m", "src"]
