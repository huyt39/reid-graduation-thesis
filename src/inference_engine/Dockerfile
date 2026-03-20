FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

RUN pip install --no-cache-dir uv
COPY pyproject.toml /app/
RUN uv pip install --system --no-cache .

COPY src /app/src
EXPOSE 8000
CMD ["python", "-m", "src"]
