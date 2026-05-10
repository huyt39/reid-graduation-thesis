FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

RUN pip install --no-cache-dir uv
COPY pyproject.toml /app/
COPY src /app/src
RUN uv pip install --system --no-cache .
EXPOSE 8765
CMD ["python", "-m", "src"]
