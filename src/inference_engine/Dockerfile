FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

RUN pip install --no-cache-dir uv
COPY pyproject.toml /app/
COPY src /app/src
RUN uv pip install --system --no-cache .

# Model weights are expected to be mounted or copied here
# COPY assets/models /app/src/assets/models

EXPOSE 8000
CMD ["python", "-m", "src"]
