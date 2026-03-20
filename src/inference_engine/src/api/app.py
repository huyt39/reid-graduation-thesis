from fastapi import FastAPI
from src.core.config import settings

app = FastAPI(title=settings.service_name)

@app.get("/healthz")
def healthz():
    return {"status": "ok", "service": settings.service_name}

@app.get("/readyz")
def readyz():
    return {"status": "ready", "service": settings.service_name}
