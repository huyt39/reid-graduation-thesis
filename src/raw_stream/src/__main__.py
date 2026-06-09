import uvicorn

from src.config import settings

if __name__ == "__main__":
    uvicorn.run("src.main:app", host=settings.host, port=settings.port, log_level="info")
