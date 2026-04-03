import uvicorn
from src.config import ModelServingSettings

settings = ModelServingSettings()

if __name__ == "__main__":
    uvicorn.run(
        "src.api.app:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
    )
