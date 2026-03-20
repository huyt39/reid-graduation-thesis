import uvicorn
from src.api.app import app
from src.core.config import settings
from src.utils.logger import setup_logging

if __name__ == "__main__":
    setup_logging()
    uvicorn.run(app, host="0.0.0.0", port=settings.service_port)
