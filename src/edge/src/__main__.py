from src.utils.logger import setup_logging
from src.workers.main import run

if __name__ == "__main__":
    setup_logging()
    run()
