import time
import structlog
from src.core.config import settings

log = structlog.get_logger()

def run() -> None:
    log.info("worker_started", service=settings.service_name)
    while True:
        # TODO: consume -> process -> publish
        log.info("worker_tick", service=settings.service_name)
        time.sleep(settings.poll_interval_s)
