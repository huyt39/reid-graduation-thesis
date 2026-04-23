import logging
try:
    import structlog
except ModuleNotFoundError:  # pragma: no cover
    structlog = None


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if structlog is None:
        return
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ],
    )


class Logger:
    def __init__(self, name: str = "reid_worker"):
        if structlog is not None:
            self._log = structlog.get_logger(name)
        else:
            self._log = logging.getLogger(name)

    def info(self, msg: str, extra: dict | None = None):
        if structlog is not None:
            self._log.info(msg, **(extra or {}))
        else:
            self._log.info("%s %s", msg, extra or {})

    def error(self, msg: str, exc_info: bool = False, extra: dict | None = None):
        if structlog is not None:
            self._log.error(msg, exc_info=exc_info, **(extra or {}))
        else:
            self._log.error("%s %s", msg, extra or {}, exc_info=exc_info)

    def warning(self, msg: str, extra: dict | None = None):
        if structlog is not None:
            self._log.warning(msg, **(extra or {}))
        else:
            self._log.warning("%s %s", msg, extra or {})

    def debug(self, msg: str, extra: dict | None = None):
        if structlog is not None:
            self._log.debug(msg, **(extra or {}))
        else:
            self._log.debug("%s %s", msg, extra or {})
