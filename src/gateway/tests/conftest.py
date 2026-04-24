import sys
from pathlib import Path
import types


_PKG_PARENT = Path(__file__).resolve().parents[1]
if str(_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_PKG_PARENT))


if "structlog" not in sys.modules:
    try:
        import structlog  # noqa: F401
    except Exception:
        class _DummyLogger:
            def info(self, *args, **kwargs):
                pass

            def warning(self, *args, **kwargs):
                pass

            def error(self, *args, **kwargs):
                pass

        sys.modules["structlog"] = types.SimpleNamespace(get_logger=lambda *a, **k: _DummyLogger())


if "websockets" not in sys.modules:
    try:
        import websockets  # noqa: F401
    except Exception:
        class _DummyInvalidURI(Exception):
            pass

        class _DummyExceptions:
            InvalidURI = _DummyInvalidURI

        async def _dummy_connect(*args, **kwargs):
            raise _DummyInvalidURI()

        sys.modules["websockets"] = types.SimpleNamespace(
            connect=_dummy_connect,
            exceptions=_DummyExceptions,
        )
