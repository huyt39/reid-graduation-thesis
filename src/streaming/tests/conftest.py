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

