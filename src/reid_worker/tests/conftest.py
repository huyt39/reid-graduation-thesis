import sys
from pathlib import Path
import types


# Allow `import src...` from `src/reid_worker/src` when running pytest directly.
# We need the parent of the `src/` package on `sys.path`.
_PKG_PARENT = Path(__file__).resolve().parents[1]
if str(_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_PKG_PARENT))


# Some unit tests import modules that depend on optional runtime deps (e.g. structlog).
# When running tests in a minimal environment, stub them so we can still run logic tests.
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


if "minio" not in sys.modules:
    try:
        import minio  # noqa: F401
    except Exception:
        class _DummyMinio:
            def __init__(self, *args, **kwargs):
                pass

            def bucket_exists(self, *args, **kwargs):
                return True

            def make_bucket(self, *args, **kwargs):
                return None

            def put_object(self, *args, **kwargs):
                return None

            def presigned_get_object(self, *args, **kwargs):
                return ""

        sys.modules["minio"] = types.SimpleNamespace(Minio=_DummyMinio)
