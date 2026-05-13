import logging
from contextvars import ContextVar

from pythonjsonlogger.json import JsonFormatter

# One ContextVar per process — each async request gets its own isolated value.
# Default is empty string so logs emitted outside a request context don't error.
request_id_var: ContextVar[str] = ContextVar("request_id", default="")


class _RequestIdFilter(logging.Filter):
    """Injects the current request_id into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


def configure_logging(log_level: str = "INFO") -> None:
    """
    Configure the root logger with JSON formatting and request ID injection.

    Replaces any existing handlers — intended to be called once at application startup.
    Structured JSON output allows log aggregators (CloudWatch, ELK, Datadog) to index
    and filter on individual fields without custom parsers.

    Args:
        log_level: Root logging level, e.g. "INFO", "DEBUG", "WARNING".
    """
    handler = logging.StreamHandler()
    handler.setFormatter(
        JsonFormatter(
            fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%SZ",
        )
    )
    handler.addFilter(_RequestIdFilter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(log_level.upper())
