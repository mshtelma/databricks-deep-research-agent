"""Request/response logging middleware and configuration."""

import json
import logging
import os
import sys
import time
import uuid
from typing import Any

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger(__name__)


class JsonFormatter(logging.Formatter):
    """JSON log formatter for structured logging in production."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_obj: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add extra fields if present
        if hasattr(record, "request_id"):
            log_obj["request_id"] = record.request_id
        if hasattr(record, "session_id"):
            log_obj["session_id"] = record.session_id

        # Add exception info if present
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_obj)


class ColoredFormatter(logging.Formatter):
    """Colored formatter for development console output."""

    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Format with colors for terminal output."""
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname:8}{self.RESET}"
        return super().format(record)


class RequestLoggingMiddleware:
    """Pure ASGI middleware for request logging.

    Unlike BaseHTTPMiddleware, this does NOT wrap requests in task scopes
    that get cancelled on client disconnect. This prevents CancelledError
    from propagating to database connections during SSE streaming.

    Key differences from BaseHTTPMiddleware:
    - No task wrapping - requests flow directly through
    - Client disconnects don't trigger task cancellation cascade
    - Streaming responses flow through uninterrupted
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Process ASGI request."""
        # Pass through non-HTTP requests unchanged (lifespan, websocket)
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Generate request ID for tracing
        request_id = str(uuid.uuid4())[:8]

        # Initialize state dict if not present and add request_id
        # FastAPI's Request object uses scope["state"] internally,
        # so request.state.request_id will work in route handlers
        if "state" not in scope:
            scope["state"] = {}
        scope["state"]["request_id"] = request_id

        start_time = time.perf_counter()

        async def send_wrapper(message: Message) -> None:
            """Intercept response start to inject X-Request-ID header."""
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers.append("X-Request-ID", request_id)
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                "Request failed",
                extra={
                    "request_id": request_id,
                    "method": scope.get("method", ""),
                    "path": scope.get("path", ""),
                    "error": str(e),
                    "duration_ms": round(duration_ms, 2),
                },
                exc_info=True,
            )
            raise


def setup_logging(log_level: str = "INFO", log_format: str = "text") -> None:
    """Configure application logging.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_format: Format type ("text" for development, "json" for production)
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))

    # Choose formatter based on format type
    if log_format.lower() == "json":
        formatter = JsonFormatter(datefmt="%Y-%m-%dT%H:%M:%S")
    elif sys.stdout.isatty() and os.environ.get("TERM"):
        # Use colored output for interactive terminals
        formatter = ColoredFormatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    else:
        # Plain text for non-interactive
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Reduce noise from third-party libraries
    third_party_loggers = [
        "uvicorn.access",
        "uvicorn.error",
        "httpx",
        "httpcore",
        "asyncio",
        "openai",
        "openai._base_client",
        "urllib3",
        "faker",
        "databricks",
        "databricks.sdk",
        "git",
        "git.cmd",
        "mlflow",
    ]
    for logger_name in third_party_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    # But keep our own loggers at the specified level
    logging.getLogger("backend").setLevel(getattr(logging, log_level.upper()))
    logging.getLogger("src").setLevel(getattr(logging, log_level.upper()))

    logger.info(
        f"Logging configured | level={log_level} | format={log_format}"
    )
