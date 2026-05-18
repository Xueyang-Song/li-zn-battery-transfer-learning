"""Structured logging configuration for the battery ML pipeline.

Provides JSON-formatted log records that include ``job_id``, ``step``, and
optionally ``cell_id`` in every message.  Use ``setup_logging`` once at
process start, then ``get_logger`` in every module.

Usage::

    from utils.logging_config import setup_logging, get_logger, log_step_start, log_step_end

    setup_logging(job_id="abc123")
    logger = get_logger(__name__)
    log_step_start(logger, "preprocess")
    ...
    log_step_end(logger, "preprocess", n_cells=132)
"""

from __future__ import annotations

import json
import logging
import sys
import time
from contextvars import ContextVar
from typing import Any

# ContextVars allow per-coroutine / per-thread context without global state.
_job_id_var: ContextVar[str] = ContextVar("job_id", default="")
_step_var: ContextVar[str] = ContextVar("step", default="")
_cell_id_var: ContextVar[str] = ContextVar("cell_id", default="")


def set_job_context(
    job_id: str = "",
    step: str = "",
    cell_id: str = "",
) -> None:
    """Inject context values that will appear in every subsequent log record.

    Args:
        job_id: Unique identifier for the current pipeline run.
        step: Current pipeline step name (e.g. ``"preprocess"``).
        cell_id: Optional cell identifier when processing a single cell.
    """
    if job_id:
        _job_id_var.set(job_id)
    if step:
        _step_var.set(step)
    if cell_id:
        _cell_id_var.set(cell_id)


class _JsonFormatter(logging.Formatter):
    """Emit log records as single-line JSON objects.

    Every record carries: timestamp, level, logger, job_id, step, cell_id,
    message.  Any ``extra`` dict passed to the logger call is merged in.
    """

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "job_id": _job_id_var.get(),
            "step": _step_var.get(),
            "cell_id": _cell_id_var.get(),
            "message": record.getMessage(),
        }

        # Merge any extra fields the caller supplied
        for key, value in record.__dict__.items():
            if key not in {
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "message",
                "taskName",
            }:
                try:
                    json.dumps(value)  # only include JSON-serialisable extras
                    payload[key] = value
                except (TypeError, ValueError):
                    payload[key] = str(value)

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


def setup_logging(
    job_id: str = "",
    level: int = logging.INFO,
    stream: Any = None,
) -> None:
    """Configure root logger with JSON formatting.

    Call this **once** at process startup before acquiring any logger.

    Args:
        job_id: Job identifier injected into every log record.
        level: Logging level (default: INFO).
        stream: Output stream (default: sys.stdout).
    """
    if job_id:
        _job_id_var.set(job_id)

    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(_JsonFormatter())

    root = logging.getLogger()
    # Remove any pre-existing handlers to avoid duplicate output.
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger inheriting root JSON formatting.

    Args:
        name: Logger name; use ``__name__`` from the calling module.

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    return logging.getLogger(name)


def log_step_start(
    logger: logging.Logger,
    step_name: str,
    **context: Any,
) -> float:
    """Log the start of a pipeline step and inject ``step`` into context.

    Args:
        logger: Logger to write to.
        step_name: Human-readable step name.
        **context: Additional key/value pairs to include in the log record.

    Returns:
        Wall-clock start time (seconds since epoch) for later duration calc.
    """
    _step_var.set(step_name)
    logger.info("step_start", extra={"step": step_name, **context})
    return time.time()


def log_step_end(
    logger: logging.Logger,
    step_name: str,
    start_time: float = 0.0,
    **context: Any,
) -> None:
    """Log the successful completion of a pipeline step.

    Args:
        logger: Logger to write to.
        step_name: Human-readable step name.
        start_time: Value returned by :func:`log_step_start` for duration.
        **context: Additional key/value pairs to include in the log record.
    """
    elapsed = round(time.time() - start_time, 3) if start_time else None
    extra: dict[str, Any] = {"step": step_name, **context}
    if elapsed is not None:
        extra["elapsed_s"] = elapsed
    logger.info("step_end", extra=extra)
    _step_var.set("")
