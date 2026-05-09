"""
Structured JSON logging via structlog.
Every log record includes: timestamp, level, service, request_id.
The request_id threads through all services for end-to-end tracing.
"""

import logging
import structlog
from typing import Optional


def configure_logging(service_name: str, level: str = "INFO") -> None:
    """Configure structlog with JSON output. Call once at service startup."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.stdlib.add_logger_name,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )
    structlog.contextvars.bind_contextvars(service=service_name)


def get_logger(name: str = __name__):
    return structlog.get_logger(name)


def bind_request_id(request_id: str) -> None:
    """Bind request_id to context so it appears in all subsequent log calls."""
    structlog.contextvars.bind_contextvars(request_id=request_id)


def clear_request_context() -> None:
    structlog.contextvars.clear_contextvars()
