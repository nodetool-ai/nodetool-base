import logging
from typing import Literal


def configure_logging(level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"):
    """Configure a simple logger that never writes to a closed stream."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(numeric_level)
    root.addHandler(logging.NullHandler())


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a named logger."""
    return logging.getLogger(name)
