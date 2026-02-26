from typing import Generator, List, Optional
import os
import contextlib
import logging
import tempfile

logger = logging.getLogger(__name__)

def safe_unlink(path: str) -> None:
    """Safely unlink a file, suppressing FileNotFoundError."""
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.debug(f"Failed to unlink {path}: {e}")

@contextlib.contextmanager
def managed_temp_file(suffix: str = None, delete: bool = True) -> Generator[str, None, None]:
    """
    Context manager that creates a temporary file and ensures it is deleted upon exit.
    Yields the path to the temporary file.

    Args:
        suffix (str): The suffix for the temporary file name.
        delete (bool): Whether to delete the file on exit. Defaults to True.
    """
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
            temp_path = temp.name
        yield temp_path
    finally:
        if delete and temp_path:
            safe_unlink(temp_path)
