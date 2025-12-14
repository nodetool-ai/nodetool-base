import sys
from pathlib import Path
import asyncio
import gc
import threading
from unittest.mock import MagicMock

# Mock chromadb to avoid import errors in environments where it's not installed
sys.modules["chromadb"] = MagicMock()
sys.modules["chromadb.config"] = MagicMock()
sys.modules["chromadb.utils"] = MagicMock()
sys.modules["chromadb.utils.embedding_functions"] = MagicMock()
sys.modules["chromadb.utils.embedding_functions.ollama_embedding_function"] = MagicMock()
sys.modules["chromadb.utils.embedding_functions.sentence_transformer_embedding_function"] = MagicMock()
sys.modules["langchain_core"] = MagicMock()
sys.modules["langchain_core.documents"] = MagicMock()
sys.modules["langchain_text_splitters"] = MagicMock()
sys.modules["pysqlite3"] = MagicMock()
sys.modules["huggingface_hub"] = MagicMock()
sys.modules["cryptography"] = MagicMock()
sys.modules["cryptography.fernet"] = MagicMock()
sys.modules["cryptography.hazmat"] = MagicMock()
sys.modules["cryptography.hazmat.primitives"] = MagicMock()
sys.modules["cryptography.hazmat.primitives.hashes"] = MagicMock()
sys.modules["cryptography.hazmat.primitives.kdf"] = MagicMock()
sys.modules["cryptography.hazmat.primitives.kdf.pbkdf2"] = MagicMock()
sys.modules["aiohttp"] = MagicMock()
sys.modules["pandas"] = MagicMock()
sys.modules["numpy"] = MagicMock()
sys.modules["PIL"] = MagicMock()
sys.modules["PIL.Image"] = MagicMock()
sys.modules["PIL.ImageOps"] = MagicMock()
sys.modules["PIL.ImageFilter"] = MagicMock()
sys.modules["pydantic_settings"] = MagicMock()
sys.modules["openai"] = MagicMock()
sys.modules["anthropic"] = MagicMock()
sys.modules["boto3"] = MagicMock()
sys.modules["botocore"] = MagicMock()
sys.modules["google"] = MagicMock()
sys.modules["google.generativeai"] = MagicMock()
sys.modules["keyring"] = MagicMock()
sys.modules["fastapi"] = MagicMock()
sys.modules["starlette"] = MagicMock()
sys.modules["uvicorn"] = MagicMock()
sys.modules["jinja2"] = MagicMock()
sys.modules["sqlalchemy"] = MagicMock()


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nodetool.config.logging_config import configure_logging

configure_logging("DEBUG")


def pytest_sessionfinish(session, exitstatus):
    """Clean up resources after all tests complete to prevent hanging."""
    import logging
    import os
    import time

    # Force garbage collection
    gc.collect()

    # Close any lingering event loops
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.stop()
        if not loop.is_closed():
            loop.close()
    except RuntimeError:
        pass  # No event loop in current thread

    # Log any non-daemon threads that might prevent exit
    main_thread = threading.main_thread()
    non_daemon_threads = [
        t for t in threading.enumerate()
        if t != main_thread and t.is_alive() and not t.daemon
    ]

    if non_daemon_threads:
        logging.warning(
            f"Found {len(non_daemon_threads)} non-daemon threads that may prevent exit: "
            f"{[t.name for t in non_daemon_threads]}"
        )
        # Force exit if there are hanging threads
        # Give threads a brief moment to clean up, then force exit
        def force_exit_thread():
            time.sleep(1)
            os._exit(exitstatus)

        exit_thread = threading.Thread(target=force_exit_thread, daemon=True)
        exit_thread.start()

    # Shutdown any thread pools or executors
    gc.collect()
