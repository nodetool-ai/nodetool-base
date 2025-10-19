import sys
from pathlib import Path
import asyncio
import gc
import threading

from nodetool.config.logging_config import configure_logging

configure_logging("DEBUG")

# Ensure local src is on path so tests import local package
SRC_PATH = Path(__file__).resolve().parents[1] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# Also add adjacent nodetool-core src if present (monorepo/dev setup)
CORE_SRC_PATH = Path(__file__).resolve().parents[2] / "nodetool-core" / "src"
if CORE_SRC_PATH.exists() and str(CORE_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(CORE_SRC_PATH))


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
