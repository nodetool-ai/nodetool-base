import sys
from pathlib import Path
import asyncio
import gc
import threading

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
