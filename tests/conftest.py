import sys
from pathlib import Path
import asyncio
import gc
import threading
from unittest.mock import MagicMock

import pytest_asyncio

# ---------------------------------------------------------------------------
# Compatibility shims for types that may be missing in older nodetool-core.
# pydantic_core and strenum are already transitive deps of nodetool-core,
# so they are always available in our test environment.
# ---------------------------------------------------------------------------
import builtins
from pydantic_core import PydanticUndefined  # noqa: E402

# Some node modules reference PydanticUndefined as a builtin at import time.
builtins.PydanticUndefined = PydanticUndefined  # type: ignore[attr-defined]

import nodetool.metadata.types as _nt_types  # noqa: E402

if not hasattr(_nt_types, "EmbeddingModel"):
    from pydantic import BaseModel as _BM

    class _EmbeddingModel(_BM):
        id: str = ""
        name: str = ""
        provider: _nt_types.Provider = _nt_types.Provider.Empty
        dimensions: int = 1536

    _nt_types.EmbeddingModel = _EmbeddingModel  # type: ignore[attr-defined]

if not hasattr(_nt_types.Provider, "Fake"):
    import strenum as _strenum  # already a nodetool-core dependency

    _nt_types.Provider = _strenum.StrEnum(  # type: ignore[misc]
        "Provider",
        {p.name: p.value for p in _nt_types.Provider} | {"Fake": "fake"},
    )


def _mock_module_if_missing(name: str) -> None:
    """
    Mock optional dependencies only when they're not importable.
    This prevents accidentally shadowing real modules that tests rely on (e.g. numpy, PIL).
    """

    if name in sys.modules:
        return

    try:
        __import__(name)
    except Exception:
        sys.modules[name] = MagicMock()


# Optional deps that can be absent in some environments
for module_name in [
    # Vector DB
    "chromadb",
    "chromadb.config",
    "chromadb.utils",
    "chromadb.utils.embedding_functions",
    "chromadb.utils.embedding_functions.ollama_embedding_function",
    "chromadb.utils.embedding_functions.sentence_transformer_embedding_function",
    # LangChain (some nodes/tests don't require it)
    "langchain_core",
    "langchain_core.documents",
    "langchain_text_splitters",
    # Optional SQLite shim
    "pysqlite3",
    # API SDKs (used conditionally)
    "openai",
    "anthropic",
    "mistralai",
    "boto3",
    "botocore",
    "botocore.exceptions",
    "keyring",
    "keyring.errors",
]:
    _mock_module_if_missing(module_name)


# Some integrations import `google.genai` (newer Gemini SDK); stub it if missing.
try:
    pass  # type: ignore[import-not-found]
except Exception:
    import types

    google_genai = types.ModuleType("google.genai")

    class Client:  # noqa: D401
        """Stub for google.genai.Client (tests don't exercise it)."""

        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "google.genai is not installed; this is a test stub. "
                "Install google-genai to use Gemini features."
            )

    google_genai.Client = Client  # type: ignore[attr-defined]
    sys.modules["google.genai"] = google_genai

# Older SDK path used by some codebases; stub if missing.
_mock_module_if_missing("google.generativeai")


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# Ensure a ResourceScope is bound for all tests.
# Many core APIs (assets, DB models, etc.) require an active scope.
@pytest_asyncio.fixture(autouse=True)
async def _resource_scope(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("DB_PATH", str(tmp_path / "nodetool-test.db"))

    # nodetool-core's Environment.get_db_path() intentionally raises under pytest.
    # For this repo's test suite we want an isolated sqlite DB per test.
    from nodetool.config.environment import Environment

    db_path = tmp_path / "nodetool-test.db"
    monkeypatch.setattr(
        Environment,
        "get_db_path",
        classmethod(lambda cls: str(db_path)),
    )

    # Ensure the schema exists for models used by the ProcessingContext (assets, etc.).
    from nodetool.models.migrations import run_startup_migrations

    await run_startup_migrations()

    from nodetool.runtime.resources import ResourceScope

    async with ResourceScope():
        yield

    # Clean up SQLite pools to avoid too many open files
    from nodetool.runtime.db_sqlite import shutdown_all_sqlite_pools

    await shutdown_all_sqlite_pools()


# The workflow Property builder in nodetool-core treats `default=None` as missing.
# For tests we allow None defaults; only truly required fields should fail.
def _patch_property_from_field() -> None:
    from nodetool.workflows.property import Property

    original = Property.from_field

    def from_field(name, type_, field):  # type: ignore[no-untyped-def]
        try:
            return original(name, type_, field)
        except ValueError as e:
            if str(e) != f"Field {name} has no default value":
                raise
            # Accept None defaults; required fields should still fail upstream.
            # Recreate Property using the same logic but without rejecting None.
            import annotated_types

            metadata = {type(f): f for f in field.metadata}
            ge = metadata.get(annotated_types.Ge, None)
            le = metadata.get(annotated_types.Le, None)
            title = (
                name.replace("_", " ").title() if field.title is None else field.title
            )
            return Property(
                name=name,
                type=type_,
                default=field.default,
                title=title,
                description=field.description,
                min=ge.ge if ge is not None else None,
                max=le.le if le is not None else None,
                json_schema_extra=field.json_schema_extra,  # type: ignore
            )

    Property.from_field = staticmethod(from_field)  # type: ignore[assignment]


_patch_property_from_field()


def _patch_graph_result() -> None:
    """
    nodetool-core's `graph_result()` currently calls the sync `run_graph_sync()`,
    which uses `asyncio.run()` and breaks under pytest-asyncio.
    """
    import nodetool.dsl.graph as dsl_graph

    async def graph_result(node, **kwargs):  # type: ignore[no-untyped-def]
        g = dsl_graph.graph(node)
        return await dsl_graph.run_graph_async(g, **kwargs)

    dsl_graph.graph_result = graph_result  # type: ignore[assignment]


_patch_graph_result()

# from nodetool.config.logging_config import configure_logging

# configure_logging("DEBUG")


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
        t
        for t in threading.enumerate()
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
