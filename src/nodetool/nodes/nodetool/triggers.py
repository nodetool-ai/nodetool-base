"""
Trigger Nodes for Infinite Running Workflows
=============================================

This module provides trigger nodes that enable workflows to run indefinitely,
waiting for external events such as webhooks, file system changes, scheduled
intervals, or manual inputs.

Trigger nodes are special streaming nodes that:
1. Block until an external event occurs
2. Emit the event data as output
3. Loop back to wait for the next event
4. Only terminate when explicitly stopped
"""

from __future__ import annotations

import asyncio
import json
from abc import abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, ClassVar, Generic, Optional, TypedDict, TypeVar

from pydantic import Field

from nodetool.config.logging_config import get_logger
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext

log = get_logger(__name__)

# Minimum wait time in seconds to prevent tight loops when drift compensation
# causes wait_time to be near zero. This ensures the event loop can process
# other tasks and prevents CPU spinning.
MIN_WAIT_SECONDS = 0.001

T = TypeVar("T")


class FileWatchTriggerOutput(TypedDict):
    event: str
    path: str
    dest_path: str | None
    is_directory: bool
    timestamp: str


class IntervalTriggerOutput(TypedDict):
    tick: int
    elapsed_seconds: float
    interval_seconds: float
    timestamp: str
    source: str
    event_type: str


class ManualTriggerOutput(TypedDict):
    data: Any
    timestamp: str
    source: str
    event_type: str


class WebhookTriggerOutput(TypedDict):
    body: Any
    headers: dict[str, Any]
    query: dict[str, Any]
    method: str
    path: str
    timestamp: str
    source: str
    event_type: str


class TriggerNode(BaseNode, Generic[T]):
    """
    Base class for trigger nodes that enable infinite-running workflows.

    Trigger nodes are special streaming nodes that:
    1. Wait for external events (webhooks, file changes, timers, etc.)
    2. Emit event data when triggered
    3. Loop back to wait for the next event
    4. Only terminate when the workflow is explicitly stopped

    Subclasses must implement:
    - setup_trigger(): Initialize the event source
    - wait_for_event(): Block until an event occurs and return event data
    - cleanup_trigger(): Clean up the event source

    Attributes:
        _is_running: Flag to control the trigger loop
        _event_queue: Queue for receiving events from external sources
    """

    # Mark this as a streaming output node
    _layout: ClassVar[str] = "default"

    # Configuration
    max_events: int = Field(
        default=0,
        description="Maximum number of events to process (0 = unlimited)",
        ge=0,
    )

    def __init__(self, **data: Any):
        super().__init__(**data)
        self._is_running = False
        self._event_queue: asyncio.Queue[T | None] = asyncio.Queue()
        self._setup_complete = False
        self._loop: asyncio.AbstractEventLoop | None = None

    @classmethod
    def is_streaming_output(cls) -> bool:
        """Trigger nodes always produce streaming output."""
        return True

    @classmethod
    def is_cacheable(cls) -> bool:
        """Trigger nodes should never be cached."""
        return False

    @abstractmethod
    async def setup_trigger(self, context: ProcessingContext) -> None:
        """
        Initialize the trigger's event source.

        This method is called once when the workflow starts. Subclasses should
        set up any resources needed to receive events (start servers, register
        watchers, etc.).

        Args:
            context: The processing context for the workflow.
        """
        pass

    @abstractmethod
    async def wait_for_event(self, context: ProcessingContext) -> Optional[T]:
        """
        Wait for and return the next event.

        This method should block until an event is available or the trigger
        is stopped. Return None to signal that the trigger should stop.

        Args:
            context: The processing context for the workflow.

        Returns:
            The event data, or None to stop the trigger.
        """
        pass

    @abstractmethod
    async def cleanup_trigger(self, context: ProcessingContext) -> None:
        """
        Clean up trigger resources.

        This method is called when the workflow is stopping. Subclasses should
        release any resources acquired in setup_trigger().

        Args:
            context: The processing context for the workflow.
        """
        pass

    def stop(self) -> None:
        """Signal the trigger to stop processing events."""
        log.info(f"Stopping trigger {self.get_title()} ({self._id})")
        self._is_running = False
        # Push None to unblock wait_for_event if it's using the queue
        try:
            self._event_queue.put_nowait(None)
        except Exception:
            pass

    async def initialize(self, context: ProcessingContext, skip_cache: bool = False):
        """Initialize the trigger when the workflow starts."""
        await super().initialize(context, skip_cache)
        self._is_running = True
        self._setup_complete = False
        # Reset the event queue to avoid stale stop signals from prior runs.
        self._event_queue = asyncio.Queue()
        # Capture the event loop for thread-safe event pushing
        self._loop = asyncio.get_running_loop()

    async def finalize(self, context: ProcessingContext):
        """Clean up when the workflow is stopping."""
        log.info(f"Finalizing trigger {self.get_title()} ({self._id})")
        self.stop()
        if self._setup_complete:
            try:
                await self.cleanup_trigger(context)
            except Exception as e:
                log.error(f"Error cleaning up trigger: {e}")
        await super().finalize(context)

    async def gen_process(self, context: ProcessingContext) -> AsyncGenerator[T, None]:
        """
        Main processing loop for the trigger.

        This method:
        1. Sets up the trigger
        2. Loops waiting for events
        3. Emits each event as it arrives
        4. Continues until stopped or max_events reached
        """
        log.info(f"Starting trigger {self.get_title()} ({self._id})")

        # Re-enable processing in case stop() was called before gen_process started
        # but there are still events in the queue to process
        self._is_running = True

        # Set up the trigger
        try:
            await self.setup_trigger(context)
            self._setup_complete = True
        except Exception as e:
            log.error(f"Failed to set up trigger: {e}")
            raise

        events_processed = 0

        while True:
            try:
                # Wait for the next event
                event = await self.wait_for_event(context)

                log.info(f"Trigger {self.get_title()} received event: {event}")

                # None signals to stop
                if event is None:
                    log.info(f"Trigger {self.get_title()} received stop signal")
                    break

                # Emit the event
                if isinstance(event, dict):
                    event_type_value = (
                        event.get("event_type") or event.get("event") or "unknown"
                    )
                    event_type = str(event_type_value)
                else:
                    event_type = getattr(event, "event_type", "unknown")
                log.debug(f"Trigger {self.get_title()} emitting event: {event_type}")

                yield event

                events_processed += 1

                # Check max events
                if self.max_events > 0 and events_processed >= self.max_events:
                    log.info(
                        f"Trigger {self.get_title()} reached max events: {self.max_events}"
                    )
                    break

            except asyncio.CancelledError:
                log.info(f"Trigger {self.get_title()} was cancelled")
                break
            except Exception as e:
                log.error(f"Error in trigger loop: {e}")
                # Continue processing unless it's a fatal error
                if not self._is_running:
                    break

        log.info(f"Trigger {self.get_title()} finished after {events_processed} events")

    def push_event(self, event: T) -> None:
        """
        Push an event to the trigger's queue.

        This method is thread-safe and can be called from external sources
        (HTTP handlers, file watchers, etc.) to deliver events to the trigger.

        Args:
            event: The event to push.
        """
        try:
            # Check if we're calling from a different thread than the event loop
            if self._loop is not None:
                try:
                    running_loop = asyncio.get_running_loop()
                    # Same thread, same loop - use put_nowait directly
                    if running_loop is self._loop:
                        self._event_queue.put_nowait(event)
                        log.debug("Event pushed directly to queue (same loop)")
                        return
                except RuntimeError:
                    # No running loop - check if we're in a different thread
                    log.debug("No running loop in push_event, trying thread-safe method")
                    pass

                # Different thread or no running loop - use thread-safe method
                if self._loop.is_running():
                    self._loop.call_soon_threadsafe(self._event_queue.put_nowait, event)
                    log.debug("Event pushed via call_soon_threadsafe")
                    return

            # Fallback: direct put (e.g., loop not yet started)
            log.debug(f"Event pushed via fallback (loop={self._loop})")
            self._event_queue.put_nowait(event)
        except Exception as e:
            log.error(f"Failed to push event: {e}")

    async def get_event_from_queue(self, timeout: float | None = None) -> T | None:
        """
        Wait for and retrieve an event from the queue.

        This is a helper method for subclasses that use the event queue.

        Args:
            timeout: Maximum time to wait in seconds, or None for no timeout.

        Returns:
            The event, or None if stopped or timeout reached.
        """
        try:
            if timeout is not None:
                return await asyncio.wait_for(self._event_queue.get(), timeout=timeout)
            else:
                return await self._event_queue.get()
        except asyncio.TimeoutError:
            return None
        except asyncio.CancelledError:
            return None


class FileWatchTrigger(TriggerNode[FileWatchTriggerOutput]):
    """
    Trigger node that monitors filesystem changes.

    This trigger uses the watchdog library to monitor a directory or file
    for changes. When a change is detected, an event is emitted containing:
    - The path of the changed file
    - The type of change (created, modified, deleted, moved)
    - Timestamp of the event

    This trigger is useful for:
    - Processing files as they arrive in a directory
    - Triggering workflows on configuration changes
    - Building file-based automation pipelines
    """

    path: str = Field(
        default=".",
        description="Path to watch (file or directory)",
    )
    recursive: bool = Field(
        default=False,
        description="Watch subdirectories recursively",
    )
    patterns: list[str] = Field(
        default=["*"],
        description="File patterns to watch (e.g., ['*.txt', '*.json'])",
    )
    ignore_patterns: list[str] = Field(
        default=[],
        description="File patterns to ignore",
    )
    events: list[str] = Field(
        default=["created", "modified", "deleted", "moved"],
        description="Types of events to watch for",
    )
    debounce_seconds: float = Field(
        default=0.5,
        description="Debounce time to avoid duplicate events",
        ge=0,
    )

    OutputType: ClassVar[type] = FileWatchTriggerOutput

    def __init__(self, **data: Any):
        super().__init__(**data)
        self._observer = None
        self._last_events: dict[str, float] = {}

    async def setup_trigger(self, context: ProcessingContext) -> None:
        """Start the filesystem watcher."""
        try:
            import fnmatch

            from watchdog.events import (
                DirCreatedEvent,
                DirDeletedEvent,
                DirModifiedEvent,
                DirMovedEvent,
                FileCreatedEvent,
                FileDeletedEvent,
                FileModifiedEvent,
                FileMovedEvent,
                FileSystemEventHandler,
            )
            from watchdog.observers import Observer
        except ImportError:
            raise ImportError(
                "watchdog is required for FileWatchTrigger. "
                "Install it with: pip install watchdog"
            )

        watch_path = Path(self.path).expanduser().resolve()
        if not watch_path.exists():
            raise ValueError(f"Watch path does not exist: {watch_path}")

        if self._observer is not None:
            if self._observer.is_alive():
                log.info(f"File watcher already running on {watch_path}")
                return
            try:
                self._observer.stop()
                self._observer.join(timeout=2.0)
            except Exception as e:
                log.warning(f"Error stopping existing observer: {e}")
            finally:
                self._observer = None

        log.info(f"Setting up file watch trigger on {watch_path}")

        trigger = self

        class EventHandler(FileSystemEventHandler):
            """Handler for filesystem events."""

            def _should_process(self, path: str) -> bool:
                """Check if the path matches the configured patterns."""
                name = Path(path).name

                # Check ignore patterns
                for pattern in trigger.ignore_patterns:
                    if fnmatch.fnmatch(name, pattern):
                        return False

                # Check include patterns
                for pattern in trigger.patterns:
                    if fnmatch.fnmatch(name, pattern):
                        return True

                return False

            def _debounce(self, path: str) -> bool:
                """Check if event should be debounced."""
                now = datetime.now(timezone.utc).timestamp()
                last = trigger._last_events.get(path, 0)

                if now - last < trigger.debounce_seconds:
                    return True

                trigger._last_events[path] = now
                return False

            def _emit_event(
                self,
                event_type: str,
                src_path: str,
                dest_path: str | None = None,
                is_directory: bool = False,
            ):
                """Emit a filesystem event."""
                if event_type not in trigger.events:
                    return

                if not self._should_process(src_path):
                    return

                if self._debounce(src_path):
                    return

                event = {
                    "event": event_type,
                    "path": src_path,
                    "dest_path": dest_path,
                    "is_directory": is_directory,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

                log.debug(f"File event: {event_type} - {src_path}")
                trigger.push_event(event)

            def on_created(self, event):
                if isinstance(event, (FileCreatedEvent, DirCreatedEvent)):
                    self._emit_event(
                        "created",
                        event.src_path,
                        is_directory=isinstance(event, DirCreatedEvent),
                    )

            def on_modified(self, event):
                if isinstance(event, (FileModifiedEvent, DirModifiedEvent)):
                    self._emit_event(
                        "modified",
                        event.src_path,
                        is_directory=isinstance(event, DirModifiedEvent),
                    )

            def on_deleted(self, event):
                if isinstance(event, (FileDeletedEvent, DirDeletedEvent)):
                    self._emit_event(
                        "deleted",
                        event.src_path,
                        is_directory=isinstance(event, DirDeletedEvent),
                    )

            def on_moved(self, event):
                if isinstance(event, (FileMovedEvent, DirMovedEvent)):
                    self._emit_event(
                        "moved",
                        event.src_path,
                        event.dest_path,
                        is_directory=isinstance(event, DirMovedEvent),
                    )

        # Create and start the observer
        self._observer = Observer()
        self._observer.schedule(
            EventHandler(),
            str(watch_path),
            recursive=self.recursive,
        )
        self._observer.start()
        log.info(f"File watcher started on {watch_path}")

    async def wait_for_event(
        self, context: ProcessingContext
    ) -> FileWatchTriggerOutput | None:
        """Wait for the next filesystem event."""
        return await self.get_event_from_queue()

    async def cleanup_trigger(self, context: ProcessingContext) -> None:
        """Stop the filesystem watcher."""
        log.info("Cleaning up file watch trigger")

        if self._observer:
            try:
                self._observer.stop()
                self._observer.join(timeout=2.0)
            except Exception as e:
                log.warning(f"Error stopping observer: {e}")
            finally:
                self._observer = None


class IntervalTrigger(TriggerNode[IntervalTriggerOutput]):
    """
    Trigger node that fires at regular time intervals.

    This trigger emits events at a configured interval, similar to a timer
    or scheduler. Each event contains:
    - The tick number (how many times the trigger has fired)
    - The current timestamp
    - The configured interval

    This trigger is useful for:
    - Periodic data collection or polling
    - Scheduled batch processing
    - Heartbeat or keepalive workflows
    - Time-based automation
    """

    interval_seconds: float = Field(
        default=60.0,
        description="Interval between triggers in seconds",
        gt=0,
    )
    initial_delay_seconds: float = Field(
        default=0.0,
        description="Delay before the first trigger fires",
        ge=0,
    )
    emit_on_start: bool = Field(
        default=True,
        description="Whether to emit an event immediately on start",
    )
    include_drift_compensation: bool = Field(
        default=True,
        description="Compensate for execution time to maintain accurate intervals",
    )
    OutputType: ClassVar[type] = IntervalTriggerOutput

    def __init__(self, **data: Any):
        super().__init__(**data)
        self._tick_count = 0
        self._start_time: datetime | None = None

    async def setup_trigger(self, context: ProcessingContext) -> None:
        """Initialize the interval trigger."""
        log.info(f"Setting up interval trigger with {self.interval_seconds}s interval")
        self._tick_count = 0
        self._start_time = datetime.now(timezone.utc)

        # Apply initial delay
        if self.initial_delay_seconds > 0:
            log.debug(f"Waiting {self.initial_delay_seconds}s initial delay")
            await asyncio.sleep(self.initial_delay_seconds)

    async def wait_for_event(
        self, context: ProcessingContext
    ) -> IntervalTriggerOutput | None:
        """Wait for the next interval tick."""
        # Handle first tick
        if self._tick_count == 0:
            if self.emit_on_start:
                self._tick_count += 1
                return self._create_event()
            else:
                # Wait for first interval before first tick
                log.debug(
                    f"Interval trigger waiting {self.interval_seconds:.2f}s for first tick"
                )
                try:
                    await asyncio.sleep(self.interval_seconds)
                except asyncio.CancelledError:
                    return None

                if not self._is_running:
                    return None

                self._tick_count += 1
                return self._create_event()

        # Wait for the interval (subsequent ticks)
        if self.include_drift_compensation:
            # Calculate next tick time based on start time
            if self._start_time is None:
                self._start_time = datetime.now(timezone.utc)

            # For ticks after the first, calculate based on when we should fire next
            # If emit_on_start=False, first tick is at interval_seconds from start
            # So tick N (1-indexed) should fire at N * interval_seconds from start
            next_tick = self._tick_count * self.interval_seconds
            if self.initial_delay_seconds > 0:
                next_tick += self.initial_delay_seconds

            elapsed = (datetime.now(timezone.utc) - self._start_time).total_seconds()
            wait_time = max(MIN_WAIT_SECONDS, next_tick - elapsed)
        else:
            wait_time = self.interval_seconds

        log.debug(f"Interval trigger waiting {wait_time:.2f}s for next tick")

        try:
            await asyncio.sleep(wait_time)
        except asyncio.CancelledError:
            return None

        if not self._is_running:
            return None

        self._tick_count += 1
        return self._create_event()

    def _create_event(self) -> IntervalTriggerOutput:
        """Create an interval event."""
        now = datetime.now(timezone.utc)
        elapsed = 0.0
        if self._start_time:
            elapsed = (now - self._start_time).total_seconds()

        return {
            "tick": self._tick_count,
            "elapsed_seconds": elapsed,
            "interval_seconds": self.interval_seconds,
            "timestamp": now.isoformat(),
            "source": "interval",
            "event_type": "tick",
        }

    async def cleanup_trigger(self, context: ProcessingContext) -> None:
        """Clean up the interval trigger."""
        log.info(f"Interval trigger stopping after {self._tick_count} ticks")


class ManualTrigger(TriggerNode[ManualTriggerOutput]):
    """
    Trigger node that waits for manual events pushed via the API.

    This trigger enables interactive workflows where events are pushed
    programmatically through the workflow runner's input API. Each event
    pushed to the trigger is emitted and processed by the workflow.

    This trigger is useful for:
    - Building chatbot-style workflows
    - Interactive processing pipelines
    - Manual batch processing
    - Testing and debugging workflows
    """

    name: str = Field(
        default="manual_trigger",
        description="Name for this trigger (used in API calls)",
    )
    timeout_seconds: float | None = Field(
        default=None,
        description="Timeout waiting for events (None = wait forever)",
        ge=0,
    )
    OutputType: ClassVar[type] = ManualTriggerOutput

    async def setup_trigger(self, context: ProcessingContext) -> None:
        """Initialize the manual trigger."""
        log.info(f"Setting up manual trigger: {self.name}")
        # Manual triggers don't need special setup
        pass

    async def wait_for_event(
        self, context: ProcessingContext
    ) -> ManualTriggerOutput | None:
        """Wait for a manually pushed event."""
        log.debug(f"Manual trigger waiting for event (timeout={self.timeout_seconds})")

        event = await self.get_event_from_queue(timeout=self.timeout_seconds)

        if event is None:
            if self.timeout_seconds is not None:
                log.info(f"Manual trigger timed out after {self.timeout_seconds}s")
            return None

        return event

    async def cleanup_trigger(self, context: ProcessingContext) -> None:
        """Clean up the manual trigger."""
        log.info(f"Manual trigger {self.name} stopping")

    def push_data(self, data: Any, event_type: str = "manual") -> None:
        """
        Convenience method to push data as an event.

        This wraps the data in the trigger output structure.

        Args:
            data: The data to include in the event.
            event_type: The type of event (default: "manual").
        """
        event = {
            "data": data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": self.name,
            "event_type": event_type,
        }
        self.push_event(event)


class WebhookTrigger(TriggerNode[WebhookTriggerOutput]):
    """
    Trigger node that starts an HTTP server to receive webhook requests.

    Each incoming HTTP request is emitted as an event containing:
    - The request body (parsed as JSON if applicable)
    - Request headers
    - Query parameters
    - HTTP method

    This trigger is useful for:
    - Receiving notifications from external services
    - Building API endpoints that trigger workflows
    - Integration with third-party webhook providers
    """

    port: int = Field(
        default=8080,
        description="Port to listen on for webhook requests",
        ge=1,
        le=65535,
    )
    path: str = Field(
        default="/webhook",
        description="URL path to listen on",
    )
    host: str = Field(
        default="127.0.0.1",
        description="Host address to bind to. Use '0.0.0.0' to listen on all interfaces.",
    )
    methods: list[str] = Field(
        default=["POST"],
        description="HTTP methods to accept",
    )
    secret: str = Field(
        default="",
        description="Optional secret for validating requests (checks X-Webhook-Secret header)",
    )
    OutputType: ClassVar[type] = WebhookTriggerOutput

    def __init__(self, **data: Any):
        super().__init__(**data)
        self._server: asyncio.Server | None = None
        self._server_task: asyncio.Task | None = None

    async def setup_trigger(self, context: ProcessingContext) -> None:
        """Start the HTTP server for receiving webhooks."""
        log.info(f"Setting up webhook trigger on {self.host}:{self.port}{self.path}")

        # Use aiohttp to create a simple HTTP server
        try:
            from aiohttp import web
        except ImportError:
            raise ImportError(
                "aiohttp is required for WebhookTrigger. "
                "Install it with: pip install aiohttp"
            )

        app = web.Application()

        async def handle_webhook(request: web.Request) -> web.Response:
            """Handle incoming webhook requests."""
            # Check method
            if request.method not in self.methods:
                return web.Response(
                    status=405,
                    text=f"Method {request.method} not allowed",
                )

            # Check secret if configured
            if self.secret:
                provided_secret = request.headers.get("X-Webhook-Secret", "")
                if provided_secret != self.secret:
                    return web.Response(status=401, text="Invalid secret")

            # Parse body
            body: Any = None
            content_type = request.content_type
            try:
                if content_type == "application/json":
                    body = await request.json()
                elif content_type in (
                    "application/x-www-form-urlencoded",
                    "multipart/form-data",
                ):
                    body = dict(await request.post())
                else:
                    body = await request.text()
            except Exception as e:
                log.warning(f"Failed to parse request body: {e}")
                body = await request.text()

            # Build event
            event = {
                "body": body,
                "headers": dict(request.headers),
                "query": dict(request.query),
                "method": request.method,
                "path": request.path,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": f"{request.remote}",
                "event_type": "webhook",
            }

            # Push to queue
            self.push_event(event)

            return web.Response(
                status=200,
                text=json.dumps({"status": "accepted"}),
                content_type="application/json",
            )

        # Register handler for all methods at the configured path
        for method in self.methods:
            app.router.add_route(method, self.path, handle_webhook)

        # Start the server
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)

        try:
            await site.start()
            log.info(f"Webhook server started on {self.host}:{self.port}")
        except OSError as e:
            raise RuntimeError(
                f"Failed to start webhook server on {self.host}:{self.port}: {e}"
            )

        # Store references for cleanup
        self._runner = runner
        self._site = site

    async def wait_for_event(
        self, context: ProcessingContext
    ) -> WebhookTriggerOutput | None:
        """Wait for the next webhook request."""
        return await self.get_event_from_queue()

    async def cleanup_trigger(self, context: ProcessingContext) -> None:
        """Stop the HTTP server."""
        log.info("Cleaning up webhook trigger")

        if hasattr(self, "_site") and self._site:
            try:
                await self._site.stop()
            except Exception as e:
                log.warning(f"Error stopping site: {e}")

        if hasattr(self, "_runner") and self._runner:
            try:
                await self._runner.cleanup()
            except Exception as e:
                log.warning(f"Error cleaning up runner: {e}")


__all__ = [
    "TriggerNode",
    "WebhookTrigger",
    "FileWatchTrigger",
    "IntervalTrigger",
    "ManualTrigger",
]
