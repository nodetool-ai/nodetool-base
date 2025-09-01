from __future__ import annotations

from typing import Any
from pydantic import Field

from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.io import NodeInputs, NodeOutputs
from nodetool.code_runners.server_runner import ServerDockerRunner
from nodetool.code_runners.runtime_base import StreamRunnerBase


class SimpleHttpServer(BaseNode):
    """
    Starts a simple HTTP server inside Docker and streams logs.
    http, server, web

    Emits the reachable endpoint URL on the "endpoint" output when ready,
    then streams stdout/stderr lines on the corresponding outputs.
    """

    _supports_dynamic_outputs: bool = True
    _runner: StreamRunnerBase | None = None

    image: str = Field(
        default="python:3.11-slim",
        description="Docker image to run the server in",
    )
    container_port: int = Field(
        default=8000, description="Port the server listens on inside the container"
    )
    command: str = Field(
        default="",
        description=(
            "Startup command. If empty, uses 'python -m http.server <container_port> --bind 0.0.0.0'"
        ),
    )
    timeout_seconds: int = Field(
        default=600, description="Max lifetime of the server container (seconds)"
    )
    ready_timeout_seconds: int = Field(
        default=15, description="Seconds to wait for server readiness"
    )

    @classmethod
    def is_streaming_output(cls) -> bool:
        return True

    @classmethod
    def return_type(cls) -> dict[str, Any]:
        return {"endpoint": str, "stdout": str, "stderr": str}

    def get_timeout_seconds(self) -> float | None:  # type: ignore[override]
        """Return an overall timeout for the server container.

        Uses the configured ``timeout_seconds`` plus small headroom to include
        startup/teardown time.

        Returns:
            float | None: Timeout in seconds.
        """
        try:
            return max(5.0, float(self.timeout_seconds) + 5.0)
        except Exception:
            return 60.0

    async def run(self, context: ProcessingContext, inputs: NodeInputs, outputs: NodeOutputs) -> None:  # type: ignore[override]
        cmd = (
            self.command.strip()
            or f"python -m http.server {self.container_port} --bind 0.0.0.0"
        )

        runner = ServerDockerRunner(
            image=self.image,
            container_port=self.container_port,
            scheme="http",
            timeout_seconds=self.timeout_seconds,
            ready_timeout_seconds=self.ready_timeout_seconds,
            endpoint_path="",  # plain http root
        )
        self._runner = runner

        async for slot, value in runner.stream(
            user_code=cmd,
            env_locals={},
            context=context,
            node=self,
        ):
            if value is None:
                continue
            text = value if isinstance(value, str) else str(value)
            if slot not in ("stdout", "stderr", "endpoint"):
                # Pass through unknown slots as stdout for visibility
                slot = "stdout"
            await outputs.emit(slot, text)

    async def finalize(self, context: ProcessingContext):  # type: ignore[override]
        """Stop the server container if still running.

        Args:
            context: Processing context (unused).

        Returns:
            None
        """
        if self._runner:
            try:
                self._runner.stop()
            except Exception:
                pass
