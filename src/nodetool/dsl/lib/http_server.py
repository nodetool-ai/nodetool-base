from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode


class SimpleHttpServer(GraphNode):
    """
    Starts a simple HTTP server inside Docker and streams logs.
    http, server, web

    Emits the reachable endpoint URL on the "endpoint" output when ready,
    then streams stdout/stderr lines on the corresponding outputs.
    """

    image: str | GraphNode | tuple[GraphNode, str] = Field(
        default="python:3.11-slim", description="Docker image to run the server in"
    )
    container_port: int | GraphNode | tuple[GraphNode, str] = Field(
        default=8000, description="Port the server listens on inside the container"
    )
    command: str | GraphNode | tuple[GraphNode, str] = Field(
        default="",
        description="Startup command. If empty, uses 'python -m http.server <container_port> --bind 0.0.0.0'",
    )
    timeout_seconds: int | GraphNode | tuple[GraphNode, str] = Field(
        default=600, description="Max lifetime of the server container (seconds)"
    )
    ready_timeout_seconds: int | GraphNode | tuple[GraphNode, str] = Field(
        default=15, description="Seconds to wait for server readiness"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.http_server.SimpleHttpServer"
