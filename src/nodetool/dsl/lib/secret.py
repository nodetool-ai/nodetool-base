from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode


class GetSecret(GraphNode):
    """
    Get a secret value from configuration.
    secrets, credentials, configuration
    """

    name: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="Secret key name"
    )
    default: str | None | GraphNode | tuple[GraphNode, str] = Field(
        default=None, description="Default value if not found"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.secret.GetSecret"


class SetSecret(GraphNode):
    """
    Set a secret value and persist it.
    secrets, credentials, configuration
    """

    name: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="Secret key name"
    )
    value: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="Secret value"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.secret.SetSecret"
