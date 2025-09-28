from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode


class HashFile(GraphNode):
    """Compute the cryptographic hash of a file.
    hash, hashlib, digest, file

    Use cases:
    - Verify downloaded files
    - Detect file changes
    - Identify duplicates
    """

    file: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The file to hash"
    )
    algorithm: str | GraphNode | tuple[GraphNode, str] = Field(
        default="md5", description="Hash algorithm name (e.g. md5, sha1, sha256)"
    )
    chunk_size: int | GraphNode | tuple[GraphNode, str] = Field(
        default=8192, description="Read size for hashing in bytes"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.hashlib.HashFile"


class HashString(GraphNode):
    """Compute the cryptographic hash of a string using hashlib.
    hash, hashlib, digest, string

    Use cases:
    - Generate deterministic identifiers
    - Verify data integrity
    - Create fingerprints for caching
    """

    text: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The text to hash"
    )
    algorithm: str | GraphNode | tuple[GraphNode, str] = Field(
        default="md5", description="Hash algorithm name (e.g. md5, sha1, sha256)"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.hashlib.HashString"
