import hashlib
from typing import ClassVar
from pydantic import Field
from nodetool.metadata.types import FilePath
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext


class HashString(BaseNode):
    """Compute the cryptographic hash of a string using hashlib.
    hash, hashlib, digest, string

    Use cases:
    - Generate deterministic identifiers
    - Verify data integrity
    - Create fingerprints for caching
    """

    text: str = Field(default="", description="The text to hash")
    algorithm: str = Field(
        default="md5",
        description="Hash algorithm name (e.g. md5, sha1, sha256)",
    )

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> str:
        try:
            hasher = getattr(hashlib, self.algorithm)()
        except AttributeError as exc:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}") from exc
        hasher.update(self.text.encode("utf-8"))
        return hasher.hexdigest()


class HashFile(BaseNode):
    """Compute the cryptographic hash of a file.
    hash, hashlib, digest, file

    Use cases:
    - Verify downloaded files
    - Detect file changes
    - Identify duplicates
    """

    file: FilePath = Field(default=FilePath(), description="The file to hash")
    algorithm: str = Field(
        default="md5",
        description="Hash algorithm name (e.g. md5, sha1, sha256)",
    )
    chunk_size: int = Field(
        default=8192, ge=1, description="Read size for hashing in bytes"
    )

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> str:
        try:
            hasher = getattr(hashlib, self.algorithm)()
        except AttributeError as exc:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}") from exc

        with open(self.file.path, "rb") as f:
            for chunk in iter(lambda: f.read(self.chunk_size), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
