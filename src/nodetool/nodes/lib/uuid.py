import uuid
from enum import Enum
from typing import ClassVar
from pydantic import Field
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext


class GenerateUUID4(BaseNode):
    """Generate a random UUID (version 4).
    uuid, random, identifier, unique, guid

    Use cases:
    - Create unique identifiers for records
    - Generate session IDs
    - Produce random unique keys
    """

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> str:
        return str(uuid.uuid4())


class GenerateUUID1(BaseNode):
    """Generate a time-based UUID (version 1).
    uuid, time, identifier, unique, guid, timestamp

    Use cases:
    - Create sortable unique identifiers
    - Generate time-ordered IDs
    - Track creation timestamps in IDs
    """

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> str:
        return str(uuid.uuid1())


class GenerateUUID3(BaseNode):
    """Generate a name-based UUID using MD5 (version 3).
    uuid, name, identifier, unique, guid, md5, deterministic

    Use cases:
    - Create deterministic IDs from names
    - Generate consistent identifiers for the same input
    - Map names to unique identifiers
    """

    namespace: str = Field(
        default="dns",
        description="Namespace (dns, url, oid, x500, or a UUID string)",
    )
    name: str = Field(default="", description="Name to generate UUID from")

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> str:
        ns = self._get_namespace(self.namespace)
        return str(uuid.uuid3(ns, self.name))

    def _get_namespace(self, namespace: str) -> uuid.UUID:
        """Get the namespace UUID from a string identifier."""
        namespace_map = {
            "dns": uuid.NAMESPACE_DNS,
            "url": uuid.NAMESPACE_URL,
            "oid": uuid.NAMESPACE_OID,
            "x500": uuid.NAMESPACE_X500,
        }
        if namespace.lower() in namespace_map:
            return namespace_map[namespace.lower()]
        try:
            return uuid.UUID(namespace)
        except ValueError as exc:
            raise ValueError(
                f"Invalid namespace: {namespace}. Use dns, url, oid, x500, or a valid UUID string"
            ) from exc


class GenerateUUID5(BaseNode):
    """Generate a name-based UUID using SHA-1 (version 5).
    uuid, name, identifier, unique, guid, sha1, deterministic

    Use cases:
    - Create deterministic IDs from names (preferred over UUID3)
    - Generate consistent identifiers for the same input
    - Map names to unique identifiers with better collision resistance
    """

    namespace: str = Field(
        default="dns",
        description="Namespace (dns, url, oid, x500, or a UUID string)",
    )
    name: str = Field(default="", description="Name to generate UUID from")

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> str:
        ns = self._get_namespace(self.namespace)
        return str(uuid.uuid5(ns, self.name))

    def _get_namespace(self, namespace: str) -> uuid.UUID:
        """Get the namespace UUID from a string identifier."""
        namespace_map = {
            "dns": uuid.NAMESPACE_DNS,
            "url": uuid.NAMESPACE_URL,
            "oid": uuid.NAMESPACE_OID,
            "x500": uuid.NAMESPACE_X500,
        }
        if namespace.lower() in namespace_map:
            return namespace_map[namespace.lower()]
        try:
            return uuid.UUID(namespace)
        except ValueError as exc:
            raise ValueError(
                f"Invalid namespace: {namespace}. Use dns, url, oid, x500, or a valid UUID string"
            ) from exc


class ParseUUID(BaseNode):
    """Parse and validate a UUID string.
    uuid, parse, validate, check, identifier

    Use cases:
    - Validate UUID format
    - Normalize UUID strings
    - Extract UUID version information
    """

    uuid_string: str = Field(default="", description="UUID string to parse")

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> dict:
        try:
            parsed = uuid.UUID(self.uuid_string)
            return {
                "uuid": str(parsed),
                "version": parsed.version,
                "variant": parsed.variant,
                "hex": parsed.hex,
                "int": parsed.int,
                "is_valid": True,
            }
        except ValueError as exc:
            return {"uuid": self.uuid_string, "is_valid": False, "error": str(exc)}


class UUIDFormat(str, Enum):
    STANDARD = "standard"
    HEX = "hex"
    URN = "urn"
    INT = "int"
    BYTES_HEX = "bytes_hex"


class FormatUUID(BaseNode):
    """Format a UUID string in different representations.
    uuid, format, convert, hex, urn, identifier

    Use cases:
    - Convert UUID to different formats
    - Generate URN representations
    - Format UUIDs for specific use cases
    """

    uuid_string: str = Field(default="", description="UUID string to format")
    format: UUIDFormat = Field(
        default=UUIDFormat.STANDARD,
        description="Output format (standard, hex, urn, int, bytes_hex)",
    )

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> str:
        try:
            parsed = uuid.UUID(self.uuid_string)
        except ValueError as exc:
            raise ValueError(f"Invalid UUID string: {self.uuid_string}") from exc

        if self.format == UUIDFormat.STANDARD:
            return str(parsed)
        elif self.format == UUIDFormat.HEX:
            return parsed.hex
        elif self.format == UUIDFormat.URN:
            return parsed.urn
        elif self.format == UUIDFormat.INT:
            return str(parsed.int)
        elif self.format == UUIDFormat.BYTES_HEX:
            return parsed.bytes.hex()
        else:
            raise ValueError(f"Unsupported format: {self.format}")


class IsValidUUID(BaseNode):
    """Check if a string is a valid UUID.
    uuid, validate, check, verify, identifier

    Use cases:
    - Validate user input
    - Filter valid UUIDs from a dataset
    - Conditional workflow based on UUID validity
    """

    uuid_string: str = Field(default="", description="String to check")

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> bool:
        try:
            uuid.UUID(self.uuid_string)
            return True
        except (ValueError, AttributeError):
            return False
