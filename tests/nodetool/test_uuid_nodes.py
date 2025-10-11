import pytest
import uuid
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.nodes.lib.uuid import (
    GenerateUUID4,
    GenerateUUID1,
    GenerateUUID3,
    GenerateUUID5,
    ParseUUID,
    FormatUUID,
    IsValidUUID,
    UUIDFormat,
)


@pytest.fixture
def context():
    return ProcessingContext(user_id="test", auth_token="test")


@pytest.mark.asyncio
async def test_generate_uuid4(context: ProcessingContext):
    """Test UUID4 generation produces valid random UUIDs."""
    node = GenerateUUID4()
    result = await node.process(context)

    # Should be a valid UUID string
    parsed = uuid.UUID(result)
    assert parsed.version == 4

    # Generate multiple UUIDs to ensure randomness
    result2 = await node.process(context)
    assert result != result2


@pytest.mark.asyncio
async def test_generate_uuid1(context: ProcessingContext):
    """Test UUID1 generation produces valid time-based UUIDs."""
    node = GenerateUUID1()
    result = await node.process(context)

    # Should be a valid UUID string
    parsed = uuid.UUID(result)
    assert parsed.version == 1


@pytest.mark.asyncio
async def test_generate_uuid3(context: ProcessingContext):
    """Test UUID3 generation produces deterministic name-based UUIDs."""
    # Test with DNS namespace
    node = GenerateUUID3(namespace="dns", name="example.com")
    result1 = await node.process(context)
    result2 = await node.process(context)

    # Should be deterministic - same input produces same output
    assert result1 == result2

    # Should be a valid UUID
    parsed = uuid.UUID(result1)
    assert parsed.version == 3

    # Different names should produce different UUIDs
    node2 = GenerateUUID3(namespace="dns", name="different.com")
    result3 = await node2.process(context)
    assert result1 != result3


@pytest.mark.asyncio
async def test_generate_uuid3_with_different_namespaces(context: ProcessingContext):
    """Test UUID3 with different namespace types."""
    name = "test"

    # Test with different predefined namespaces
    for ns in ["dns", "url", "oid", "x500"]:
        node = GenerateUUID3(namespace=ns, name=name)
        result = await node.process(context)
        parsed = uuid.UUID(result)
        assert parsed.version == 3

    # Test with custom UUID namespace
    custom_ns = str(uuid.uuid4())
    node = GenerateUUID3(namespace=custom_ns, name=name)
    result = await node.process(context)
    parsed = uuid.UUID(result)
    assert parsed.version == 3


@pytest.mark.asyncio
async def test_generate_uuid3_invalid_namespace(context: ProcessingContext):
    """Test UUID3 with invalid namespace raises error."""
    node = GenerateUUID3(namespace="invalid", name="test")
    with pytest.raises(ValueError, match="Invalid namespace"):
        await node.process(context)


@pytest.mark.asyncio
async def test_generate_uuid5(context: ProcessingContext):
    """Test UUID5 generation produces deterministic name-based UUIDs."""
    # Test with URL namespace
    node = GenerateUUID5(namespace="url", name="https://example.com")
    result1 = await node.process(context)
    result2 = await node.process(context)

    # Should be deterministic
    assert result1 == result2

    # Should be a valid UUID
    parsed = uuid.UUID(result1)
    assert parsed.version == 5

    # Different names should produce different UUIDs
    node2 = GenerateUUID5(namespace="url", name="https://different.com")
    result3 = await node2.process(context)
    assert result1 != result3


@pytest.mark.asyncio
async def test_generate_uuid5_with_different_namespaces(context: ProcessingContext):
    """Test UUID5 with different namespace types."""
    name = "test"

    # Test with different predefined namespaces
    for ns in ["dns", "url", "oid", "x500"]:
        node = GenerateUUID5(namespace=ns, name=name)
        result = await node.process(context)
        parsed = uuid.UUID(result)
        assert parsed.version == 5


@pytest.mark.asyncio
async def test_generate_uuid5_invalid_namespace(context: ProcessingContext):
    """Test UUID5 with invalid namespace raises error."""
    node = GenerateUUID5(namespace="invalid", name="test")
    with pytest.raises(ValueError, match="Invalid namespace"):
        await node.process(context)


@pytest.mark.asyncio
async def test_parse_uuid_valid(context: ProcessingContext):
    """Test parsing a valid UUID."""
    test_uuid = str(uuid.uuid4())
    node = ParseUUID(uuid_string=test_uuid)
    result = await node.process(context)

    assert result["is_valid"] is True
    assert result["uuid"] == test_uuid
    assert result["version"] == 4
    assert "hex" in result
    assert "int" in result
    assert "variant" in result


@pytest.mark.asyncio
async def test_parse_uuid_invalid(context: ProcessingContext):
    """Test parsing an invalid UUID."""
    node = ParseUUID(uuid_string="not-a-uuid")
    result = await node.process(context)

    assert result["is_valid"] is False
    assert "error" in result


@pytest.mark.asyncio
async def test_format_uuid_standard(context: ProcessingContext):
    """Test formatting UUID in standard format."""
    test_uuid = str(uuid.uuid4())
    node = FormatUUID(uuid_string=test_uuid, format=UUIDFormat.STANDARD)
    result = await node.process(context)

    assert result == test_uuid


@pytest.mark.asyncio
async def test_format_uuid_hex(context: ProcessingContext):
    """Test formatting UUID as hex."""
    test_uuid = str(uuid.uuid4())
    node = FormatUUID(uuid_string=test_uuid, format=UUIDFormat.HEX)
    result = await node.process(context)

    # Hex format should have no dashes and be 32 chars
    assert "-" not in result
    assert len(result) == 32


@pytest.mark.asyncio
async def test_format_uuid_urn(context: ProcessingContext):
    """Test formatting UUID as URN."""
    test_uuid = str(uuid.uuid4())
    node = FormatUUID(uuid_string=test_uuid, format=UUIDFormat.URN)
    result = await node.process(context)

    assert result.startswith("urn:uuid:")


@pytest.mark.asyncio
async def test_format_uuid_int(context: ProcessingContext):
    """Test formatting UUID as integer."""
    test_uuid = str(uuid.uuid4())
    node = FormatUUID(uuid_string=test_uuid, format=UUIDFormat.INT)
    result = await node.process(context)

    # Should be a string representation of an integer
    assert result.isdigit()
    assert int(result) > 0


@pytest.mark.asyncio
async def test_format_uuid_bytes_hex(context: ProcessingContext):
    """Test formatting UUID as bytes hex."""
    test_uuid = str(uuid.uuid4())
    node = FormatUUID(uuid_string=test_uuid, format=UUIDFormat.BYTES_HEX)
    result = await node.process(context)

    # Should be hex string of 32 characters
    assert len(result) == 32
    assert all(c in "0123456789abcdef" for c in result)


@pytest.mark.asyncio
async def test_format_uuid_invalid(context: ProcessingContext):
    """Test formatting an invalid UUID raises error."""
    node = FormatUUID(uuid_string="not-a-uuid", format=UUIDFormat.STANDARD)
    with pytest.raises(ValueError, match="Invalid UUID string"):
        await node.process(context)


@pytest.mark.asyncio
async def test_is_valid_uuid_true(context: ProcessingContext):
    """Test IsValidUUID returns True for valid UUID."""
    test_uuid = str(uuid.uuid4())
    node = IsValidUUID(uuid_string=test_uuid)
    result = await node.process(context)

    assert result is True


@pytest.mark.asyncio
async def test_is_valid_uuid_false(context: ProcessingContext):
    """Test IsValidUUID returns False for invalid UUID."""
    node = IsValidUUID(uuid_string="not-a-uuid")
    result = await node.process(context)

    assert result is False


@pytest.mark.asyncio
async def test_is_valid_uuid_empty_string(context: ProcessingContext):
    """Test IsValidUUID returns False for empty string."""
    node = IsValidUUID(uuid_string="")
    result = await node.process(context)

    assert result is False


@pytest.mark.asyncio
async def test_uuid3_vs_uuid5_same_input(context: ProcessingContext):
    """Test that UUID3 and UUID5 produce different results for same input."""
    namespace = "dns"
    name = "example.com"

    node3 = GenerateUUID3(namespace=namespace, name=name)
    result3 = await node3.process(context)

    node5 = GenerateUUID5(namespace=namespace, name=name)
    result5 = await node5.process(context)

    # Same input but different versions should produce different UUIDs
    assert result3 != result5

    # But both should be valid
    assert uuid.UUID(result3).version == 3
    assert uuid.UUID(result5).version == 5
