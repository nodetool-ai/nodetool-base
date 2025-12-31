import pytest
from unittest.mock import patch, MagicMock

from nodetool.metadata.types import (
    ColumnDef,
    DataframeRef,
    LanguageModel,
    Provider,
    RecordType,
)
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.nodes.nodetool.generators import (
    DataGenerator,
    ListGenerator,
    build_schema_from_slots,
    build_schema_from_record_type,
    format_structured_instructions,
)
from nodetool.providers import FakeProvider


@pytest.fixture
def context():
    return ProcessingContext(user_id="test", auth_token="test")


@pytest.mark.asyncio
async def test_data_generator_process(context: ProcessingContext):
    # Arrange
    columns = RecordType(
        columns=[
            ColumnDef(name="name", data_type="string"),
            ColumnDef(name="age", data_type="int"),
        ]
    )
    node = DataGenerator(
        model=LanguageModel(provider=Provider.OpenAI, id="gpt-4"),
        prompt="Generate people",
        input_text="",
        columns=columns,
        max_tokens=256,
    )

    # Create a markdown table response
    markdown_table = """| name | age |
|------|-----|
| Alice | 30 |
| Bob | 25 |"""

    # Create a fake provider that streams markdown table
    fake_provider = FakeProvider(text_response=markdown_table, should_stream=True)

    with patch(
        "nodetool.workflows.processing_context.ProcessingContext.get_provider",
        return_value=fake_provider,
    ):
        # Act - Use gen_process to collect all outputs
        result_dataframe = None
        records = []

        async for output in node.gen_process(context):
            if output["record"] is not None:
                records.append(output["record"])
            if output["dataframe"] is not None:
                result_dataframe = output["dataframe"]

        # Assert
        assert result_dataframe is not None
        assert result_dataframe.columns is not None
        assert [c.name for c in result_dataframe.columns] == ["name", "age"]
        assert result_dataframe.data == [["Alice", 30], ["Bob", 25]]
        assert len(records) == 2
        assert records[0] == {"name": "Alice", "age": 30}
        assert records[1] == {"name": "Bob", "age": 25}


@pytest.mark.asyncio
async def test_data_generator_gen_process_streaming(context: ProcessingContext):
    # Arrange
    columns = RecordType(
        columns=[
            ColumnDef(name="name", data_type="string"),
            ColumnDef(name="age", data_type="int"),
        ]
    )
    node = DataGenerator(
        model=LanguageModel(provider=Provider.OpenAI, id="gpt-4"),
        prompt="Generate people",
        input_text="",
        columns=columns,
        max_tokens=256,
    )

    # Create a markdown table response
    markdown_table = """| name | age |
|------|-----|
| Alice | 30 |
| Bob | 25 |"""

    # Create a FakeProvider that streams markdown table chunks
    fake_provider = FakeProvider(text_response=markdown_table, should_stream=True, chunk_size=10)

    with patch(
        "nodetool.workflows.processing_context.ProcessingContext.get_provider",
        return_value=fake_provider,
    ):
        # Act: collect stream
        streamed_records = []
        final_df: DataframeRef | None = None
        async for output in node.gen_process(context):
            if output["record"] is not None:
                streamed_records.append(output["record"])
            if output["dataframe"] is not None:
                final_df = output["dataframe"]

        # Assert per-record
        assert streamed_records == [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
        ]
        # Assert final dataframe
        assert final_df is not None
        assert final_df.data == [["Alice", 30], ["Bob", 25]]
        assert [c.name for c in final_df.columns or []] == ["name", "age"]


@pytest.mark.asyncio
async def test_list_generator_gen_process_streaming(context: ProcessingContext):
    # Arrange
    node = ListGenerator(
        model=LanguageModel(provider=Provider.OpenAI, id="gpt-4"),
        prompt="Generate 3 items",
        input_text="",
        max_tokens=128,
    )

    # Simulate model streaming a numbered list
    text = """
1. Alpha
2. Beta
3. Gamma
""".strip()

    # Create a FakeProvider that streams the text as chunks
    fake_provider = FakeProvider(text_response=text, should_stream=True, chunk_size=10)

    with patch(
        "nodetool.workflows.processing_context.ProcessingContext.get_provider",
        return_value=fake_provider,
    ):
        # Act
        items: list[str] = []
        indices: list[int] = []
        async for output in node.gen_process(context):
            # ListGenerator yields dicts with "item" and "index" keys
            if "item" in output:
                assert isinstance(output["item"], str)
                items.append(output["item"])
            if "index" in output:
                assert isinstance(output["index"], int)
                indices.append(output["index"])

        # Assert streamed items and indexes
        assert items == ["Alpha", "Beta", "Gamma"]
        assert indices == [1, 2, 3]


def test_build_schema_from_record_type():
    """Test build_schema_from_record_type with various column types."""
    columns = RecordType(columns=[
        ColumnDef(name="name", data_type="string"),
        ColumnDef(name="age", data_type="int"),
        ColumnDef(name="price", data_type="float"),
        ColumnDef(name="created_at", data_type="datetime"),
        ColumnDef(name="metadata", data_type="object"),
    ])
    
    schema = build_schema_from_record_type(columns, title="Test Schema")
    
    assert schema["type"] == "object"
    assert schema["title"] == "Test Schema"
    assert schema["additionalProperties"] is False
    assert "name" in schema["properties"]
    assert schema["properties"]["name"] == {"type": "string"}
    assert schema["properties"]["age"] == {"type": "integer"}
    assert schema["properties"]["price"] == {"type": "number"}
    assert schema["properties"]["created_at"] == {"type": "string", "format": "date-time"}
    assert schema["required"] == ["name", "age", "price", "created_at", "metadata"]


def test_build_schema_from_record_type_empty_raises():
    """Test build_schema_from_record_type raises error with empty columns."""
    columns = RecordType(columns=[])
    
    with pytest.raises(ValueError, match="Define columns"):
        build_schema_from_record_type(columns)


def test_build_schema_from_slots():
    """Test build_schema_from_slots with mock slots."""
    # Create mock slots with name and type attributes
    mock_type = MagicMock()
    mock_type.get_json_schema.return_value = {"type": "string"}
    
    mock_slot = MagicMock()
    mock_slot.name = "test_field"
    mock_slot.type = mock_type
    
    schema = build_schema_from_slots([mock_slot], title="Test Output")
    
    assert schema["type"] == "object"
    assert schema["title"] == "Test Output"
    assert schema["additionalProperties"] is False
    assert schema["properties"]["test_field"] == {"type": "string"}
    assert schema["required"] == ["test_field"]


def test_build_schema_from_slots_empty_raises():
    """Test build_schema_from_slots raises error with empty slots."""
    with pytest.raises(ValueError, match="Declare outputs"):
        build_schema_from_slots([])


def test_format_structured_instructions_full():
    """Test format_structured_instructions with all parameters."""
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    result = format_structured_instructions(schema, "Generate a name", "Use formal names")
    
    assert "<JSON_SCHEMA>" in result
    assert "</JSON_SCHEMA>" in result
    assert "<INSTRUCTIONS>" in result
    assert "Generate a name" in result
    assert "<CONTEXT>" in result
    assert "Use formal names" in result


def test_format_structured_instructions_minimal():
    """Test format_structured_instructions with empty instructions and context."""
    schema = {"type": "object"}
    result = format_structured_instructions(schema, "", "")
    
    assert "<JSON_SCHEMA>" in result
    assert "<INSTRUCTIONS>" not in result
    assert "<CONTEXT>" not in result


@pytest.mark.asyncio
async def test_list_generator_with_bullet_points(context: ProcessingContext):
    """Test ListGenerator with bullet point list format."""
    node = ListGenerator(
        model=LanguageModel(provider=Provider.OpenAI, id="gpt-4"),
        prompt="Generate items",
        input_text="",
        max_tokens=128,
    )
    
    text = """
- First item
- Second item
- Third item
""".strip()
    
    fake_provider = FakeProvider(text_response=text, should_stream=True, chunk_size=10)
    
    with patch(
        "nodetool.workflows.processing_context.ProcessingContext.get_provider",
        return_value=fake_provider,
    ):
        items = []
        async for output in node.gen_process(context):
            if "item" in output:
                items.append(output["item"])
        
        assert len(items) == 3
        assert items == ["First item", "Second item", "Third item"]


@pytest.mark.asyncio
async def test_data_generator_parse_markdown_table():
    """Test DataGenerator._parse_markdown_table method."""
    columns = RecordType(columns=[
        ColumnDef(name="id", data_type="int"),
        ColumnDef(name="name", data_type="string"),
    ])
    
    node = DataGenerator(
        model=LanguageModel(provider=Provider.OpenAI, id="gpt-4"),
        columns=columns,
        prompt="",
    )
    
    table_text = """| id | name |
|-----|------|
| 1 | Alice |
| 2 | Bob |"""
    
    rows = node._parse_markdown_table(table_text)
    
    assert len(rows) == 2
    assert rows[0] == {"id": 1, "name": "Alice"}
    assert rows[1] == {"id": 2, "name": "Bob"}


@pytest.mark.asyncio
async def test_data_generator_parse_markdown_table_empty():
    """Test DataGenerator._parse_markdown_table with insufficient rows."""
    columns = RecordType(columns=[])
    node = DataGenerator(
        model=LanguageModel(provider=Provider.OpenAI, id="gpt-4"),
        columns=columns,
        prompt="",
    )
    
    # Only header, no data rows
    table_text = """| id |
|-----|"""
    
    rows = node._parse_markdown_table(table_text)
    assert rows == []


@pytest.mark.asyncio
async def test_data_generator_convert_value():
    """Test DataGenerator._convert_value method."""
    columns = RecordType(columns=[
        ColumnDef(name="count", data_type="int"),
        ColumnDef(name="price", data_type="float"),
        ColumnDef(name="name", data_type="string"),
    ])
    
    node = DataGenerator(
        model=LanguageModel(provider=Provider.OpenAI, id="gpt-4"),
        columns=columns,
        prompt="",
    )
    
    # Test int conversion
    assert node._convert_value("count", "42") == 42
    
    # Test float conversion
    assert node._convert_value("price", "19.99") == 19.99
    
    # Test string passthrough
    assert node._convert_value("name", "Alice") == "Alice"
    
    # Test None handling
    assert node._convert_value("count", "None") is None
    assert node._convert_value("count", "") is None
    
    # Test unknown column returns value as-is
    assert node._convert_value("unknown", "test") == "test"
