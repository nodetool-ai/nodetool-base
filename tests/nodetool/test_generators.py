import asyncio
import json
import pytest
from unittest.mock import patch

from nodetool.metadata.types import (
    ColumnDef,
    DataframeRef,
    LanguageModel,
    Provider,
    RecordType,
    ToolCall,
)
from nodetool.workflows.types import Chunk
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.nodes.nodetool.generators import DataGenerator, ListGenerator
from nodetool.providers import FakeProvider, create_tool_calling_fake_provider


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
    # Create tool calls for each data row
    from nodetool.metadata.types import ToolCall
    from nodetool.providers import FakeProvider

    # Create a fake provider that returns tool calls
    class DataGeneratorFakeProvider(FakeProvider):
        async def generate_messages(self, **kwargs):
            # Yield tool calls for each row
            yield ToolCall(name="generate_data", args={"name": "Alice", "age": 30})
            yield ToolCall(name="generate_data", args={"name": "Bob", "age": 25})

    fake_provider = DataGeneratorFakeProvider()

    with patch(
        "nodetool.nodes.nodetool.generators.get_provider", return_value=fake_provider
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

    # Simulate two tool calls (rows)
    tool_calls = [
        ToolCall(id="1", name="generate_data", args={"name": "Alice", "age": 30}),
        ToolCall(id="2", name="generate_data", args={"name": "Bob", "age": 25}),
    ]

    # Create a FakeProvider that yields tool calls
    fake_provider = create_tool_calling_fake_provider(tool_calls)

    with patch(
        "nodetool.nodes.nodetool.generators.get_provider", return_value=fake_provider
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
        "nodetool.nodes.nodetool.generators.get_provider", return_value=fake_provider
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
