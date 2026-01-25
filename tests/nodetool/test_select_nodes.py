"""Tests for Select constant and SelectInput nodes."""

import pytest
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.nodes.nodetool.constant import Select
from nodetool.nodes.nodetool.input import SelectInput


@pytest.fixture
def context():
    return ProcessingContext(user_id="test", auth_token="test")


# Select constant node tests


@pytest.mark.asyncio
async def test_select_constant_node(context: ProcessingContext):
    """Test Select constant node returns the selected value as a string."""
    node = Select(
        value="option1",
        options=["option1", "option2", "option3"],
        enum_type_name="TestEnum",
    )
    result = await node.process(context)
    assert result == "option1"
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_select_constant_node_empty_options(context: ProcessingContext):
    """Test Select constant node with empty options."""
    node = Select(value="", options=[], enum_type_name="")
    result = await node.process(context)
    assert result == ""
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_select_constant_node_basic_fields():
    """Test that Select node only exposes value in basic fields."""
    basic_fields = Select.get_basic_fields()
    assert "value" in basic_fields
    # options and enum_type_name should not be in basic fields (they're plumbing)
    assert "options" not in basic_fields
    assert "enum_type_name" not in basic_fields


# SelectInput node tests


@pytest.mark.asyncio
async def test_select_input_node(context: ProcessingContext):
    """Test SelectInput node returns the selected value as a string."""
    node = SelectInput(
        name="select_input",
        value="choice_a",
        options=["choice_a", "choice_b", "choice_c"],
        enum_type_name="TestEnumType",
        description="test select input",
    )
    result = await node.process(context)
    assert result == "choice_a"
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_select_input_node_empty(context: ProcessingContext):
    """Test SelectInput node with empty value and options."""
    node = SelectInput(
        name="empty_select",
        value="",
        options=[],
        enum_type_name="",
        description="empty select",
    )
    result = await node.process(context)
    assert result == ""
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_select_input_node_basic_fields():
    """Test that SelectInput node exposes value in basic fields but hides options/enum_type_name."""
    basic_fields = SelectInput.get_basic_fields()
    assert "value" in basic_fields
    # options and enum_type_name should not be in basic fields (they're plumbing)
    assert "options" not in basic_fields
    assert "enum_type_name" not in basic_fields


@pytest.mark.asyncio
async def test_select_input_node_return_type():
    """Test that SelectInput return_type is str."""
    assert SelectInput.return_type() == str


@pytest.mark.asyncio
async def test_select_input_node_json_schema():
    """Test SelectInput node generates valid JSON schema."""
    node = SelectInput(
        name="schema_test",
        value="opt1",
        options=["opt1", "opt2"],
        enum_type_name="SchemaTest",
    )
    schema = node.get_json_schema()
    assert isinstance(schema, dict)
    assert "type" in schema
    assert "properties" in schema
