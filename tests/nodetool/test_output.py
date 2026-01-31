import pytest
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.nodes.nodetool.output import Output


@pytest.fixture
def context():
    return ProcessingContext(user_id="test", auth_token="test")


def test_output_instantiation():
    """Test that Output node can be instantiated with default values."""
    node = Output()
    assert node is not None
    # Check that it has the expected fields from OutputNode
    assert hasattr(node, 'name')
    assert hasattr(node, 'value')
    assert hasattr(node, 'description')


def test_output_with_string_value():
    """Test Output node with a string value."""
    node = Output(name="test_output", value="Hello World")
    assert node.name == "test_output"
    assert node.value == "Hello World"


def test_output_with_number_value():
    """Test Output node with a numeric value."""
    node = Output(name="numeric_output", value=42)
    assert node.name == "numeric_output"
    assert node.value == 42


def test_output_with_list_value():
    """Test Output node with a list value."""
    node = Output(name="list_output", value=[1, 2, 3])
    assert node.name == "list_output"
    assert node.value == [1, 2, 3]


def test_output_with_dict_value():
    """Test Output node with a dict value."""
    node = Output(name="dict_output", value={"key": "value"})
    assert node.name == "dict_output"
    assert node.value == {"key": "value"}


def test_output_with_description():
    """Test Output node with a description."""
    node = Output(
        name="described_output",
        value="test",
        description="This is a test output"
    )
    assert node.name == "described_output"
    assert node.description == "This is a test output"
