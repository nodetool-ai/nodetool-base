import pytest
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.metadata.types import ImageRef
from nodetool.nodes.nodetool.list import (
    Length,
    ListRange,
    Slice,
    SelectElements,
    GetElement,
    Append,
    Extend,
    Dedupe,
    Reverse,
    Randomize,
)

# Create dummy inputs for testing
dummy_list = [1, 2, 3, 4, 5]
dummy_image = ImageRef(data=b"dummy_image_data")


@pytest.fixture
def context():
    return ProcessingContext(user_id="test", auth_token="test")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "node, expected_type",
    [
        (Length(values=dummy_list), int),
        (ListRange(start=0, stop=5, step=1), list),
        (Slice(values=dummy_list, start=1, stop=4, step=1), list),
        (SelectElements(values=dummy_list, indices=[0, 2, 4]), list),
        (GetElement(values=dummy_list, index=2), (int, float, str)),
        (Append(values=dummy_list, value=6), list),
        (Extend(values=dummy_list, other_values=[6, 7]), list),
        (Dedupe(values=[1, 2, 2, 3, 3, 3]), list),
        (Reverse(values=dummy_list), list),
        (Randomize(values=dummy_list), list),
    ],
)
async def test_list_nodes(context: ProcessingContext, node, expected_type):
    try:
        result = await node.process(context)
        assert isinstance(result, expected_type)
    except Exception as e:
        pytest.fail(f"Error processing {node.__class__.__name__}: {str(e)}")


# Additional tests for specific node behaviors


@pytest.mark.asyncio
async def test_length_node(context: ProcessingContext):
    node = Length(values=[1, 2, 3, 4, 5])
    result = await node.process(context)
    assert result == 5


@pytest.mark.asyncio
async def test_generate_sequence_node(context: ProcessingContext):
    node = ListRange(start=0, stop=5, step=2)
    result = await node.process(context)
    assert result == [0, 2, 4]


@pytest.mark.asyncio
async def test_get_element_out_of_range(context: ProcessingContext):
    node = GetElement(values=[1, 2, 3], index=5)
    with pytest.raises(IndexError):
        await node.process(context)


@pytest.mark.asyncio
async def test_randomize_node(context: ProcessingContext):
    input_list = [1, 2, 3, 4, 5]
    node = Randomize(values=input_list)
    result = await node.process(context)

    # Check that the result has the same length
    assert len(result) == len(input_list)

    # Check that the result contains the same elements (order independent)
    assert sorted(result) == sorted(input_list)

    # Check that the original list is not modified (Randomize should return a copy or modify a copy)
    # Looking at implementation:
    # shuffled = self.values.copy()
    # random.shuffle(shuffled)
    # return shuffled
    # So self.values should remain unchanged if it's the same object reference,
    # but `node.values` comes from pydantic field.
    assert node.values == input_list


@pytest.mark.parametrize(
    "NodeClass",
    [
        Length,
        ListRange,
        Slice,
        SelectElements,
        GetElement,
        Append,
        Extend,
        Dedupe,
        Reverse,
        Randomize,
    ],
)
def test_node_attributes(NodeClass):
    node = NodeClass()
    assert hasattr(node, "process")
    assert callable(node.process)
