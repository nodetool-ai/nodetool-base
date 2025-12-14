import pytest
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.metadata.types import TextRef
from nodetool.nodes.nodetool.text import (
    Concat,
    Join,
    Template,
    Replace,
    Split,
    Extract,
    Chunk,
    ExtractRegex,
    FindAllRegex,
    ParseJSON,
    ExtractJSON,
    ToString,
)


@pytest.fixture
def context():
    return ProcessingContext(user_id="test", auth_token="test")


# Create dummy inputs for testing
dummy_string = "Hello, world!"
dummy_text = TextRef(data=b"Hello, world!")


class CustomObj:
    def __str__(self):
        return "str_custom"
        
    def __repr__(self):
        return "repr_custom"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "node, expected_type",
    [
        (Concat(a=dummy_string, b=dummy_string), str),
        (Join(strings=[dummy_string, dummy_string], separator=","), str),
        (Template(string=dummy_string, values={"name": "Alice"}), (str,)),
        (Replace(text=dummy_string, old="world", new="universe"), str),
        (Split(text="a,b,c", delimiter=","), list),
        (Extract(text="abcdef", start=1, end=4), (str, TextRef)),
        (Chunk(text="a b c d e", length=2, overlap=0), list),
        (ExtractRegex(text="abc123def", regex=r"\d+"), list),
        (FindAllRegex(text="abc123def456", regex=r"\d+"), list),
        (ParseJSON(text='{"a": 1, "b": 2}'), dict),
        (
            ExtractJSON(text='{"a": {"b": 2}}', json_path="$.a.b"),
            (int, float, str, bool, list, dict),
        ),
        (ToString(value=123), str),
        (ToString(value=123, mode="repr"), str),
    ],
)
async def test_text_nodes(context: ProcessingContext, node, expected_type):
    try:
        result = await node.process(context)
        assert isinstance(result, expected_type)
    except Exception as e:
        pytest.fail(f"Error processing {node.__class__.__name__}: {str(e)}")


@pytest.mark.asyncio
async def test_to_string_custom(context: ProcessingContext):
    # Test str mode
    node = ToString(value=CustomObj(), mode="str")
    result = await node.process(context)
    assert result == "str_custom"

    # Test repr mode
    node = ToString(value=CustomObj(), mode="repr")
    result = await node.process(context)
    assert result == "repr_custom"



@pytest.mark.asyncio
async def test_extract_regex(context: ProcessingContext):
    node = ExtractRegex(text="The year is 2023", regex=r"(\d{4})")
    result = await node.process(context)
    assert result == ["2023"]


@pytest.mark.asyncio
async def test_parse_json(context: ProcessingContext):
    node = ParseJSON(text='{"a": 1, "b": [2, 3]}')
    result = await node.process(context)
    assert result == {"a": 1, "b": [2, 3]}


@pytest.mark.asyncio
async def test_extract_json(context: ProcessingContext):
    node = ExtractJSON(text='{"a": {"b": {"c": 42}}}', json_path="$.a.b.c")
    result = await node.process(context)
    assert result == 42
