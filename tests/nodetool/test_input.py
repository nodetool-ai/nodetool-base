from unittest.mock import AsyncMock
import pytest
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.metadata.types import (
    AudioRef,
    ImageRef,
    TextRef,
    VideoRef,
    FolderRef,
    AssetRef,
)
from nodetool.nodes.nodetool.input import (
    FloatInput,
    BooleanInput,
    IntegerInput,
    StringInput,
    ChatInput,
    TextInput,
    ImageInput,
    VideoInput,
    AudioInput,
    GroupInput,
)


@pytest.fixture
def context():
    return ProcessingContext(user_id="test", auth_token="test")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "node, input_value, expected_type",
    [
        (
            FloatInput(
                name="float_input",
                value=3.14,
            ),
            3.14,
            float,
        ),
        (
            BooleanInput(
                name="bool_input",
                value=True,
            ),
            True,
            bool,
        ),
        (
            IntegerInput(
                name="int_input",
                value=42,
            ),
            42,
            int,
        ),
        (
            StringInput(
                name="string_input",
                value="test",
            ),
            "test",
            str,
        ),
        (
            ChatInput(
                name="chat_input",
                value=[],
            ),
            [],
            dict,
        ),
        (
            TextInput(
                name="text_input",
                value=TextRef(uri="test.txt"),
            ),
            TextRef(uri="test.txt"),
            TextRef,
        ),
        (
            ImageInput(
                name="image_input",
                value=ImageRef(uri="test.jpg"),
            ),
            ImageRef(uri="test.jpg"),
            ImageRef,
        ),
        (
            VideoInput(
                name="video_input",
                value=VideoRef(uri="test.mp4"),
            ),
            VideoRef(uri="test.mp4"),
            VideoRef,
        ),
        (
            AudioInput(
                name="audio_input",
                value=AudioRef(uri="test.mp3"),
            ),
            AudioRef(uri="test.mp3"),
            AudioRef,
        ),
    ],
)
async def test_input_nodes(
    context: ProcessingContext, node, input_value, expected_type
):
    # For nodes that require setup
    if isinstance(node, GroupInput):
        node._value = input_value

    if isinstance(node, ChatInput):
        with pytest.raises(ValueError):
            await node.process(context)
        return

    try:
        result = await node.process(context)
        assert result == input_value
        assert isinstance(result, expected_type)

    except Exception as e:
        pytest.fail(f"Error processing {node.__class__.__name__}: {str(e)}")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "node_class",
    [
        FloatInput,
        BooleanInput,
        IntegerInput,
        StringInput,
        ChatInput,
        TextInput,
        ImageInput,
        VideoInput,
        AudioInput,
    ],
)
async def test_input_node_json_schema(node_class):
    node = node_class(
        label=f"{node_class.__name__} Label",
        name=f"{node_class.__name__.lower()}_name",
    )
    schema = node.get_json_schema()
    assert isinstance(schema, dict)
    assert "type" in schema
    assert "properties" in schema
