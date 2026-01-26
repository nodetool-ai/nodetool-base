import pytest
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.metadata.types import (
    AudioRef,
    ImageRef,
    VideoRef,
)
from nodetool.nodes.nodetool.input import (
    FloatInput,
    BooleanInput,
    IntegerInput,
    StringInput,
    ImageInput,
    ImageListInput,
    VideoInput,
    VideoListInput,
    AudioInput,
    AudioListInput,
    TextListInput,
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
                description="test",
            ),
            3.14,
            float,
        ),
        (
            BooleanInput(
                name="bool_input",
                value=True,
                description="test",
            ),
            True,
            bool,
        ),
        (
            IntegerInput(
                name="int_input",
                value=42,
                description="test",
            ),
            42,
            int,
        ),
        (
            StringInput(
                name="string_input",
                value="test",
                description="test",
            ),
            "test",
            str,
        ),
        (
            ImageInput(
                name="image_input",
                value=ImageRef(uri="test.jpg"),
                description="test",
            ),
            ImageRef(uri="test.jpg"),
            ImageRef,
        ),
        (
            VideoInput(
                name="video_input",
                value=VideoRef(uri="test.mp4"),
                description="test",
            ),
            VideoRef(uri="test.mp4"),
            VideoRef,
        ),
        (
            AudioInput(
                name="audio_input",
                value=AudioRef(uri="test.mp3"),
                description="test",
            ),
            AudioRef(uri="test.mp3"),
            AudioRef,
        ),
    ],
)
async def test_input_nodes(
    context: ProcessingContext, node, input_value, expected_type
):
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
        ImageInput,
        ImageListInput,
        VideoInput,
        VideoListInput,
        AudioInput,
        AudioListInput,
        TextListInput,
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


@pytest.mark.asyncio
async def test_string_input_max_length_enforced(context: ProcessingContext):
    node = StringInput(
        name="string_input",
        value="hello world",
        max_length=5,
        description="test",
    )
    result = await node.process(context)
    assert result == "hello"


@pytest.mark.asyncio
async def test_string_input_max_length_zero_unlimited(context: ProcessingContext):
    node = StringInput(
        name="string_input",
        value="hello world",
        max_length=0,
        description="test",
    )
    result = await node.process(context)
    assert result == "hello world"
