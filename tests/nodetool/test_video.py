import base64
import pytest
import PIL.Image
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.metadata.types import (
    VideoRef,
    ImageRef,
    AudioRef,
)
from nodetool.nodes.nodetool.video import (
    Fps,
    Concat,
    Trim,
    Rotate,
    ResizeNode,
    SetSpeed,
    Overlay,
    ColorBalance,
    Denoise,
    Sharpness,
    Blur,
    Saturation,
    Reverse,
    Transition,
    AddAudio,
    ChromaKey,
    ExtractFrame,
    GetVideoInfo,
)
from io import BytesIO
import os

test_mp4 = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "test.mp4",
)

with open(
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "test.mp3"), "rb"
) as f:
    dummy_audio = AudioRef(data=f.read())
dummy_video = VideoRef(
    uri=f"data:video/mp4;base64,{base64.b64encode(open(test_mp4, 'rb').read()).decode()}"
)

# Create a dummy ImageRef for testing
buffer = BytesIO()
PIL.Image.new("RGB", (100, 100), color="red").save(buffer, format="PNG")
dummy_image = ImageRef(data=buffer.getvalue())


@pytest.fixture
def context():
    return ProcessingContext(user_id="test", auth_token="test")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "node, expected_type",
    [
        (Fps(video=dummy_video), float),
        (Concat(video_a=dummy_video, video_b=dummy_video), VideoRef),
        (Trim(video=dummy_video, start_time=0, end_time=10), VideoRef),
        (ResizeNode(video=dummy_video, width=640, height=480), VideoRef),
        (Rotate(video=dummy_video, angle=90), VideoRef),
        (SetSpeed(video=dummy_video, speed_factor=1.5), VideoRef),
        (
            Overlay(
                main_video=dummy_video, overlay_video=dummy_video, x=0, y=0, scale=1.0
            ),
            VideoRef,
        ),
        (
            ColorBalance(
                video=dummy_video, red_adjust=1.0, green_adjust=1.0, blue_adjust=1.0
            ),
            VideoRef,
        ),
        (Denoise(video=dummy_video, strength=5.0), VideoRef),
        (Sharpness(video=dummy_video, luma_amount=1.0, chroma_amount=0.5), VideoRef),
        (Blur(video=dummy_video, strength=5.0), VideoRef),
        (Saturation(video=dummy_video, saturation=1.0), VideoRef),
        (Reverse(video=dummy_video), VideoRef),
        (
            Transition(
                video_a=dummy_video,
                video_b=dummy_video,
                transition_type=Transition.TransitionType.fade,
                duration=1.0,
            ),
            VideoRef,
        ),
        (AddAudio(video=dummy_video, audio=dummy_audio), VideoRef),
        (ChromaKey(video=dummy_video), VideoRef),
        (ExtractFrame(video=dummy_video, time=0.0), ImageRef),
    ],
)
async def test_video_nodes(context: ProcessingContext, node, expected_type):
    try:
        result = await node.process(context)
        assert isinstance(result, expected_type)
    except Exception as e:
        pytest.fail(f"Error processing {node.__class__.__name__}: {str(e)}")


@pytest.mark.asyncio
async def test_get_video_info(context: ProcessingContext):
    """Test GetVideoInfo node returns correct metadata structure."""
    node = GetVideoInfo(video=dummy_video)
    result = await node.process(context)
    
    assert isinstance(result, dict)
    assert "duration" in result
    assert "width" in result
    assert "height" in result
    assert "fps" in result
    assert "frame_count" in result
    assert "codec" in result
    assert "has_audio" in result
    
    # Check types
    assert isinstance(result["duration"], float)
    assert isinstance(result["width"], int)
    assert isinstance(result["height"], int)
    assert isinstance(result["fps"], float)
    assert isinstance(result["frame_count"], int)
    assert isinstance(result["codec"], str)
    assert isinstance(result["has_audio"], bool)
