"""Tests for Kie.ai image generation nodes."""

import pytest
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

from nodetool.workflows.processing_context import ProcessingContext
from nodetool.nodes.kie.image import (
    KieBaseNode,
    Flux2ProTextToImage,
    Flux2ProImageToImage,
    Flux2FlexTextToImage,
    Flux2FlexImageToImage,
    Seedream45TextToImage,
    Seedream45Edit,
    ZImage,
    NanoBanana,
    NanoBananaPro,
    NanoBananaEdit,
    FluxKontext,
    GrokImagineTextToImage,
    TopazImageUpscale,
    GPTImage4oTextToImage,
    GPTImage4oImageToImage,
    GPTImage15TextToImage,
    GPTImage15ImageToImage,
    IdeogramV3TextToImage,
    IdeogramV3ImageToImage,
    Seedream40TextToImage,
    Seedream40ImageToImage,
)
from nodetool.nodes.kie.video import (
    Sora2TextToVideo,
    Sora2ProTextToVideo,
    Sora2ProImageToVideo,
    Sora2ProStoryboard,
    SeedanceV1LiteTextToVideo,
    SeedanceV1ProTextToVideo,
    SeedanceV1LiteImageToVideo,
    SeedanceV1ProImageToVideo,
    SeedanceV1ProFastImageToVideo,
    HailuoImageToVideoPro,
    HailuoImageToVideoStandard,
    KlingTextToVideo,
    KlingImageToVideo,
    KlingAIAvatarStandard,
    KlingMotionControl,
    TopazVideoUpscale,
    GrokImagineImageToVideo,
    GrokImagineTextToVideo,
    Veo31TextToVideo,
    Veo31ImageToVideo,
    Veo31ReferenceToVideo,
    Kling21TextToVideo,
    Kling21ImageToVideo,
    Wan25TextToVideo,
    Wan25ImageToVideo,
    WanAnimate,
    WanSpeechToVideo,
    Wan22TextToVideo,
    Wan22ImageToVideo,
    Hailuo02TextToVideo,
    Hailuo02ImageToVideo,
    Sora2WatermarkRemover,
)
from nodetool.nodes.kie.audio import (
    GenerateMusic,
    ElevenLabsAudioIsolation,
    ElevenLabsSoundEffect,
    ElevenLabsSpeechToText,
    ElevenLabsV3Dialogue,
)
from nodetool.metadata.types import VideoRef


class MockResponse:
    """Mock aiohttp response."""

    def __init__(
        self,
        json_data: dict | None = None,
        content: bytes = b"",
        status: int = 200,
        content_type: str = "application/json",
    ):
        self._json_data = json_data or {}
        self._content = content
        self.status = status
        self.headers = {"Content-Type": content_type}

    async def json(self):
        return self._json_data

    async def text(self):
        return str(self._json_data)

    async def read(self):
        return self._content

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class MockSession:
    """Mock aiohttp session."""

    def __init__(self, responses: list[MockResponse]):
        self._responses = responses
        self._call_index = 0

    def _get_next_response(self):
        if self._call_index < len(self._responses):
            response = self._responses[self._call_index]
            self._call_index += 1
            return response
        return MockResponse(status=500)

    def post(self, url, **kwargs):
        return self._get_next_response()

    def get(self, url, **kwargs):
        return self._get_next_response()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


@pytest.fixture
def mock_context():
    """Create a mock ProcessingContext."""
    ctx = MagicMock(spec=ProcessingContext)
    ctx.get_secret = AsyncMock(return_value="test-api-key")
    ctx.image_from_bytes = AsyncMock(return_value=MagicMock())
    ctx.asset_to_io = AsyncMock(return_value=MagicMock(name="test.jpg"))
    return ctx


class TestKieBaseNode:
    """Tests for KieBaseNode base class."""

    def test_is_not_visible(self):
        """KieBaseNode should not be visible in UI."""
        assert not KieBaseNode.is_visible()

    def test_subclass_is_visible(self):
        """Subclasses should be visible."""
        # Image generation nodes
        assert Flux2ProTextToImage.is_visible()
        assert Flux2ProImageToImage.is_visible()
        assert Flux2FlexTextToImage.is_visible()
        assert Flux2FlexImageToImage.is_visible()
        assert Seedream45TextToImage.is_visible()
        assert Seedream45Edit.is_visible()
        assert ZImage.is_visible()
        assert NanoBanana.is_visible()
        assert NanoBananaPro.is_visible()
        assert FluxKontext.is_visible()
        assert GrokImagineTextToImage.is_visible()
        assert TopazImageUpscale.is_visible()
        # Video generation nodes
        assert Sora2TextToVideo.is_visible()
        assert Sora2ProTextToVideo.is_visible()
        assert Sora2ProImageToVideo.is_visible()
        assert Sora2ProStoryboard.is_visible()
        assert SeedanceV1LiteTextToVideo.is_visible()
        assert SeedanceV1ProTextToVideo.is_visible()
        assert SeedanceV1LiteImageToVideo.is_visible()
        assert SeedanceV1ProImageToVideo.is_visible()
        assert SeedanceV1ProFastImageToVideo.is_visible()
        assert HailuoImageToVideoPro.is_visible()
        assert HailuoImageToVideoStandard.is_visible()
        assert KlingTextToVideo.is_visible()
        assert KlingImageToVideo.is_visible()
        assert KlingAIAvatarStandard.is_visible()
        assert KlingMotionControl.is_visible()
        assert TopazVideoUpscale.is_visible()
        assert GrokImagineImageToVideo.is_visible()
        assert GrokImagineTextToVideo.is_visible()
        # Audio generation nodes
        assert GenerateMusic.is_visible()
        # New image nodes
        assert GPTImage4oTextToImage.is_visible()
        assert GPTImage4oImageToImage.is_visible()
        assert GPTImage15TextToImage.is_visible()
        assert GPTImage15ImageToImage.is_visible()
        assert IdeogramV3TextToImage.is_visible()
        assert IdeogramV3ImageToImage.is_visible()
        assert Seedream40TextToImage.is_visible()
        assert Seedream40ImageToImage.is_visible()
        # New video nodes
        assert Kling21TextToVideo.is_visible()
        assert Kling21ImageToVideo.is_visible()
        assert Wan25TextToVideo.is_visible()
        assert Wan25ImageToVideo.is_visible()
        assert WanAnimate.is_visible()
        assert WanSpeechToVideo.is_visible()
        assert Wan22TextToVideo.is_visible()
        assert Wan22ImageToVideo.is_visible()
        assert Hailuo02TextToVideo.is_visible()
        assert Hailuo02ImageToVideo.is_visible()
        assert Sora2WatermarkRemover.is_visible()
        # New audio nodes
        assert ElevenLabsAudioIsolation.is_visible()
        assert ElevenLabsSoundEffect.is_visible()
        assert ElevenLabsSpeechToText.is_visible()
        assert ElevenLabsV3Dialogue.is_visible()

    @pytest.mark.asyncio
    async def test_upload_video_transcodes_non_mp4(self, mock_context):
        node = KlingMotionControl()
        video_ref = VideoRef(uri="file:///tmp/input.mov", format="mov")
        mock_context.asset_to_io = AsyncMock(return_value=BytesIO(b"mov-bytes"))

        with patch.object(
            node, "_convert_video_to_mp4", AsyncMock(return_value=b"mp4-bytes")
        ) as mock_convert, patch.object(
            node, "_upload_asset", AsyncMock(return_value="https://example.com/video.mp4")
        ) as mock_upload:
            result = await node._upload_video(mock_context, video_ref)

        assert result == "https://example.com/video.mp4"
        mock_convert.assert_awaited_once()
        call_args = mock_upload.await_args.args
        assert call_args[2] == "videos/user-uploads"
        assert isinstance(call_args[1], VideoRef)
        assert call_args[1].data == b"mp4-bytes"


class TestFlux2ProTextToImage:
    """Tests for Flux2ProTextToImage node."""

    @pytest.mark.asyncio
    async def test_submit_payload(self, mock_context):
        """Test that submit payload is correctly generated."""
        node = Flux2ProTextToImage(
            prompt="a cat", aspect_ratio=Flux2ProTextToImage.AspectRatio.LANDSCAPE
        )
        payload = await node._get_submit_payload()
        assert payload == {
            "model": "flux-2/pro-text-to-image",
            "input": {
                "prompt": "a cat",
                "aspect_ratio": "16:9",
                "resolution": "1K",
                "steps": 25,
                "guidance_scale": 7.5,
            },
        }

    @pytest.mark.asyncio
    async def test_empty_prompt_raises_error(self, mock_context):
        """Test that empty prompt raises ValueError."""
        node = Flux2ProTextToImage(prompt="")
        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            await node._get_submit_payload()

    @pytest.mark.asyncio
    async def test_model_and_params(self):
        """Test model name and input parameters."""
        node = Flux2ProTextToImage(prompt="test")
        assert node._get_model() == "flux-2/pro-text-to-image"
        params = await node._get_input_params()
        assert params["prompt"] == "test"
        assert params["aspect_ratio"] == "1:1"

    @pytest.mark.asyncio
    async def test_process_success(self, mock_context):
        """Test successful image generation."""
        node = Flux2ProTextToImage(
            prompt="a beautiful sunset",
        )
        node._poll_interval = 0.5
        node._max_poll_attempts = 5

        # Mock responses:
        # 1. Submit: {code: 200, message: "success", data: {taskId: "task123"}}
        # 2. Status (completed): {code: 200, message: "success", data: {state: "success", resultJson: "{\"resultUrls\":[\"https://example.com/img.png\"]}"}}
        # 3. Download (get status again): same as #2
        # 4. Download image: actual image bytes
        responses = [
            MockResponse(
                json_data={
                    "code": 200,
                    "message": "success",
                    "data": {"taskId": "task123"},
                }
            ),
            MockResponse(
                json_data={
                    "code": 200,
                    "message": "success",
                    "data": {
                        "state": "success",
                        "resultJson": '{"resultUrls":["https://example.com/img.png"]}',
                    },
                }
            ),
            MockResponse(
                json_data={
                    "code": 200,
                    "message": "success",
                    "data": {
                        "state": "success",
                        "resultJson": '{"resultUrls":["https://example.com/img.png"]}',
                    },
                }
            ),
            MockResponse(content=b"image_bytes", content_type="image/png"),
        ]

        with patch("aiohttp.ClientSession", return_value=MockSession(responses)):
            await node.process(mock_context)

        mock_context.image_from_bytes.assert_called_once_with(
            b"image_bytes", metadata={"task_id": "task123"}
        )

    @pytest.mark.asyncio
    async def test_missing_api_key(self, mock_context):
        """Test that missing API key raises error."""
        mock_context.get_secret = AsyncMock(return_value=None)
        node = Flux2ProTextToImage(prompt="test")

        with pytest.raises(ValueError, match="KIE_API_KEY secret is not configured"):
            await node._get_api_key(mock_context)


class TestSeedream45TextToImage:
    """Tests for Seedream45TextToImage node."""

    @pytest.mark.asyncio
    async def test_model_and_params(self):
        """Test model name and input parameters."""
        node = Seedream45TextToImage(prompt="test")
        assert node._get_model() == "seedream/4.5-text-to-image"
        params = await node._get_input_params()
        assert params["prompt"] == "test"
        assert params["aspect_ratio"] == "1:1"

    @pytest.mark.asyncio
    async def test_submit_payload(self):
        """Test submit payload generation."""
        node = Seedream45TextToImage(
            prompt="artistic scene",
            aspect_ratio=Seedream45TextToImage.AspectRatio.PORTRAIT,
        )
        payload = await node._get_submit_payload()
        assert payload == {
            "model": "seedream/4.5-text-to-image",
            "input": {
                "prompt": "artistic scene",
                "aspect_ratio": "9:16",
                "quality": "basic",
            },
        }


class TestZImage:
    """Tests for ZImage node."""

    @pytest.mark.asyncio
    async def test_model_and_params(self):
        """Test model name and input parameters."""
        node = ZImage(prompt="test")
        assert node._get_model() == "z-image"
        params = await node._get_input_params()
        assert params["prompt"] == "test"
        assert params["aspect_ratio"] == "1:1"


class TestNanoBanana:
    """Tests for NanoBanana node."""

    @pytest.mark.asyncio
    async def test_model_and_params(self):
        """Test model name and input parameters."""
        node = NanoBanana(prompt="test")
        assert node._get_model() == "google/nano-banana"
        params = await node._get_input_params()
        assert params["prompt"] == "test"
        assert params["image_size"] == "1:1"


class TestFlux2Pro:
    """Tests for Flux2Pro node."""

    @pytest.mark.asyncio
    async def test_model_and_params(self):
        """Test model name and input parameters."""
        node = Flux2ProTextToImage(prompt="test")
        assert node._get_model() == "flux-2/pro-text-to-image"
        params = await node._get_input_params()
        assert params["prompt"] == "test"
        assert params["aspect_ratio"] == "1:1"

    @pytest.mark.asyncio
    async def test_submit_payload_with_extra_params(self):
        """Test submit payload includes steps and guidance_scale."""
        node = Flux2ProTextToImage(
            prompt="detailed artwork",
            aspect_ratio=Flux2ProTextToImage.AspectRatio.WIDE,
            steps=30,
            guidance_scale=10.0,
        )
        payload = await node._get_submit_payload()
        assert payload == {
            "model": "flux-2/pro-text-to-image",
            "input": {
                "prompt": "detailed artwork",
                "aspect_ratio": "4:3",
                "resolution": "1K",
                "steps": 30,
                "guidance_scale": 10.0,
            },
        }


class TestKiePollingLogic:
    """Tests for the polling logic in KieBaseNode."""

    @pytest.mark.asyncio
    async def test_task_status_extraction(self):
        """Test various status extraction scenarios."""
        node = Flux2ProTextToImage(prompt="test")

        # State is in data.state field
        assert node._is_task_complete({"data": {"state": "success"}})
        assert not node._is_task_complete({"data": {"state": "processing"}})
        assert not node._is_task_complete({"data": {"state": "pending"}})

    @pytest.mark.asyncio
    async def test_task_failure_detection(self):
        """Test task failure detection."""
        node = Flux2ProTextToImage(prompt="test")

        assert node._is_task_failed({"data": {"state": "failed"}})
        assert not node._is_task_failed({"data": {"state": "processing"}})
        assert not node._is_task_failed({"data": {"state": "success"}})

    @pytest.mark.asyncio
    async def test_task_id_extraction(self):
        """Test task ID extraction from response format.

        Expected format: {code: 200, message: "success", data: {taskId: "task123"}}
        """
        node = Flux2ProTextToImage(prompt="test")

        assert node._extract_task_id({"data": {"taskId": "abc123"}}) == "abc123"

        with pytest.raises(ValueError, match="Could not extract taskId"):
            node._extract_task_id({"other_field": "value"})

    @pytest.mark.asyncio
    async def test_headers_generation(self):
        """Test API headers generation."""
        node = Flux2ProTextToImage(prompt="test")
        headers = node._get_headers("my-api-key")

        assert headers["Authorization"] == "Bearer my-api-key"
        assert headers["Content-Type"] == "application/json"

    @pytest.mark.asyncio
    async def test_poll_timeout(self, mock_context):
        """Test that polling times out after max attempts."""
        node = Flux2ProTextToImage(prompt="test")
        node._poll_interval = 0.5
        node._max_poll_attempts = 3

        # Submit succeeds, but status always returns processing
        responses = [
            MockResponse(json_data={"data": {"taskId": "task123"}}),
            MockResponse(json_data={"data": {"state": "processing"}}),
            MockResponse(json_data={"data": {"state": "processing"}}),
            MockResponse(json_data={"data": {"state": "processing"}}),
        ]

        with patch("aiohttp.ClientSession", return_value=MockSession(responses)):
            with pytest.raises(TimeoutError, match="did not complete"):
                await node.process(mock_context)

    @pytest.mark.asyncio
    async def test_task_failure_handling(self, mock_context):
        """Test that task failure is properly handled."""
        node = Flux2ProTextToImage(prompt="test")
        node._poll_interval = 0.5

        responses = [
            MockResponse(json_data={"data": {"taskId": "task123"}}),
            MockResponse(
                json_data={
                    "data": {"state": "failed", "failMsg": "Content policy violation"}
                }
            ),
        ]

        with patch("aiohttp.ClientSession", return_value=MockSession(responses)):
            with pytest.raises(
                ValueError, match="Task failed: Content policy violation"
            ):
                await node.process(mock_context)

    @pytest.mark.asyncio
    async def test_download_result_from_status_response(self, mock_context):
        """Test that result is downloaded from the status response's resultJson."""
        node = Flux2ProTextToImage(prompt="test")
        node._poll_interval = 0.5

        # Submit, status (completed), status again (for download), then actual image
        responses = [
            MockResponse(json_data={"data": {"taskId": "task123"}}),
            MockResponse(
                json_data={
                    "data": {
                        "state": "success",
                        "resultJson": '{"resultUrls":["https://example.com/img.png"]}',
                    }
                }
            ),
            MockResponse(
                json_data={
                    "data": {
                        "state": "success",
                        "resultJson": '{"resultUrls":["https://example.com/img.png"]}',
                    }
                }
            ),
            MockResponse(content=b"actual_image_bytes", content_type="image/png"),
        ]

        with patch("aiohttp.ClientSession", return_value=MockSession(responses)):
            await node.process(mock_context)

        mock_context.image_from_bytes.assert_called_once_with(
            b"actual_image_bytes", metadata={"task_id": "task123"}
        )


class TestNanoBananaProGenerate:
    """Tests for NanoBananaProGenerate node."""

    @pytest.mark.asyncio
    async def test_model_and_params(self):
        """Test model name and input parameters."""
        node = NanoBananaPro(prompt="test")
        assert node._get_model() == "nano-banana-pro"
        params = await node._get_input_params()
        assert params["prompt"] == "test"
        assert params["aspect_ratio"] == "1:1"

    @pytest.mark.asyncio
    async def test_image_input_parameter_name(self):
        """Test image inputs use 'image_urls' key, consistent with NanoBananaEdit."""
        from nodetool.metadata.types import ImageRef
        from unittest.mock import AsyncMock

        # Create mock context
        mock_context = AsyncMock(spec=ProcessingContext)

        # Create test image refs
        img1 = ImageRef(uri="http://example.com/test1.png")
        img2 = ImageRef(uri="http://example.com/test2.png")

        # Create node with image inputs
        node = NanoBananaPro(
            prompt="test prompt",
            image_input=[img1, img2]
        )

        # Mock the _upload_image method to return test URLs
        async def mock_upload(ctx, img):
            return img.uri

        node._upload_image = mock_upload

        # Get parameters
        params = await node._get_input_params(mock_context)

        # Verify 'image_urls' key is used (not 'image_input')
        assert "image_urls" in params
        assert "image_input" not in params
        assert params["image_urls"] == ["http://example.com/test1.png", "http://example.com/test2.png"]


class TestNanoBananaEdit:
    """Tests for NanoBananaEdit node."""

    @pytest.mark.asyncio
    async def test_model_and_params(self):
        """Test model name and input parameters."""
        node = NanoBananaEdit(prompt="test")
        assert node._get_model() == "google/nano-banana-edit"
        params = await node._get_input_params()
        assert params["prompt"] == "test"
        assert params["image_size"] == "1:1"

    @pytest.mark.asyncio
    async def test_image_input_parameter_name(self):
        """Test image inputs use 'image_urls' key, consistent with NanoBananaPro."""
        from nodetool.metadata.types import ImageRef
        from unittest.mock import AsyncMock

        # Create mock context
        mock_context = AsyncMock(spec=ProcessingContext)

        # Create test image refs
        img1 = ImageRef(uri="http://example.com/test1.png")
        img2 = ImageRef(uri="http://example.com/test2.png")

        # Create node with image inputs
        node = NanoBananaEdit(
            prompt="test prompt",
            image_input=[img1, img2]
        )

        # Mock the _upload_image method to return test URLs
        async def mock_upload(ctx, img):
            return img.uri

        node._upload_image = mock_upload

        # Get parameters
        params = await node._get_input_params(mock_context)

        # Verify 'image_urls' key is used
        assert "image_urls" in params
        assert "image_input" not in params
        assert params["image_urls"] == ["http://example.com/test1.png", "http://example.com/test2.png"]


class TestFluxKontext:
    """Tests for FluxKontext node."""

    @pytest.mark.asyncio
    async def test_model_and_params(self):
        """Test model name and input parameters."""
        node = FluxKontext(prompt="test")
        assert node._get_model() == "flux-kontext"
        params = await node._get_input_params()
        assert params["prompt"] == "test"
        assert params["aspect_ratio"] == "1:1"
        assert params["mode"] == "pro"

    @pytest.mark.asyncio
    async def test_submit_payload_with_mode(self):
        """Test submit payload includes mode."""
        node = FluxKontext(
            prompt="artwork",
            mode=FluxKontext.Mode.MAX,
        )
        payload = await node._get_submit_payload()
        assert payload["model"] == "flux-kontext"
        assert payload["input"]["mode"] == "max"


class TestGrokImagineTextToImage:
    """Tests for GrokImagineTextToImage node."""

    @pytest.mark.asyncio
    async def test_model_and_params(self):
        """Test model name and input parameters."""
        node = GrokImagineTextToImage(prompt="test")
        assert node._get_model() == "grok-imagine/text-to-image"
        params = await node._get_input_params()
        assert params["prompt"] == "test"
        assert params["aspect_ratio"] == "1:1"


class TestSora2ProTextToVideo:
    """Tests for Sora2ProTextToVideo node."""

    @pytest.mark.asyncio
    async def test_model_and_params(self):
        """Test model name and input parameters."""
        node = Sora2ProTextToVideo(prompt="test")
        assert node._get_model() == "sora-2-pro-text-to-video"
        params = await node._get_input_params()
        assert params["prompt"] == "test"
        assert params["aspect_ratio"] == "landscape"
        assert params["n_frames"] == 10
        assert params["remove_watermark"] is True


class TestSora2ProImageToVideo:
    """Tests for Sora2ProImageToVideo node."""

    @pytest.mark.asyncio
    async def test_model_and_params(self, mock_context):
        """Test model name and input parameters."""
        from nodetool.metadata.types import ImageRef

        node = Sora2ProImageToVideo(
            image=ImageRef(uri="http://example.com/image.jpg"), prompt="test"
        )
        assert node._get_model() == "sora-2-pro-image-to-video"
        with patch.object(
            node, "_upload_image", return_value="http://uploaded-url.com/image.jpg"
        ):
            params = await node._get_input_params(mock_context)
        assert params["prompt"] == "test"
        assert params["image_url"] == "http://uploaded-url.com/image.jpg"
        assert params["n_frames"] == 10
        assert params["remove_watermark"] is True


class TestSora2ProStoryboard:
    """Tests for Sora2ProStoryboard node."""

    @pytest.mark.asyncio
    async def test_model_and_params(self):
        """Test model name and input parameters."""
        node = Sora2ProStoryboard(prompt="test storyboard")
        assert node._get_model() == "sora-2-pro-story-board"
        assert node.aspect_ratio.value == "landscape"
        assert node.n_frames == 10
        assert node.remove_watermark is True


class TestSora2TextToVideo:
    """Tests for Sora2TextToVideo node."""

    @pytest.mark.asyncio
    async def test_model_and_params(self):
        """Test model name and input parameters."""
        node = Sora2TextToVideo(prompt="test")
        assert node._get_model() == "sora-2-text-to-video"
        params = await node._get_input_params()
        assert params["prompt"] == "test"
        assert params["aspect_ratio"] == "landscape"
        assert params["n_frames"] == 10
        assert params["remove_watermark"] is True


class TestSeedanceV1LiteTextToVideo:
    """Tests for SeedanceV1LiteTextToVideo node."""

    @pytest.mark.asyncio
    async def test_model_and_params(self):
        """Test model name and input parameters."""
        node = SeedanceV1LiteTextToVideo(prompt="test")
        assert node._get_model() == "seedance/v1-lite-text-to-video"
        params = await node._get_input_params()
        assert params["prompt"] == "test"
        assert params["aspect_ratio"] == "16:9"
        assert params["resolution"] == "720p"
        assert params["duration"] == "5"
        assert params["remove_watermark"] is True


class TestSeedanceV1ProTextToVideo:
    """Tests for SeedanceV1ProTextToVideo node."""

    @pytest.mark.asyncio
    async def test_model_and_params(self):
        """Test model name and input parameters."""
        node = SeedanceV1ProTextToVideo(prompt="test")
        assert node._get_model() == "seedance/v1-pro-text-to-video"
        params = await node._get_input_params()
        assert params["prompt"] == "test"
        assert params["aspect_ratio"] == "16:9"
        assert params["resolution"] == "720p"
        assert params["duration"] == "5"
        assert params["remove_watermark"] is True


class TestSeedanceV1LiteImageToVideo:
    """Tests for SeedanceV1LiteImageToVideo node."""

    @pytest.mark.asyncio
    async def test_model_and_params(self, mock_context):
        """Test model name and input parameters."""
        from nodetool.metadata.types import ImageRef

        node = SeedanceV1LiteImageToVideo(
            image1=ImageRef(uri="http://example.com/image.jpg"), prompt="test"
        )
        assert node._get_model() == "seedance/v1-lite-image-to-video"
        with patch.object(
            node, "_upload_image", return_value="http://uploaded-url.com/image.jpg"
        ):
            params = await node._get_input_params(mock_context)
        assert params["prompt"] == "test"
        assert params["image_urls"] == ["http://uploaded-url.com/image.jpg"]
        assert params["aspect_ratio"] == "16:9"
        assert params["resolution"] == "720p"
        assert params["duration"] == "5"
        assert params["remove_watermark"] is True


class TestSeedanceV1ProImageToVideo:
    """Tests for SeedanceV1ProImageToVideo node."""

    @pytest.mark.asyncio
    async def test_model_and_params(self, mock_context):
        """Test model name and input parameters."""
        from nodetool.metadata.types import ImageRef

        node = SeedanceV1ProImageToVideo(
            image1=ImageRef(uri="http://example.com/image.jpg"), prompt="test"
        )
        assert node._get_model() == "seedance/v1-pro-image-to-video"
        with patch.object(
            node, "_upload_image", return_value="http://uploaded-url.com/image.jpg"
        ):
            params = await node._get_input_params(mock_context)
        assert params["prompt"] == "test"
        assert params["image_urls"] == ["http://uploaded-url.com/image.jpg"]
        assert params["aspect_ratio"] == "16:9"
        assert params["resolution"] == "720p"
        assert params["duration"] == "5"
        assert params["remove_watermark"] is True


class TestSeedanceV1ProFastImageToVideo:
    """Tests for SeedanceV1ProFastImageToVideo node."""

    @pytest.mark.asyncio
    async def test_model_and_params(self, mock_context):
        """Test model name and input parameters."""
        from nodetool.metadata.types import ImageRef

        node = SeedanceV1ProFastImageToVideo(
            image1=ImageRef(uri="http://example.com/image.jpg"), prompt="test"
        )
        assert node._get_model() == "seedance/v1-pro-fast-image-to-video"
        with patch.object(
            node, "_upload_image", return_value="http://uploaded-url.com/image.jpg"
        ):
            params = await node._get_input_params(mock_context)
        assert params["image_urls"] == ["http://uploaded-url.com/image.jpg"]
        assert params["aspect_ratio"] == "16:9"
        assert params["resolution"] == "720p"
        assert params["duration"] == "5"
        assert params["remove_watermark"] is True


class TestHailuoImageToVideoPro:
    """Tests for HailuoImageToVideoPro node."""

    @pytest.mark.asyncio
    async def test_model_and_params(self, mock_context):
        """Test model name and input parameters."""
        from nodetool.metadata.types import ImageRef

        node = HailuoImageToVideoPro(
            image=ImageRef(uri="http://example.com/image.jpg"), prompt="test"
        )
        assert node._get_model() == "hailuo/2-3-image-to-video-pro"
        with patch.object(
            node, "_upload_image", return_value="http://uploaded-url.com/image.jpg"
        ):
            params = await node._get_input_params(mock_context)
        assert params["prompt"] == "test"
        assert params["image_url"] == "http://uploaded-url.com/image.jpg"
        assert params["resolution"] == "768P"


class TestHailuoImageToVideoStandard:
    """Tests for HailuoImageToVideoStandard node."""

    @pytest.mark.asyncio
    async def test_model_and_params(self, mock_context):
        """Test model name and input parameters."""
        from nodetool.metadata.types import ImageRef

        node = HailuoImageToVideoStandard(
            image=ImageRef(uri="http://example.com/image.jpg"), prompt="test"
        )
        assert node._get_model() == "hailuo/2-3-image-to-video-standard"
        with patch.object(
            node, "_upload_image", return_value="http://uploaded-url.com/image.jpg"
        ):
            params = await node._get_input_params(mock_context)
        assert params["prompt"] == "test"
        assert params["image_url"] == "http://uploaded-url.com/image.jpg"
        assert params["resolution"] == "768P"


class TestVeo31TextToVideo:
    """Tests for Veo31TextToVideo node."""

    @pytest.mark.asyncio
    async def test_model_name_veo3(self):
        """Test model name for veo3."""
        node = Veo31TextToVideo(prompt="test", model=Veo31TextToVideo.Model.VEO3)
        assert node._get_model() == "google/veo3"

    @pytest.mark.asyncio
    async def test_model_name_veo3_fast(self):
        """Test model name for veo3_fast."""
        node = Veo31TextToVideo(prompt="test", model=Veo31TextToVideo.Model.VEO3_FAST)
        assert node._get_model() == "google/veo3_fast"

    @pytest.mark.asyncio
    async def test_input_params_basic(self):
        """Test input parameters for text-to-video."""
        node = Veo31TextToVideo(prompt="A dog playing in a park")
        params = await node._get_input_params()
        assert params["prompt"] == "A dog playing in a park"
        assert params["model"] == "veo3_fast"
        assert params["generationType"] == "TEXT_2_VIDEO"
        assert params["aspectRatio"] == "16:9"
        assert params["enableTranslation"] is True

    @pytest.mark.asyncio
    async def test_input_params_with_seed(self):
        """Test input parameters with custom seed."""
        node = Veo31TextToVideo(prompt="test", seed=12345)
        params = await node._get_input_params()
        assert params["seeds"] == 12345

    @pytest.mark.asyncio
    async def test_input_params_with_watermark(self):
        """Test input parameters with watermark."""
        node = Veo31TextToVideo(prompt="test", watermark="MyBrand")
        params = await node._get_input_params()
        assert params["watermark"] == "MyBrand"

    @pytest.mark.asyncio
    async def test_input_params_aspect_ratio(self):
        """Test input parameters with different aspect ratios."""
        node = Veo31TextToVideo(
            prompt="test", aspect_ratio=Veo31TextToVideo.AspectRatio.RATIO_9_16
        )
        params = await node._get_input_params()
        assert params["aspectRatio"] == "9:16"

    @pytest.mark.asyncio
    async def test_disabled_translation(self):
        """Test input parameters with translation disabled."""
        node = Veo31TextToVideo(prompt="test", enable_translation=False)
        params = await node._get_input_params()
        assert params["enableTranslation"] is False

    @pytest.mark.asyncio
    async def test_empty_prompt_raises_error(self):
        """Test that empty prompt raises ValueError."""
        node = Veo31TextToVideo(prompt="")
        with pytest.raises(ValueError, match="Prompt is required"):
            await node._get_input_params()


class TestVeo31ImageToVideo:
    """Tests for Veo31ImageToVideo node."""

    @pytest.mark.asyncio
    async def test_model_name(self):
        """Test model name."""
        node = Veo31ImageToVideo(prompt="test")
        assert node._get_model() == "google/veo3_fast"

    @pytest.mark.asyncio
    async def test_input_params_single_image(self, mock_context):
        """Test input parameters with single image."""
        from nodetool.metadata.types import ImageRef

        node = Veo31ImageToVideo(
            prompt="Animate this image",
            image1=ImageRef(uri="http://example.com/image1.jpg"),
        )
        with patch.object(
            node, "_upload_image", return_value="http://uploaded-url.com/image1.jpg"
        ):
            params = await node._get_input_params(mock_context)
        assert params["prompt"] == "Animate this image"
        assert params["imageUrls"] == ["http://uploaded-url.com/image1.jpg"]
        assert params["model"] == "veo3_fast"
        assert params["generationType"] == "FIRST_AND_LAST_FRAMES_2_VIDEO"
        assert params["aspectRatio"] == "16:9"

    @pytest.mark.asyncio
    async def test_input_params_two_images(self, mock_context):
        """Test input parameters with two images (first and last frame)."""
        from nodetool.metadata.types import ImageRef

        node = Veo31ImageToVideo(
            prompt="Transition between images",
            image1=ImageRef(uri="http://example.com/image1.jpg"),
            image2=ImageRef(uri="http://example.com/image2.jpg"),
        )

        with patch.object(
            node, "_upload_image", return_value="http://uploaded-url.com/image.jpg"
        ):
            params = await node._get_input_params(mock_context)
        assert len(params["imageUrls"]) == 2

    @pytest.mark.asyncio
    async def test_requires_at_least_one_image(self, mock_context):
        """Test that at least one image is required."""
        from nodetool.metadata.types import ImageRef

        node = Veo31ImageToVideo(prompt="test", image1=ImageRef())
        with pytest.raises(ValueError, match="At least one image is required"):
            await node._get_input_params(mock_context)


class TestVeo31ReferenceToVideo:
    """Tests for Veo31ReferenceToVideo node."""

    @pytest.mark.asyncio
    async def test_model_name(self):
        """Test model name is always veo3_fast."""
        node = Veo31ReferenceToVideo(prompt="test")
        assert node._get_model() == "google/veo3_fast"

    @pytest.mark.asyncio
    async def test_input_params_single_image(self, mock_context):
        """Test input parameters with single reference image."""
        from nodetool.metadata.types import ImageRef

        node = Veo31ReferenceToVideo(
            prompt="Generate video from material",
            image1=ImageRef(uri="http://example.com/material1.jpg"),
        )
        with patch.object(
            node, "_upload_image", return_value="http://uploaded-url.com/material1.jpg"
        ):
            params = await node._get_input_params(mock_context)
        assert params["prompt"] == "Generate video from material"
        assert params["imageUrls"] == ["http://uploaded-url.com/material1.jpg"]
        assert params["model"] == "veo3_fast"
        assert params["generationType"] == "REFERENCE_2_VIDEO"
        assert params["aspectRatio"] == "16:9"

    @pytest.mark.asyncio
    async def test_input_params_multiple_images(self, mock_context):
        """Test input parameters with multiple reference images."""
        from nodetool.metadata.types import ImageRef

        node = Veo31ReferenceToVideo(
            prompt="Generate video from multiple materials",
            image1=ImageRef(uri="http://example.com/material1.jpg"),
            image2=ImageRef(uri="http://example.com/material2.jpg"),
            image3=ImageRef(uri="http://example.com/material3.jpg"),
        )

        with patch.object(
            node, "_upload_image", return_value="http://uploaded-url.com/material.jpg"
        ):
            params = await node._get_input_params(mock_context)
        assert len(params["imageUrls"]) == 3

    @pytest.mark.asyncio
    async def test_requires_at_least_one_image(self, mock_context):
        """Test that at least one reference image is required."""
        from nodetool.metadata.types import ImageRef

        node = Veo31ReferenceToVideo(prompt="test", image1=ImageRef())
        with pytest.raises(
            ValueError, match="At least one reference image is required"
        ):
            await node._get_input_params(mock_context)

    @pytest.mark.asyncio
    async def test_requires_16_9_aspect_ratio(self, mock_context):
        """Test that only 16:9 aspect ratio is supported."""
        from nodetool.metadata.types import ImageRef

        node = Veo31ReferenceToVideo(
            prompt="test",
            image1=ImageRef(uri="http://example.com/image.jpg"),
            aspect_ratio=Veo31ReferenceToVideo.AspectRatio.RATIO_9_16,
        )
        with pytest.raises(
            ValueError, match="REFERENCE_2_VIDEO mode only supports 16:9 aspect ratio"
        ):
            await node._get_input_params(mock_context)

    @pytest.mark.asyncio
    async def test_max_three_images(self, mock_context):
        """Test that maximum 3 reference images are allowed."""
        from nodetool.metadata.types import ImageRef

        node = Veo31ReferenceToVideo(prompt="test")
        node.image1 = ImageRef(uri="http://example.com/1.jpg")
        node.image2 = ImageRef(uri="http://example.com/2.jpg")
        node.image3 = ImageRef(uri="http://example.com/3.jpg")

        with patch.object(
            node, "_upload_image", return_value="http://uploaded-url.com/image.jpg"
        ):
            # Should not raise - exactly 3 images is valid
            params = await node._get_input_params(mock_context)
            assert len(params["imageUrls"]) == 3


class TestVeo31Visibility:
    """Tests for Veo 3.1 node visibility."""

    def test_veo31_text_to_video_visible(self):
        """Veo31TextToVideo should be visible in UI."""
        assert Veo31TextToVideo.is_visible()

    def test_veo31_image_to_video_visible(self):
        """Veo31ImageToVideo should be visible in UI."""
        assert Veo31ImageToVideo.is_visible()

    def test_veo31_reference_to_video_visible(self):
        """Veo31ReferenceToVideo should be visible in UI."""
        assert Veo31ReferenceToVideo.is_visible()


class TestKlingMotionControl:
    """Tests for KlingMotionControl node."""

    @pytest.mark.asyncio
    async def test_model_name(self):
        """Test model name returns correct value."""
        node = KlingMotionControl()
        assert node._get_model() == "kling-2.6/motion-control"

    @pytest.mark.asyncio
    async def test_get_title(self):
        """Test get_title returns correct value."""
        assert KlingMotionControl.get_title() == "Kling 2.6 Motion Control"

    @pytest.mark.asyncio
    async def test_is_visible(self):
        """Test node is visible in UI."""
        assert KlingMotionControl.is_visible()

    @pytest.mark.asyncio
    async def test_default_values(self):
        """Test default values are set correctly."""
        node = KlingMotionControl()
        assert node.prompt == "The cartoon character is dancing."
        assert node.character_orientation == KlingMotionControl.CharacterOrientation.VIDEO
        assert node.mode == KlingMotionControl.Mode.R720P

    @pytest.mark.asyncio
    async def test_missing_image_raises_error(self, mock_context):
        """Test that missing image raises ValueError."""
        from nodetool.metadata.types import ImageRef, VideoRef

        node = KlingMotionControl(
            image=ImageRef(),
            video=VideoRef(uri="http://example.com/video.mp4"),
        )
        with pytest.raises(ValueError, match="Reference image is required"):
            await node._get_input_params(mock_context)

    @pytest.mark.asyncio
    async def test_missing_video_raises_error(self, mock_context):
        """Test that missing video raises ValueError."""
        from nodetool.metadata.types import ImageRef, VideoRef

        node = KlingMotionControl(
            image=ImageRef(uri="http://example.com/image.jpg"),
            video=VideoRef(),
        )
        with pytest.raises(ValueError, match="Reference video is required"):
            await node._get_input_params(mock_context)

    @pytest.mark.asyncio
    async def test_input_params(self, mock_context):
        """Test input parameters are correctly generated."""
        from nodetool.metadata.types import ImageRef, VideoRef

        node = KlingMotionControl(
            prompt="A character dancing",
            image=ImageRef(uri="http://example.com/image.jpg"),
            video=VideoRef(uri="http://example.com/video.mp4"),
            character_orientation=KlingMotionControl.CharacterOrientation.IMAGE,
            mode=KlingMotionControl.Mode.R1080P,
        )
        with patch.object(
            node, "_upload_image", return_value="http://uploaded-url.com/image.jpg"
        ):
            with patch.object(
                node, "_upload_video", return_value="http://uploaded-url.com/video.mp4"
            ):
                params = await node._get_input_params(mock_context)

        assert params == {
            "prompt": "A character dancing",
            "input_urls": ["http://uploaded-url.com/image.jpg"],
            "video_urls": ["http://uploaded-url.com/video.mp4"],
            "character_orientation": "image",
            "mode": "1080p",
        }

    @pytest.mark.asyncio
    async def test_submit_payload(self, mock_context):
        """Test that submit payload is correctly generated."""
        from nodetool.metadata.types import ImageRef, VideoRef

        node = KlingMotionControl(
            prompt="Character animation",
            image=ImageRef(uri="http://example.com/image.jpg"),
            video=VideoRef(uri="http://example.com/video.mp4"),
        )
        with patch.object(
            node, "_upload_image", return_value="http://uploaded-url.com/image.jpg"
        ):
            with patch.object(
                node, "_upload_video", return_value="http://uploaded-url.com/video.mp4"
            ):
                payload = await node._get_submit_payload(mock_context)

        assert payload == {
            "model": "kling-2.6/motion-control",
            "input": {
                "prompt": "Character animation",
                "input_urls": ["http://uploaded-url.com/image.jpg"],
                "video_urls": ["http://uploaded-url.com/video.mp4"],
                "character_orientation": "video",
                "mode": "720p",
            },
        }


# Tests for new image nodes
class TestGPTImage4oTextToImage:
    """Tests for GPTImage4oTextToImage node."""

    @pytest.mark.asyncio
    async def test_model_and_params(self):
        """Test model name and input parameters."""
        node = GPTImage4oTextToImage(prompt="test image")
        assert node._get_model() == "4o-image/text-to-image"
        params = await node._get_input_params()
        assert params["prompt"] == "test image"
        assert params["aspect_ratio"] == "1:1"
        assert params["quality"] == "auto"

    @pytest.mark.asyncio
    async def test_empty_prompt_raises_error(self):
        """Test that empty prompt raises ValueError."""
        node = GPTImage4oTextToImage(prompt="")
        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            await node._get_input_params()


class TestGPTImage15TextToImage:
    """Tests for GPTImage15TextToImage node."""

    @pytest.mark.asyncio
    async def test_model_and_params(self):
        """Test model name and input parameters."""
        node = GPTImage15TextToImage(prompt="test image")
        assert node._get_model() == "gpt-image-1.5/text-to-image"
        params = await node._get_input_params()
        assert params["prompt"] == "test image"

    @pytest.mark.asyncio
    async def test_empty_prompt_raises_error(self):
        """Test that empty prompt raises ValueError."""
        node = GPTImage15TextToImage(prompt="")
        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            await node._get_input_params()


class TestIdeogramV3TextToImage:
    """Tests for IdeogramV3TextToImage node."""

    @pytest.mark.asyncio
    async def test_model_and_params(self):
        """Test model name and input parameters."""
        node = IdeogramV3TextToImage(prompt="test image")
        assert node._get_model() == "ideogram/v3-text-to-image"
        params = await node._get_input_params()
        assert params["prompt"] == "test image"
        assert params["rendering_speed"] == "BALANCED"
        assert params["style"] == "AUTO"

    @pytest.mark.asyncio
    async def test_empty_prompt_raises_error(self):
        """Test that empty prompt raises ValueError."""
        node = IdeogramV3TextToImage(prompt="")
        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            await node._get_input_params()


class TestSeedream40TextToImage:
    """Tests for Seedream40TextToImage node."""

    @pytest.mark.asyncio
    async def test_model_and_params(self):
        """Test model name and input parameters."""
        node = Seedream40TextToImage(prompt="test image")
        assert node._get_model() == "seedream/4.0-text-to-image"
        params = await node._get_input_params()
        assert params["prompt"] == "test image"
        assert params["quality"] == "basic"

    @pytest.mark.asyncio
    async def test_empty_prompt_raises_error(self):
        """Test that empty prompt raises ValueError."""
        node = Seedream40TextToImage(prompt="")
        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            await node._get_input_params()


# Tests for new video nodes
class TestKling21TextToVideo:
    """Tests for Kling21TextToVideo node."""

    @pytest.mark.asyncio
    async def test_model_and_params(self):
        """Test model name and input parameters."""
        node = Kling21TextToVideo(prompt="test video")
        assert node._get_model() == "kling/v2-1-text-to-video"
        params = await node._get_input_params()
        assert params["prompt"] == "test video"
        assert params["duration"] == "5"

    @pytest.mark.asyncio
    async def test_empty_prompt_raises_error(self):
        """Test that empty prompt raises ValueError."""
        node = Kling21TextToVideo(prompt="")
        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            await node._get_input_params()


class TestWan25TextToVideo:
    """Tests for Wan25TextToVideo node."""

    @pytest.mark.asyncio
    async def test_model_and_params(self):
        """Test model name and input parameters."""
        node = Wan25TextToVideo(prompt="test video")
        assert node._get_model() == "wan/2-5-text-to-video"
        params = await node._get_input_params()
        assert params["prompt"] == "test video"
        assert params["duration"] == "5s"


class TestWan22TextToVideo:
    """Tests for Wan22TextToVideo node."""

    @pytest.mark.asyncio
    async def test_model_and_params(self):
        """Test model name and input parameters."""
        node = Wan22TextToVideo(prompt="test video")
        assert node._get_model() == "wan/v2-2-text-to-video"
        params = await node._get_input_params()
        assert params["prompt"] == "test video"


class TestHailuo02TextToVideo:
    """Tests for Hailuo02TextToVideo node."""

    @pytest.mark.asyncio
    async def test_model_and_params(self):
        """Test model name and input parameters."""
        node = Hailuo02TextToVideo(prompt="test video")
        assert node._get_model() == "hailuo/02-text-to-video"
        params = await node._get_input_params()
        assert params["prompt"] == "test video"


class TestSora2WatermarkRemover:
    """Tests for Sora2WatermarkRemover node."""

    def test_model(self):
        """Test model name."""
        node = Sora2WatermarkRemover()
        assert node._get_model() == "sora-2-watermark-remover"


# Tests for new audio nodes
class TestElevenLabsAudioIsolation:
    """Tests for ElevenLabsAudioIsolation node."""

    def test_model(self):
        """Test model name."""
        node = ElevenLabsAudioIsolation()
        assert node._get_model() == "elevenlabs/audio-isolation"


class TestElevenLabsSoundEffect:
    """Tests for ElevenLabsSoundEffect node."""

    @pytest.mark.asyncio
    async def test_model_and_params(self):
        """Test model name and input parameters."""
        node = ElevenLabsSoundEffect(text="explosion sound")
        assert node._get_model() == "elevenlabs/sound-effect"
        params = await node._get_input_params()
        assert params["text"] == "explosion sound"
        assert params["duration_seconds"] == 5.0

    @pytest.mark.asyncio
    async def test_empty_text_raises_error(self):
        """Test that empty text raises ValueError."""
        node = ElevenLabsSoundEffect(text="")
        with pytest.raises(ValueError, match="Text cannot be empty"):
            await node._get_input_params()


class TestElevenLabsSpeechToText:
    """Tests for ElevenLabsSpeechToText node."""

    def test_model(self):
        """Test model name."""
        node = ElevenLabsSpeechToText()
        assert node._get_model() == "elevenlabs/speech-to-text"


class TestElevenLabsV3Dialogue:
    """Tests for ElevenLabsV3Dialogue node."""

    @pytest.mark.asyncio
    async def test_model_and_params(self):
        """Test model name and input parameters."""
        node = ElevenLabsV3Dialogue(text="Hello, how are you?")
        assert node._get_model() == "elevenlabs/text-to-dialogue-v3"
        params = await node._get_input_params()
        assert params["text"] == "Hello, how are you?"
        assert params["voice"] == "Rachel"

    @pytest.mark.asyncio
    async def test_empty_text_raises_error(self):
        """Test that empty text raises ValueError."""
        node = ElevenLabsV3Dialogue(text="")
        with pytest.raises(ValueError, match="Text cannot be empty"):
            await node._get_input_params()
