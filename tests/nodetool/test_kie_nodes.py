"""Tests for Kie.ai image generation nodes."""

import pytest
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
    FluxKontext,
    GrokImagineTextToImage,
    GrokImagineUpscale,
    TopazImageUpscale,
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
    KlingAIAvatarPro,
    TopazVideoUpscale,
    GrokImagineImageToVideo,
    GrokImagineTextToVideo,
)
from nodetool.nodes.kie.audio import Suno


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
        assert KlingAIAvatarPro.is_visible()
        assert TopazVideoUpscale.is_visible()
        assert GrokImagineImageToVideo.is_visible()
        assert GrokImagineTextToVideo.is_visible()
        # Audio generation nodes
        assert Suno.is_visible()


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
            poll_interval=0.5,
            max_poll_attempts=5,
        )

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

        mock_context.image_from_bytes.assert_called_once_with(b"image_bytes")

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
            "input": {"prompt": "artistic scene", "aspect_ratio": "9:16"},
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
        assert params["aspect_ratio"] == "1:1"


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
        node = Flux2ProTextToImage(
            prompt="test",
            poll_interval=0.5,
            max_poll_attempts=3,
        )

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
        node = Flux2ProTextToImage(
            prompt="test",
            poll_interval=0.5,
        )

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
        node = Flux2ProTextToImage(
            prompt="test",
            poll_interval=0.5,
        )

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

        mock_context.image_from_bytes.assert_called_once_with(b"actual_image_bytes")


class TestNanoBananaProGenerate:
    """Tests for NanoBananaProGenerate node."""

    @pytest.mark.asyncio
    async def test_model_and_params(self):
        """Test model name and input parameters."""
        node = NanoBananaPro(prompt="test")
        assert node._get_model() == "google/nano-banana-pro"
        params = await node._get_input_params()
        assert params["prompt"] == "test"
        assert params["aspect_ratio"] == "1:1"


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
        assert params["n_frames"] == "10"
        assert params["remove_watermark"] is True


class TestSora2ProImageToVideo:
    """Tests for Sora2ProImageToVideo node."""

    @pytest.mark.asyncio
    async def test_model_and_params(self, mock_context):
        """Test model name and input parameters."""
        from nodetool.metadata.types import ImageRef

        # Mock the image upload method
        with patch.object(Sora2ProImageToVideo, '_upload_image', new_callable=AsyncMock) as mock_upload:
            mock_upload.return_value = "http://example.com/image.jpg"
            
            node = Sora2ProImageToVideo(
                image=ImageRef(uri="http://example.com/image.jpg"), prompt="test"
            )
            assert node._get_model() == "sora-2-pro-image-to-video"
            params = await node._get_input_params(mock_context)
            assert params["prompt"] == "test"
            assert params["image_urls"] == ["http://example.com/image.jpg"]
            assert params["aspect_ratio"] == "landscape"
            assert params["n_frames"] == "10"
            assert params["remove_watermark"] is True


class TestSora2ProStoryboard:
    """Tests for Sora2ProStoryboard node."""

    @pytest.mark.asyncio
    async def test_model_and_params(self):
        """Test model name and input parameters."""
        node = Sora2ProStoryboard(prompt="test storyboard")
        assert node._get_model() == "sora-2-pro-storyboard"
        params = await node._get_input_params()
        assert params["prompt"] == "test storyboard"
        assert params["aspect_ratio"] == "landscape"
        assert params["n_frames"] == "10"
        assert params["remove_watermark"] is True


class TestSora2TextToVideo:
    """Tests for Sora2TextToVideo node."""

    @pytest.mark.asyncio
    async def test_model_and_params(self, mock_context):
        """Test model name and input parameters."""
        node = Sora2TextToVideo(prompt="test")
        assert node._get_model() == "sora-2-text-to-video"
        params = await node._get_input_params(mock_context)
        assert params["prompt"] == "test"
        assert params["aspect_ratio"] == "landscape"
        assert params["n_frames"] == "10"
        assert params["remove_watermark"] is True
        assert params["mode"] == "standard"


class TestSeedanceV1LiteTextToVideo:
    """Tests for SeedanceV1LiteTextToVideo node."""

    @pytest.mark.asyncio
    async def test_model_and_params(self):
        """Test model name and input parameters."""
        node = SeedanceV1LiteTextToVideo(prompt="test")
        assert node._get_model() == "bytedance/v1-lite-text-to-video"
        params = await node._get_input_params()
        assert params["prompt"] == "test"
        assert params["aspect_ratio"] == "16:9"
        assert params["resolution"] == "720p"
        assert params["duration"] == "5"
        assert params["camera_fixed"] is False
        assert params["seed"] == -1
        assert params["enable_safety_checker"] is True


class TestSeedanceV1ProTextToVideo:
    """Tests for SeedanceV1ProTextToVideo node."""

    @pytest.mark.asyncio
    async def test_model_and_params(self):
        """Test model name and input parameters."""
        node = SeedanceV1ProTextToVideo(prompt="test")
        assert node._get_model() == "bytedance/v1-pro-text-to-video"
        params = await node._get_input_params()
        assert params["prompt"] == "test"
        assert params["aspect_ratio"] == "16:9"
        assert params["resolution"] == "720p"
        assert params["duration"] == "5"
        assert params["camera_fixed"] is False
        assert params["seed"] == -1
        assert params["enable_safety_checker"] is True


class TestSeedanceV1LiteImageToVideo:
    """Tests for SeedanceV1LiteImageToVideo node."""

    @pytest.mark.asyncio
    async def test_model_and_params(self, mock_context):
        """Test model name and input parameters."""
        from nodetool.metadata.types import ImageRef

        with patch.object(SeedanceV1LiteImageToVideo, '_upload_image', new_callable=AsyncMock) as mock_upload:
            mock_upload.return_value = "http://example.com/image.jpg"
            
            node = SeedanceV1LiteImageToVideo(
                image=ImageRef(uri="http://example.com/image.jpg"), prompt="test"
            )
            assert node._get_model() == "bytedance/v1-lite-image-to-video"
            params = await node._get_input_params(mock_context)
            assert params["prompt"] == "test"
            assert params["image_url"] == "http://example.com/image.jpg"
            assert params["resolution"] == "720p"
            assert params["duration"] == "5"
            assert params["camera_fixed"] is False
            assert params["seed"] == -1
            assert params["enable_safety_checker"] is True


class TestSeedanceV1ProImageToVideo:
    """Tests for SeedanceV1ProImageToVideo node."""

    @pytest.mark.asyncio
    async def test_model_and_params(self, mock_context):
        """Test model name and input parameters."""
        from nodetool.metadata.types import ImageRef

        with patch.object(SeedanceV1ProImageToVideo, '_upload_image', new_callable=AsyncMock) as mock_upload:
            mock_upload.return_value = "http://example.com/image.jpg"
            
            node = SeedanceV1ProImageToVideo(
                image=ImageRef(uri="http://example.com/image.jpg"), prompt="test"
            )
            assert node._get_model() == "bytedance/v1-pro-image-to-video"
            params = await node._get_input_params(mock_context)
            assert params["prompt"] == "test"
            assert params["image_url"] == "http://example.com/image.jpg"
            assert params["resolution"] == "720p"
            assert params["duration"] == "5"
            assert params["camera_fixed"] is False
            assert params["seed"] == -1
            assert params["enable_safety_checker"] is True


class TestSeedanceV1ProFastImageToVideo:
    """Tests for SeedanceV1ProFastImageToVideo node."""

    @pytest.mark.asyncio
    async def test_model_and_params(self, mock_context):
        """Test model name and input parameters."""
        from nodetool.metadata.types import ImageRef

        with patch.object(SeedanceV1ProFastImageToVideo, '_upload_image', new_callable=AsyncMock) as mock_upload:
            mock_upload.return_value = "http://example.com/image.jpg"
            
            node = SeedanceV1ProFastImageToVideo(
                image=ImageRef(uri="http://example.com/image.jpg"), prompt="test"
            )
            assert node._get_model() == "bytedance/v1-pro-fast-image-to-video"
            params = await node._get_input_params(mock_context)
            assert params["prompt"] == "test"
            assert params["image_url"] == "http://example.com/image.jpg"
            assert params["resolution"] == "720p"
            assert params["duration"] == "5"
            assert params["camera_fixed"] is False
            assert params["seed"] == -1
            assert params["enable_safety_checker"] is True


class TestHailuoImageToVideoPro:
    """Tests for HailuoImageToVideoPro node."""

    @pytest.mark.asyncio
    async def test_model_and_params(self, mock_context):
        """Test model name and input parameters."""
        from nodetool.metadata.types import ImageRef

        with patch.object(HailuoImageToVideoPro, '_upload_image', new_callable=AsyncMock) as mock_upload:
            mock_upload.return_value = "http://example.com/image.jpg"
            
            node = HailuoImageToVideoPro(
                image=ImageRef(uri="http://example.com/image.jpg"), prompt="test"
            )
            assert node._get_model() == "hailuo/2-3-image-to-video-pro"
            params = await node._get_input_params(mock_context)
            assert params["prompt"] == "test"
            assert params["image_url"] == "http://example.com/image.jpg"
            assert params["resolution"] == "1080p"


class TestHailuoImageToVideoStandard:
    """Tests for HailuoImageToVideoStandard node."""

    @pytest.mark.asyncio
    async def test_model_and_params(self, mock_context):
        """Test model name and input parameters."""
        from nodetool.metadata.types import ImageRef

        with patch.object(HailuoImageToVideoStandard, '_upload_image', new_callable=AsyncMock) as mock_upload:
            mock_upload.return_value = "http://example.com/image.jpg"
            
            node = HailuoImageToVideoStandard(
                image=ImageRef(uri="http://example.com/image.jpg"), prompt="test"
            )
            assert node._get_model() == "hailuo/2-3-image-to-video-standard"
            params = await node._get_input_params(mock_context)
            assert params["prompt"] == "test"
            assert params["image_url"] == "http://example.com/image.jpg"
            assert params["resolution"] == "720p"


class TestSuno:
    """Tests for Suno node."""

    @pytest.mark.asyncio
    async def test_model_and_params(self):
        """Test model name and input parameters."""
        node = Suno(prompt="upbeat pop song")
        assert node._get_model() == "suno"
        # Suno has a sync _get_input_params method
        params = node._get_input_params()
        assert params["prompt"] == "upbeat pop song"
        assert params["instrumental"] is False
        assert params["duration"] == 60
        assert params["model"] == "v4.5+"

    @pytest.mark.asyncio
    async def test_submit_payload(self):
        """Test submit payload generation."""
        node = Suno(
            prompt="energetic rock song",
            style=Suno.Style.ROCK,
            instrumental=True,
            duration=120,
            model=Suno.Model.V4_5_PLUS,
        )
        # Suno has a sync _get_input_params, so _get_submit_payload needs context
        # For this test, just check input params directly
        params = node._get_input_params()
        assert params["prompt"] == "energetic rock song"
        assert params["style"] == "rock"
        assert params["instrumental"] is True
        assert params["duration"] == 120
        assert params["model"] == "v4.5+"
