"""Tests for Kie.ai image generation nodes."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from nodetool.workflows.processing_context import ProcessingContext
from nodetool.nodes.kie.image import (
    KieBaseNode,
    Generate4OImage,
    SeedreamGenerate,
    ZImageGenerate,
    NanoBananaGenerate,
    FluxProTextToImage,
    NanoBananaProGenerate,
    FluxKontextGenerate,
    GrokImagineGenerate,
    TopazImageUpscale,
)
from nodetool.nodes.kie.video import (
    Veo31Generate,
    Wan26Generate,
    Sora2Generate,
    Sora2ProGenerate,
    Seedance10Generate,
    Hailuo23Generate,
    KlingAIAvatar,
    TopazVideoUpscale,
)
from nodetool.nodes.kie.audio import SunoMusicGenerate


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
        assert Generate4OImage.is_visible()
        assert SeedreamGenerate.is_visible()
        assert ZImageGenerate.is_visible()
        assert NanoBananaGenerate.is_visible()
        assert FluxProTextToImage.is_visible()
        assert NanoBananaProGenerate.is_visible()
        assert FluxKontextGenerate.is_visible()
        assert GrokImagineGenerate.is_visible()
        assert TopazImageUpscale.is_visible()
        # Video generation nodes
        assert Veo31Generate.is_visible()
        assert Wan26Generate.is_visible()
        assert Sora2Generate.is_visible()
        assert Sora2ProGenerate.is_visible()
        assert Seedance10Generate.is_visible()
        assert Hailuo23Generate.is_visible()
        assert KlingAIAvatar.is_visible()
        assert TopazVideoUpscale.is_visible()
        # Audio generation nodes
        assert SunoMusicGenerate.is_visible()


class TestGenerate4OImage:
    """Tests for Generate4OImage node."""

    @pytest.mark.asyncio
    async def test_submit_payload(self, mock_context):
        """Test that submit payload is correctly generated."""
        node = Generate4OImage(
            prompt="a cat", aspect_ratio=Generate4OImage.AspectRatio.LANDSCAPE
        )
        payload = node._get_submit_payload()
        assert payload == {"prompt": "a cat", "aspect_ratio": "16:9"}

    @pytest.mark.asyncio
    async def test_empty_prompt_raises_error(self, mock_context):
        """Test that empty prompt raises ValueError."""
        node = Generate4OImage(prompt="")
        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            node._get_submit_payload()

    @pytest.mark.asyncio
    async def test_endpoints(self):
        """Test endpoint generation."""
        node = Generate4OImage(prompt="test")
        assert node._get_base_endpoint() == "/v1/4o-images"
        # Derived endpoints
        assert node._get_submit_endpoint() == "/v1/4o-images/generate"
        assert node._get_status_endpoint("task123") == "/v1/4o-images/task123"
        assert (
            node._get_download_endpoint("task123") == "/v1/4o-images/task123/download"
        )

    @pytest.mark.asyncio
    async def test_process_success(self, mock_context):
        """Test successful image generation."""
        node = Generate4OImage(
            prompt="a beautiful sunset",
            poll_interval=0.5,
            max_poll_attempts=5,
        )

        # Mock responses: submit, status (completed), download
        responses = [
            MockResponse(json_data={"task_id": "task123"}),
            MockResponse(json_data={"status": "completed"}),
            MockResponse(content=b"image_bytes", content_type="image/png"),
        ]

        with patch("aiohttp.ClientSession", return_value=MockSession(responses)):
            await node.process(mock_context)

        mock_context.image_from_bytes.assert_called_once_with(b"image_bytes")

    @pytest.mark.asyncio
    async def test_missing_api_key(self, mock_context):
        """Test that missing API key raises error."""
        mock_context.get_secret = AsyncMock(return_value=None)
        node = Generate4OImage(prompt="test")

        with pytest.raises(ValueError, match="KIE_API_KEY secret is not configured"):
            await node._get_api_key(mock_context)


class TestSeedreamGenerate:
    """Tests for SeedreamGenerate node."""

    @pytest.mark.asyncio
    async def test_endpoints(self):
        """Test endpoint generation."""
        node = SeedreamGenerate(prompt="test")
        assert node._get_base_endpoint() == "/v1/market/seedream-4-5"
        # Derived endpoints
        assert node._get_submit_endpoint() == "/v1/market/seedream-4-5/generate"
        assert node._get_status_endpoint("task123") == "/v1/market/seedream-4-5/task123"
        assert (
            node._get_download_endpoint("task123")
            == "/v1/market/seedream-4-5/task123/download"
        )

    @pytest.mark.asyncio
    async def test_submit_payload(self):
        """Test submit payload generation."""
        node = SeedreamGenerate(
            prompt="artistic scene",
            aspect_ratio=SeedreamGenerate.AspectRatio.PORTRAIT,
        )
        payload = node._get_submit_payload()
        assert payload == {"prompt": "artistic scene", "aspect_ratio": "9:16"}


class TestZImageGenerate:
    """Tests for ZImageGenerate node."""

    @pytest.mark.asyncio
    async def test_endpoints(self):
        """Test endpoint generation."""
        node = ZImageGenerate(prompt="test")
        assert node._get_base_endpoint() == "/v1/market/z-image"
        # Derived endpoints
        assert node._get_submit_endpoint() == "/v1/market/z-image/generate"
        assert node._get_status_endpoint("task123") == "/v1/market/z-image/task123"
        assert (
            node._get_download_endpoint("task123")
            == "/v1/market/z-image/task123/download"
        )


class TestNanoBananaGenerate:
    """Tests for NanoBananaGenerate node."""

    @pytest.mark.asyncio
    async def test_endpoints(self):
        """Test endpoint generation."""
        node = NanoBananaGenerate(prompt="test")
        assert node._get_base_endpoint() == "/v1/market/google/nano-banana"
        # Derived endpoints
        assert node._get_submit_endpoint() == "/v1/market/google/nano-banana/generate"
        assert (
            node._get_status_endpoint("task123")
            == "/v1/market/google/nano-banana/task123"
        )
        assert (
            node._get_download_endpoint("task123")
            == "/v1/market/google/nano-banana/task123/download"
        )


class TestFluxProTextToImage:
    """Tests for FluxProTextToImage node."""

    @pytest.mark.asyncio
    async def test_endpoints(self):
        """Test endpoint generation."""
        node = FluxProTextToImage(prompt="test")
        assert node._get_base_endpoint() == "/v1/market/flux2/pro"
        # FluxPro uses text-to-image instead of generate
        assert node._get_submit_endpoint() == "/v1/market/flux2/pro/text-to-image"
        assert node._get_status_endpoint("task123") == "/v1/market/flux2/pro/task123"
        assert (
            node._get_download_endpoint("task123")
            == "/v1/market/flux2/pro/task123/download"
        )

    @pytest.mark.asyncio
    async def test_submit_payload_with_extra_params(self):
        """Test submit payload includes steps and guidance_scale."""
        node = FluxProTextToImage(
            prompt="detailed artwork",
            aspect_ratio=FluxProTextToImage.AspectRatio.WIDE,
            steps=30,
            guidance_scale=10.0,
        )
        payload = node._get_submit_payload()
        assert payload == {
            "prompt": "detailed artwork",
            "aspect_ratio": "4:3",
            "steps": 30,
            "guidance_scale": 10.0,
        }


class TestKiePollingLogic:
    """Tests for the polling logic in KieBaseNode."""

    @pytest.mark.asyncio
    async def test_task_status_extraction(self):
        """Test various status extraction scenarios."""
        node = Generate4OImage(prompt="test")

        # Direct status field
        assert node._is_task_complete({"status": "completed"})
        assert node._is_task_complete({"status": "success"})
        assert node._is_task_complete({"status": "done"})

        # Nested in data
        assert node._is_task_complete({"data": {"status": "completed"}})

        # Not complete
        assert not node._is_task_complete({"status": "processing"})
        assert not node._is_task_complete({"status": "pending"})

    @pytest.mark.asyncio
    async def test_task_failure_detection(self):
        """Test task failure detection."""
        node = Generate4OImage(prompt="test")

        assert node._is_task_failed({"status": "failed"})
        assert node._is_task_failed({"status": "error"})
        assert node._is_task_failed({"status": "cancelled"})
        assert not node._is_task_failed({"status": "processing"})

    @pytest.mark.asyncio
    async def test_task_id_extraction(self):
        """Test task ID extraction from various response formats."""
        node = Generate4OImage(prompt="test")

        assert node._extract_task_id({"task_id": "abc123"}) == "abc123"
        assert node._extract_task_id({"data": {"task_id": "xyz789"}}) == "xyz789"

        with pytest.raises(ValueError, match="Could not extract task_id"):
            node._extract_task_id({"other_field": "value"})

    @pytest.mark.asyncio
    async def test_headers_generation(self):
        """Test API headers generation."""
        node = Generate4OImage(prompt="test")
        headers = node._get_headers("my-api-key")

        assert headers["Authorization"] == "Bearer my-api-key"
        assert headers["Content-Type"] == "application/json"

    @pytest.mark.asyncio
    async def test_poll_timeout(self, mock_context):
        """Test that polling times out after max attempts."""
        node = Generate4OImage(
            prompt="test",
            poll_interval=0.5,
            max_poll_attempts=3,
        )

        # Submit succeeds, but status always returns processing
        responses = [
            MockResponse(json_data={"task_id": "task123"}),
            MockResponse(json_data={"status": "processing"}),
            MockResponse(json_data={"status": "processing"}),
            MockResponse(json_data={"status": "processing"}),
        ]

        with patch("aiohttp.ClientSession", return_value=MockSession(responses)):
            with pytest.raises(TimeoutError, match="did not complete"):
                await node.process(mock_context)

    @pytest.mark.asyncio
    async def test_task_failure_handling(self, mock_context):
        """Test that task failure is properly handled."""
        node = Generate4OImage(
            prompt="test",
            poll_interval=0.5,
        )

        responses = [
            MockResponse(json_data={"task_id": "task123"}),
            MockResponse(
                json_data={"status": "failed", "message": "Content policy violation"}
            ),
        ]

        with patch("aiohttp.ClientSession", return_value=MockSession(responses)):
            with pytest.raises(
                ValueError, match="Task failed: Content policy violation"
            ):
                await node.process(mock_context)

    @pytest.mark.asyncio
    async def test_download_json_response_with_url(self, mock_context):
        """Test downloading when response contains a URL instead of raw bytes."""
        node = Generate4OImage(
            prompt="test",
            poll_interval=0.5,
        )

        # Simulate a download response that returns JSON with an image URL
        responses = [
            MockResponse(json_data={"task_id": "task123"}),
            MockResponse(json_data={"status": "completed"}),
            MockResponse(json_data={"url": "https://example.com/image.png"}),
            MockResponse(content=b"actual_image_bytes", content_type="image/png"),
        ]

        with patch("aiohttp.ClientSession", return_value=MockSession(responses)):
            await node.process(mock_context)

        mock_context.image_from_bytes.assert_called_once_with(b"actual_image_bytes")


class TestNanoBananaProGenerate:
    """Tests for NanoBananaProGenerate node."""

    @pytest.mark.asyncio
    async def test_endpoints(self):
        """Test endpoint generation."""
        node = NanoBananaProGenerate(prompt="test")
        assert node._get_base_endpoint() == "/v1/market/google/nano-banana-pro"
        assert (
            node._get_submit_endpoint() == "/v1/market/google/nano-banana-pro/generate"
        )


class TestFluxKontextGenerate:
    """Tests for FluxKontextGenerate node."""

    @pytest.mark.asyncio
    async def test_endpoints(self):
        """Test endpoint generation."""
        node = FluxKontextGenerate(prompt="test")
        assert node._get_base_endpoint() == "/v1/market/flux-kontext"

    @pytest.mark.asyncio
    async def test_submit_payload_with_mode(self):
        """Test submit payload includes mode."""
        node = FluxKontextGenerate(
            prompt="artwork",
            mode=FluxKontextGenerate.Mode.MAX,
        )
        payload = node._get_submit_payload()
        assert payload["mode"] == "max"


class TestGrokImagineGenerate:
    """Tests for GrokImagineGenerate node."""

    @pytest.mark.asyncio
    async def test_endpoints(self):
        """Test endpoint generation."""
        node = GrokImagineGenerate(prompt="test")
        assert node._get_base_endpoint() == "/v1/market/grok-imagine"


class TestVeo31Generate:
    """Tests for Veo31Generate node."""

    @pytest.mark.asyncio
    async def test_endpoints(self):
        """Test endpoint generation."""
        node = Veo31Generate(prompt="test")
        assert node._get_base_endpoint() == "/v1/market/veo-3-1"

    @pytest.mark.asyncio
    async def test_submit_payload(self):
        """Test submit payload generation."""
        node = Veo31Generate(
            prompt="a sunset",
            mode=Veo31Generate.Mode.FAST,
            duration=10,
        )
        payload = node._get_submit_payload()
        assert payload["prompt"] == "a sunset"
        assert payload["mode"] == "fast"
        assert payload["duration"] == 10


class TestWan26Generate:
    """Tests for Wan26Generate node."""

    @pytest.mark.asyncio
    async def test_endpoints(self):
        """Test endpoint generation."""
        node = Wan26Generate(prompt="test")
        assert node._get_base_endpoint() == "/v1/market/wan-2-6"

    @pytest.mark.asyncio
    async def test_submit_payload_with_multi_shots(self):
        """Test submit payload with multi_shots enabled."""
        node = Wan26Generate(
            prompt="action scene",
            multi_shots=True,
            resolution=Wan26Generate.Resolution.FULL_HD_1080P,
        )
        payload = node._get_submit_payload()
        assert payload["multi_shots"] is True
        assert payload["resolution"] == "1080p"


class TestSora2Generate:
    """Tests for Sora2Generate node."""

    @pytest.mark.asyncio
    async def test_endpoints(self):
        """Test endpoint generation."""
        node = Sora2Generate(prompt="test")
        assert node._get_base_endpoint() == "/v1/market/sora-2"


class TestSora2ProGenerate:
    """Tests for Sora2ProGenerate node."""

    @pytest.mark.asyncio
    async def test_endpoints(self):
        """Test endpoint generation."""
        node = Sora2ProGenerate(prompt="test")
        assert node._get_base_endpoint() == "/v1/market/sora-2-pro"


class TestSeedance10Generate:
    """Tests for Seedance10Generate node."""

    @pytest.mark.asyncio
    async def test_endpoints(self):
        """Test endpoint generation."""
        node = Seedance10Generate(prompt="test")
        assert node._get_base_endpoint() == "/v1/market/seedance-1-0"


class TestHailuo23Generate:
    """Tests for Hailuo23Generate node."""

    @pytest.mark.asyncio
    async def test_endpoints(self):
        """Test endpoint generation."""
        node = Hailuo23Generate(prompt="test")
        assert node._get_base_endpoint() == "/v1/market/hailuo-2-3"


class TestSunoMusicGenerate:
    """Tests for SunoMusicGenerate node."""

    @pytest.mark.asyncio
    async def test_endpoints(self):
        """Test endpoint generation."""
        node = SunoMusicGenerate(prompt="upbeat pop song")
        assert node._get_base_endpoint() == "/v1/market/suno"

    @pytest.mark.asyncio
    async def test_submit_payload(self):
        """Test submit payload generation."""
        node = SunoMusicGenerate(
            prompt="energetic rock song",
            style=SunoMusicGenerate.Style.ROCK,
            instrumental=True,
            duration=120,
            model=SunoMusicGenerate.Model.V4_5_PLUS,
        )
        payload = node._get_submit_payload()
        assert payload["prompt"] == "energetic rock song"
        assert payload["style"] == "rock"
        assert payload["instrumental"] is True
        assert payload["duration"] == 120
        assert payload["model"] == "v4.5+"
