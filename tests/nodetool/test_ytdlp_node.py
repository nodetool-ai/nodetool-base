"""Tests for yt-dlp download node."""

import os
import tempfile
from unittest.mock import patch

import pytest

from nodetool.nodes.lib.ytdlp import DownloadError, YtDlpDownload
from nodetool.workflows.processing_context import ProcessingContext


@pytest.fixture
def context():
    """Create a mock processing context."""
    return ProcessingContext(user_id="test", auth_token="test")


@pytest.fixture
def sample_info_dict():
    """Sample yt-dlp info dict for testing."""
    return {
        "id": "test_video_id",
        "title": "Test Video Title",
        "description": "Test description",
        "uploader": "Test Uploader",
        "upload_date": "20240101",
        "duration": 120,
        "view_count": 1000,
        "ext": "mp4",
        "width": 1920,
        "height": 1080,
        "fps": 30,
    }


class TestYtDlpDownloadValidation:
    """Tests for input validation."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        node = YtDlpDownload()
        assert node.url == ""
        assert node.mode == "video"
        assert node.format_selector == "best"
        assert node.container == "auto"
        assert node.subtitles is False
        assert node.thumbnail is False
        assert node.overwrite is False
        assert node.rate_limit_kbps == 0
        assert node.timeout == 600

    @pytest.mark.asyncio
    async def test_empty_url_raises_error(self, context):
        """Test that empty URL raises DownloadError."""
        node = YtDlpDownload(url="")
        with pytest.raises(DownloadError, match="URL cannot be empty"):
            await node.process(context)

    @pytest.mark.asyncio
    async def test_whitespace_url_raises_error(self, context):
        """Test that whitespace-only URL raises DownloadError."""
        node = YtDlpDownload(url="   ")
        with pytest.raises(DownloadError, match="URL cannot be empty"):
            await node.process(context)

    @pytest.mark.asyncio
    async def test_invalid_url_raises_error(self, context):
        """Test that invalid URL raises DownloadError."""
        node = YtDlpDownload(url="not-a-valid-url")
        with pytest.raises(DownloadError, match="Invalid URL format"):
            await node.process(context)

    def test_is_valid_url_accepts_valid_urls(self):
        """Test URL validation accepts valid URLs."""
        node = YtDlpDownload()
        assert node._is_valid_url("https://www.youtube.com/watch?v=test")
        assert node._is_valid_url("http://example.com/video")
        assert node._is_valid_url("https://vimeo.com/123456")

    def test_is_valid_url_rejects_invalid_urls(self):
        """Test URL validation rejects invalid URLs."""
        node = YtDlpDownload()
        assert not node._is_valid_url("not-a-url")
        assert not node._is_valid_url("ftp://example.com")
        assert not node._is_valid_url("")


class TestYtDlpDownloadOptions:
    """Tests for yt-dlp options building."""

    def test_build_ydl_opts_video_mode(self):
        """Test options for video mode."""
        node = YtDlpDownload(mode="video")
        opts = node._build_ydl_opts("/tmp/test")

        assert opts["format"] == "bestvideo+bestaudio/best"
        assert opts["quiet"] is True
        assert opts["noplaylist"] is True
        assert "skip_download" not in opts or opts.get("skip_download") is not True

    def test_build_ydl_opts_audio_mode(self):
        """Test options for audio mode."""
        node = YtDlpDownload(mode="audio")
        opts = node._build_ydl_opts("/tmp/test")

        assert opts["format"] == "bestaudio/best"
        assert "postprocessors" in opts
        assert any(
            pp.get("key") == "FFmpegExtractAudio" for pp in opts["postprocessors"]
        )

    def test_build_ydl_opts_metadata_mode(self):
        """Test options for metadata-only mode."""
        node = YtDlpDownload(mode="metadata")
        opts = node._build_ydl_opts("/tmp/test")

        assert opts["skip_download"] is True

    def test_build_ydl_opts_with_rate_limit(self):
        """Test options with rate limiting."""
        node = YtDlpDownload(rate_limit_kbps=500)
        opts = node._build_ydl_opts("/tmp/test")

        assert opts["ratelimit"] == 500 * 1024  # Converted to bytes/sec

    def test_build_ydl_opts_with_subtitles(self):
        """Test options with subtitles enabled."""
        node = YtDlpDownload(subtitles=True)
        opts = node._build_ydl_opts("/tmp/test")

        assert opts["writesubtitles"] is True
        assert opts["writeautomaticsub"] is True

    def test_build_ydl_opts_with_thumbnail(self):
        """Test options with thumbnail enabled."""
        node = YtDlpDownload(thumbnail=True)
        opts = node._build_ydl_opts("/tmp/test")

        assert opts["writethumbnail"] is True

    def test_build_ydl_opts_custom_container(self):
        """Test options with custom container format."""
        node = YtDlpDownload(mode="video", container="mkv")
        opts = node._build_ydl_opts("/tmp/test")

        assert opts["merge_output_format"] == "mkv"

    def test_build_ydl_opts_no_auth(self):
        """Test that authentication is disabled."""
        node = YtDlpDownload()
        opts = node._build_ydl_opts("/tmp/test")

        assert opts["username"] is None
        assert opts["password"] is None


class TestYtDlpDownloadMetadata:
    """Tests for metadata extraction."""

    def test_extract_metadata_basic(self, sample_info_dict):
        """Test basic metadata extraction."""
        node = YtDlpDownload()
        metadata = node._extract_metadata(sample_info_dict)

        assert metadata["id"] == "test_video_id"
        assert metadata["title"] == "Test Video Title"
        assert metadata["description"] == "Test description"
        assert metadata["duration"] == 120

    def test_extract_metadata_empty_input(self):
        """Test metadata extraction with empty input."""
        node = YtDlpDownload()
        metadata = node._extract_metadata({})
        assert metadata == {}

    def test_extract_metadata_none_input(self):
        """Test metadata extraction with None input."""
        node = YtDlpDownload()
        metadata = node._extract_metadata(None)
        assert metadata == {}

    def test_extract_metadata_filters_sensitive_data(self, sample_info_dict):
        """Test that sensitive data is not included."""
        sample_info_dict["_filename"] = "/path/to/file"
        sample_info_dict["cookies"] = "sensitive"
        sample_info_dict["http_headers"] = {"auth": "secret"}

        node = YtDlpDownload()
        metadata = node._extract_metadata(sample_info_dict)

        assert "_filename" not in metadata
        assert "cookies" not in metadata
        assert "http_headers" not in metadata


class TestYtDlpDownloadFileHandling:
    """Tests for file handling utilities."""

    def test_find_media_file_by_id(self):
        """Test finding media file by video ID."""
        node = YtDlpDownload()

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            video_path = os.path.join(temp_dir, "test_id.mp4")
            with open(video_path, "wb") as f:
                f.write(b"test video data")

            result = node._find_media_file(temp_dir, "test_id", ["mp4"])
            assert result == video_path

    def test_find_media_file_by_extension(self):
        """Test finding media file by extension."""
        node = YtDlpDownload()

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test file
            video_path = os.path.join(temp_dir, "video.webm")
            with open(video_path, "wb") as f:
                f.write(b"test video data")

            result = node._find_media_file(temp_dir, "", ["webm", "mp4"])
            assert result == video_path

    def test_find_media_file_not_found(self):
        """Test when media file is not found."""
        node = YtDlpDownload()

        with tempfile.TemporaryDirectory() as temp_dir:
            result = node._find_media_file(temp_dir, "test_id", ["mp4"])
            assert result is None

    def test_cleanup_temp_dir(self):
        """Test temp directory cleanup."""
        node = YtDlpDownload()

        temp_dir = tempfile.mkdtemp()
        test_file = os.path.join(temp_dir, "test.txt")
        with open(test_file, "w") as f:
            f.write("test")

        node._cleanup_temp_dir(temp_dir)
        assert not os.path.exists(temp_dir)


class TestYtDlpDownloadIntegration:
    """Integration tests with mocked yt-dlp."""

    @pytest.mark.asyncio
    async def test_video_mode_success(self, context, sample_info_dict):
        """Test successful video download."""
        node = YtDlpDownload(url="https://example.com/video", mode="video")

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a mock video file
            video_path = os.path.join(temp_dir, "test_video_id.mp4")
            with open(video_path, "wb") as f:
                f.write(b"mock video data")

            # Mock the yt-dlp execution
            with patch.object(node, "_run_ytdlp", return_value=sample_info_dict):
                with patch.object(
                    node, "_build_ydl_opts", return_value={"outtmpl": temp_dir}
                ):
                    with patch("tempfile.mkdtemp", return_value=temp_dir):
                        with patch.object(node, "_cleanup_temp_dir"):
                            result = await node.process(context)

        assert "metadata" in result
        assert result["metadata"]["id"] == "test_video_id"

    @pytest.mark.asyncio
    async def test_metadata_mode_success(self, context, sample_info_dict):
        """Test successful metadata-only extraction."""
        node = YtDlpDownload(url="https://example.com/video", mode="metadata")

        with patch.object(node, "_run_ytdlp", return_value=sample_info_dict):
            with patch("tempfile.mkdtemp", return_value="/tmp/mock"):
                with patch.object(node, "_cleanup_temp_dir"):
                    result = await node.process(context)

        assert result["metadata"]["title"] == "Test Video Title"
        assert result["video"].is_empty()
        assert result["audio"].is_empty()

    @pytest.mark.asyncio
    async def test_timeout_handling(self, context):
        """Test timeout handling."""
        node = YtDlpDownload(url="https://example.com/video", timeout=1)

        async def slow_download(*args, **kwargs):
            import asyncio

            await asyncio.sleep(10)

        with patch("asyncio.to_thread", slow_download):
            with patch("tempfile.mkdtemp", return_value="/tmp/mock"):
                with patch.object(node, "_cleanup_temp_dir"):
                    with pytest.raises(DownloadError, match="timed out"):
                        await node.process(context)


class TestYtDlpDownloadOutputs:
    """Tests for output structure."""

    def test_output_type_structure(self):
        """Test that OutputType has correct structure."""
        output_type = YtDlpDownload.OutputType

        # Verify all required keys are present in TypedDict
        assert "video" in output_type.__annotations__
        assert "audio" in output_type.__annotations__
        assert "metadata" in output_type.__annotations__
        assert "subtitles" in output_type.__annotations__
        assert "thumbnail" in output_type.__annotations__
