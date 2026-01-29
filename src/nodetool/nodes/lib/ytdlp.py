"""
yt-dlp Media Download Node

Provides a node for downloading media from URLs using yt-dlp.
"""

import asyncio
import os
import re
import shutil
import tempfile
from enum import Enum
from io import BytesIO
from typing import ClassVar, TypedDict

import PIL.Image
from pydantic import Field

from nodetool.metadata.types import AudioRef, ImageRef, VideoRef
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext


class DownloadError(Exception):
    """Raised when media download fails."""

    pass


class DownloadMode(str, Enum):
    VIDEO = "video"
    AUDIO = "audio"
    METADATA = "metadata"


class YtDlpDownload(BaseNode):
    """
    Download media from URLs using yt-dlp.
    download, video, audio, youtube, media, yt-dlp, metadata, subtitles

    Use cases:
    - Download videos from YouTube and other platforms
    - Extract audio from video URLs
    - Retrieve video/audio metadata without downloading
    - Download subtitles and thumbnails
    """

    # File extension constants for media type detection
    VIDEO_EXTENSIONS: ClassVar[list[str]] = ["mp4", "webm", "mkv", "avi", "mov", "flv"]
    AUDIO_EXTENSIONS: ClassVar[list[str]] = ["mp3", "m4a", "opus", "ogg", "wav", "webm"]
    SUBTITLE_EXTENSIONS: ClassVar[list[str]] = ["srt", "vtt", "ass", "ssa"]
    THUMBNAIL_EXTENSIONS: ClassVar[list[str]] = ["jpg", "jpeg", "png", "webp"]

    url: str = Field(default="", description="URL of the media to download")
    mode: DownloadMode = Field(
        default=DownloadMode.VIDEO,
        description="Download mode: video, audio, or metadata only",
    )
    format_selector: str = Field(
        default="best",
        description="yt-dlp format selector (e.g., 'best', 'bestvideo+bestaudio')",
    )
    container: str = Field(
        default="auto",
        description="Output container format (e.g., 'mp4', 'webm', 'auto')",
    )
    subtitles: bool = Field(
        default=False, description="Download subtitles if available"
    )
    thumbnail: bool = Field(
        default=False, description="Download thumbnail if available"
    )
    overwrite: bool = Field(default=False, description="Overwrite existing files")
    rate_limit_kbps: int = Field(
        default=0,
        ge=0,
        description="Rate limit in KB/s (0 = unlimited)",
    )
    timeout: int = Field(
        default=600,
        ge=1,
        le=3600,
        description="Timeout in seconds",
    )

    @classmethod
    def get_title(cls) -> str:
        return "YouTube Downloader"

    class OutputType(TypedDict):
        video: VideoRef
        audio: AudioRef
        metadata: dict
        subtitles: str
        thumbnail: ImageRef | None

    async def process(self, context: ProcessingContext) -> OutputType:
        """Process the download request."""
        # Validate URL
        if not self.url or not self.url.strip():
            raise DownloadError("URL cannot be empty")

        url = self.url.strip()
        if not self._is_valid_url(url):
            raise DownloadError(f"Invalid URL format: {url}")

        # Create temp directory for downloads
        temp_dir = tempfile.mkdtemp(prefix="ytdlp_")

        try:
            # Build yt-dlp options
            ydl_opts = self._build_ydl_opts(temp_dir)

            # Run yt-dlp in a thread pool
            info_dict = await asyncio.wait_for(
                asyncio.to_thread(self._run_ytdlp, url, ydl_opts),
                timeout=self.timeout,
            )

            # Process results based on mode
            result: YtDlpDownload.OutputType = {
                "video": VideoRef(),
                "audio": AudioRef(),
                "metadata": {},
                "subtitles": "",
                "thumbnail": None,
            }

            # Always extract metadata when available
            result["metadata"] = self._extract_metadata(info_dict)

            # Handle media based on mode
            if self.mode == DownloadMode.VIDEO:
                result["video"] = await self._process_video(
                    context, temp_dir, info_dict
                )
            elif self.mode == DownloadMode.AUDIO:
                result["audio"] = await self._process_audio(
                    context, temp_dir, info_dict
                )
            # metadata mode: no media to process

            # Handle subtitles if requested
            if self.subtitles:
                result["subtitles"] = await self._process_subtitles(temp_dir, info_dict)

            # Handle thumbnail if requested
            if self.thumbnail:
                result["thumbnail"] = await self._process_thumbnail(
                    context, temp_dir, info_dict
                )

            return result

        except asyncio.TimeoutError:
            raise DownloadError(f"Download timed out after {self.timeout} seconds")
        except Exception as e:
            if isinstance(e, DownloadError):
                raise
            raise DownloadError(f"Download failed: {str(e)}") from e
        finally:
            # Clean up temp directory
            self._cleanup_temp_dir(temp_dir)

    def _is_valid_url(self, url: str) -> bool:
        """Validate URL format."""
        # Basic URL pattern validation
        pattern = r"^https?://[^\s/$.?#].[^\s]*$"
        return bool(re.match(pattern, url, re.IGNORECASE))

    def _build_ydl_opts(self, temp_dir: str) -> dict:
        """Build yt-dlp options dictionary."""
        # Base options for deterministic behavior
        opts: dict = {
            "outtmpl": os.path.join(temp_dir, "%(id)s.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
            "no_color": True,
            "noplaylist": True,
            "extract_flat": False,
            "ignoreerrors": False,
            "socket_timeout": 30,
            "retries": 3,
            "fragment_retries": 3,
            # Ensure deterministic filename without sanitization issues
            "restrictfilenames": True,
            # No authentication or DRM bypass
            "username": None,
            "password": None,
            # Progress hooks disabled for clean output
            "progress_hooks": [],
            "postprocessor_hooks": [],
        }

        # Handle rate limiting
        if self.rate_limit_kbps > 0:
            opts["ratelimit"] = self.rate_limit_kbps * 1024  # Convert to bytes/sec

        # Handle mode-specific options
        if self.mode == DownloadMode.METADATA:
            opts["skip_download"] = True
        elif self.mode == DownloadMode.AUDIO:
            opts["format"] = "bestaudio/best"
            opts["postprocessors"] = [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }
            ]
        else:  # video mode
            if self.format_selector != "best":
                opts["format"] = self.format_selector
            else:
                opts["format"] = "bestvideo+bestaudio/best"

            # Handle container format
            if self.container != "auto":
                opts["merge_output_format"] = self.container

        # Handle subtitles
        if self.subtitles:
            opts["writesubtitles"] = True
            opts["writeautomaticsub"] = True
            opts["subtitlesformat"] = "srt/vtt/ass/best"
            opts["subtitleslangs"] = ["en", "en-US", "en-GB"]

        # Handle thumbnail
        if self.thumbnail:
            opts["writethumbnail"] = True

        # Handle overwrite
        if not self.overwrite:
            opts["nooverwrites"] = True

        return opts

    def _run_ytdlp(self, url: str, opts: dict) -> dict:
        """Run yt-dlp and return info dict."""
        import yt_dlp

        with yt_dlp.YoutubeDL(opts) as ydl:
            try:
                info = ydl.extract_info(url, download=(self.mode != "metadata"))
                if info is None:
                    raise DownloadError("Failed to extract media information")
                return info
            except yt_dlp.utils.DownloadError as e:
                raise DownloadError(f"yt-dlp download error: {str(e)}") from e
            except yt_dlp.utils.ExtractorError as e:
                raise DownloadError(f"yt-dlp extractor error: {str(e)}") from e

    def _extract_metadata(self, info_dict: dict) -> dict:
        """Extract safe metadata from info dict."""
        if not info_dict:
            return {}

        # Extract only safe, non-sensitive metadata
        safe_keys = [
            "id",
            "title",
            "description",
            "uploader",
            "uploader_id",
            "upload_date",
            "duration",
            "view_count",
            "like_count",
            "comment_count",
            "categories",
            "tags",
            "age_limit",
            "webpage_url",
            "original_url",
            "extractor",
            "extractor_key",
            "width",
            "height",
            "fps",
            "vcodec",
            "acodec",
            "abr",
            "vbr",
            "tbr",
            "filesize",
            "filesize_approx",
            "format",
            "format_id",
            "format_note",
            "resolution",
            "aspect_ratio",
            "channel",
            "channel_id",
            "channel_url",
            "playlist",
            "playlist_index",
            "is_live",
            "was_live",
            "language",
        ]

        metadata = {}
        for key in safe_keys:
            if key in info_dict and info_dict[key] is not None:
                value = info_dict[key]
                # Only include JSON-serializable primitive values
                # Exclude dicts and complex nested structures
                if isinstance(value, (str, int, float, bool)):
                    metadata[key] = value
                elif isinstance(value, list):
                    # Only include lists of primitives
                    if all(isinstance(item, (str, int, float, bool)) for item in value):
                        metadata[key] = value

        return metadata

    async def _process_video(
        self, context: ProcessingContext, temp_dir: str, info_dict: dict
    ) -> VideoRef:
        """Process downloaded video file."""
        video_id = info_dict.get("id", "video")
        ext = info_dict.get("ext", "mp4")

        # Find the downloaded video file using class constant
        extensions = list(self.VIDEO_EXTENSIONS)
        if ext not in extensions:
            extensions.append(ext)
        video_path = self._find_media_file(temp_dir, video_id, extensions)
        if not video_path:
            raise DownloadError("Video file not found after download")

        # Read and create VideoRef
        with open(video_path, "rb") as f:
            video_bytes = f.read()

        return await context.video_from_bytes(video_bytes)

    async def _process_audio(
        self, context: ProcessingContext, temp_dir: str, info_dict: dict
    ) -> AudioRef:
        """Process downloaded audio file."""
        video_id = info_dict.get("id", "audio")

        # Find the downloaded audio file using class constant
        audio_path = self._find_media_file(temp_dir, video_id, self.AUDIO_EXTENSIONS)
        if not audio_path:
            raise DownloadError("Audio file not found after download")

        # Read and create AudioRef
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()

        return await context.audio_from_bytes(audio_bytes)

    async def _process_subtitles(self, temp_dir: str, info_dict: dict) -> str:
        """Process downloaded subtitles."""
        subtitle_content = ""

        for filename in os.listdir(temp_dir):
            filepath = os.path.join(temp_dir, filename)
            if not os.path.isfile(filepath):
                continue

            # Check if this is a subtitle file
            _, ext = os.path.splitext(filename)
            ext = ext.lower().lstrip(".")

            if ext in self.SUBTITLE_EXTENSIONS:
                try:
                    # Read as UTF-8
                    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                        content = f.read()
                    # Use the first subtitle file found
                    if content:
                        subtitle_content = content
                        break
                except Exception:
                    continue

        return subtitle_content

    async def _process_thumbnail(
        self, context: ProcessingContext, temp_dir: str, info_dict: dict
    ) -> ImageRef | None:
        """Process downloaded thumbnail."""
        for filename in os.listdir(temp_dir):
            filepath = os.path.join(temp_dir, filename)
            if not os.path.isfile(filepath):
                continue

            _, ext = os.path.splitext(filename)
            ext = ext.lower().lstrip(".")

            if ext in self.THUMBNAIL_EXTENSIONS:
                try:
                    # Load and normalize to PNG
                    with PIL.Image.open(filepath) as img:
                        # Convert to RGB if needed (remove alpha, handle palettes)
                        if img.mode not in ("RGB", "L"):
                            img = img.convert("RGB")

                        # Save as PNG bytes
                        buffer = BytesIO()
                        img.save(buffer, format="PNG")
                        buffer.seek(0)

                        return await context.image_from_bytes(buffer.read())
                except Exception:
                    continue

        return None

    def _find_media_file(
        self, temp_dir: str, video_id: str, extensions: list[str]
    ) -> str | None:
        """Find media file in temp directory."""
        for filename in os.listdir(temp_dir):
            filepath = os.path.join(temp_dir, filename)
            if not os.path.isfile(filepath):
                continue

            _, ext = os.path.splitext(filename)
            ext = ext.lower().lstrip(".")

            # Match by extension
            if ext in extensions:
                # Prefer files containing the video ID
                if video_id and video_id in filename:
                    return filepath

        # Fallback: return any file with matching extension
        for filename in os.listdir(temp_dir):
            filepath = os.path.join(temp_dir, filename)
            if not os.path.isfile(filepath):
                continue

            _, ext = os.path.splitext(filename)
            ext = ext.lower().lstrip(".")

            if ext in extensions:
                return filepath

        return None

    def _cleanup_temp_dir(self, temp_dir: str) -> None:
        """Clean up temporary directory."""
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass  # Best effort cleanup
