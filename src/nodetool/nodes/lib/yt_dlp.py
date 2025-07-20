import asyncio
import os
import tempfile
from pydantic import Field
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.metadata.types import VideoRef


class VideoDownload(BaseNode):
    """Download a video from a URL using ``yt_dlp``.
    youtube, download, video, ytdlp

    Use cases:
    - Save videos for offline processing
    - Fetch clips for video editing workflows
    - Archive online video content
    """

    url: str = Field(default="", description="URL of the video to download")
    format: str = Field(default="best", description="yt-dlp format string")

    @classmethod
    def get_title(cls):
        return "Download Video"

    async def process(self, context: ProcessingContext) -> VideoRef:
        if not self.url:
            raise ValueError("url cannot be empty")

        import yt_dlp

        def _download() -> bytes:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                filename = tmp.name
            ydl_opts = {"format": self.format, "outtmpl": filename, "quiet": True}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([self.url])
            with open(filename, "rb") as f:
                data = f.read()
            os.remove(filename)
            return data

        video_bytes = await asyncio.to_thread(_download)
        return await context.video_from_bytes(video_bytes)
