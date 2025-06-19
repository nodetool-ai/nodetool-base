from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode


class VideoDownload(GraphNode):
    """Download a video from a URL using ``yt_dlp``.
    youtube, download, video, ytdlp

    Use cases:
    - Save videos for offline processing
    - Fetch clips for video editing workflows
    - Archive online video content
    """

    url: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="URL of the video to download"
    )
    format: str | GraphNode | tuple[GraphNode, str] = Field(
        default="best", description="yt-dlp format string"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.yt_dlp.VideoDownload"
