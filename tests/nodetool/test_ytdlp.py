import types
import os
import sys
import pytest
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.nodes.lib.yt_dlp import VideoDownload
from nodetool.metadata.types import VideoRef

TEST_MP4 = os.path.join(os.path.dirname(__file__), "../test.mp4")


class DummyYDL:
    def __init__(self, opts):
        self.out = opts.get("outtmpl")

    def download(self, urls):
        with open(TEST_MP4, "rb") as src, open(self.out, "wb") as dst:
            dst.write(src.read())

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass


@pytest.fixture
def context():
    return ProcessingContext(user_id="test", auth_token="test")


@pytest.mark.asyncio
async def test_video_download(context, monkeypatch):
    monkeypatch.setitem(
        sys.modules, "yt_dlp", types.SimpleNamespace(YoutubeDL=DummyYDL)
    )
    node = VideoDownload(url="http://example.com/video")
    result = await node.process(context)
    assert isinstance(result, VideoRef)
