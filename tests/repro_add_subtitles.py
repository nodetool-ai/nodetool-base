import sys
import os
import shutil
import tempfile
import asyncio
from unittest.mock import MagicMock, patch
import numpy as np
import PIL.Image
import PIL.ImageFont
import PIL.ImageDraw

# Mock dependencies
sys.modules['nodetool'] = MagicMock()
sys.modules['nodetool.config'] = MagicMock()
sys.modules['nodetool.config.logging_config'] = MagicMock()
sys.modules['nodetool.workflows'] = MagicMock()
sys.modules['nodetool.workflows.io'] = MagicMock()
sys.modules['nodetool.workflows.types'] = MagicMock()
sys.modules['nodetool.workflows.processing_context'] = MagicMock()
sys.modules['nodetool.workflows.base_node'] = MagicMock()
sys.modules['nodetool.metadata'] = MagicMock()
sys.modules['nodetool.metadata.types'] = MagicMock()
sys.modules['nodetool.config.environment'] = MagicMock()
sys.modules['nodetool.providers'] = MagicMock()
sys.modules['nodetool.providers.types'] = MagicMock()
sys.modules['ffmpeg'] = MagicMock()

from pydantic import BaseModel, Field, ConfigDict
import enum

# Define real classes for testing
class BaseNode(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    id: str = "test_node"

class ProcessingContext:
    async def asset_to_io(self, asset):
        # Create a dummy file
        f = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        f.write(b"dummy video content")
        f.close()
        return open(f.name, "rb")

    def get_system_font_path(self, font_name):
        # Use a path that likely doesn't exist but PIL handles it or mock PIL.ImageFont
        return "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

    async def video_from_io(self, f):
        return VideoRef(uri="processed_video_uri")

class VideoRef(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    uri: str = ""
    def is_empty(self): return not self.uri

class AudioChunk(BaseModel):
    timestamp: tuple[float, float]
    text: str

class FontRef(BaseModel):
    name: str

class ColorRef(BaseModel):
    value: str

class ImageRef(BaseModel):
    pass

class FolderRef(BaseModel):
    pass

class VideoModel(BaseModel):
    pass

class Provider(enum.Enum):
    Gemini = "gemini"

# Mock cv2
mock_cv2 = MagicMock()
mock_cv2.CAP_PROP_FPS = 1
mock_cv2.CAP_PROP_FRAME_WIDTH = 2
mock_cv2.CAP_PROP_FRAME_HEIGHT = 3
mock_cv2.COLOR_BGR2RGB = 4
mock_cv2.COLOR_RGB2BGR = 5
mock_cv2.VideoWriter_fourcc.return_value = 0

# Mock VideoCapture
class MockVideoCapture:
    def __init__(self, path):
        self.frame_count = 0
        self.total_frames = 5
        self.fps = 30.0
        self.width = 100
        self.height = 100

    def get(self, prop):
        if prop == mock_cv2.CAP_PROP_FPS:
            return self.fps
        if prop == mock_cv2.CAP_PROP_FRAME_WIDTH:
            return self.width
        if prop == mock_cv2.CAP_PROP_FRAME_HEIGHT:
            return self.height
        return 0

    def read(self):
        if self.frame_count >= self.total_frames:
            return False, None

        # Create a black frame
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.frame_count += 1
        return True, frame

    def release(self):
        pass

mock_cv2.VideoCapture = MockVideoCapture
mock_cv2.VideoWriter = MagicMock()
mock_cv2.cvtColor.side_effect = lambda img, code: img # Pass through

# Patch sys.modules
sys.modules['cv2'] = mock_cv2

# Assign types
nodetool_types = sys.modules['nodetool.metadata.types']
nodetool_types.VideoRef = VideoRef
nodetool_types.AudioChunk = AudioChunk
nodetool_types.FontRef = FontRef
nodetool_types.ColorRef = ColorRef
nodetool_types.ImageRef = ImageRef
nodetool_types.FolderRef = FolderRef
nodetool_types.VideoModel = VideoModel
nodetool_types.Provider = Provider

sys.modules['nodetool.workflows.base_node'].BaseNode = BaseNode
sys.modules['nodetool.workflows.processing_context'].ProcessingContext = ProcessingContext
sys.modules['nodetool.workflows.processing_context'].create_file_uri = lambda x: f"file://{x}"
sys.modules['nodetool.config.environment'].Environment.is_production.return_value = False

# Import module
import src.nodetool.nodes.nodetool.video as video_module

# Override _require_cv2 to return our mock
video_module._require_cv2 = lambda: mock_cv2

# Mock PIL.ImageFont.truetype because the font path might not exist
original_truetype = PIL.ImageFont.truetype
def mock_truetype(font=None, size=10, index=0, encoding='', layout_engine=None):
    try:
        return original_truetype(font, size, index, encoding, layout_engine)
    except IOError:
        return PIL.ImageFont.load_default()

PIL.ImageFont.truetype = mock_truetype

async def run_test():
    node = video_module.AddSubtitles()
    node.video = VideoRef(uri="test.mp4")
    node.chunks = [AudioChunk(timestamp=(0.0, 1.0), text="Hello World")]
    node.font = FontRef(name="Arial")
    node.font_size = 10
    node.font_color = ColorRef(value="#FFFFFF")

    context = ProcessingContext()

    # We expect an error because of ffmpeg usage at the end, but we can catch it.
    # We are interested if the frame processing loop runs without error.

    try:
        await node.process(context)
    except RuntimeError as e:
        print(f"Caught expected RuntimeError (likely ffmpeg): {e}")
    except Exception as e:
        print(f"Caught unexpected exception: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise

    print("Test finished execution")

if __name__ == "__main__":
    asyncio.run(run_test())
