import sys
from unittest.mock import MagicMock
from typing import TypedDict, ClassVar
import enum
from pydantic import BaseModel, Field, ConfigDict

# Create dummy classes for inheritance
class BaseNode(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    id: str = "test_node"

class ProcessingContext:
    pass

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

# Function to mock modules recursively
def mock_module(module_name):
    if module_name in sys.modules:
        return sys.modules[module_name]
    m = MagicMock()
    m.__spec__ = MagicMock() # Try to satisfy import machinery
    sys.modules[module_name] = m
    return m

mock_module('nodetool')
mock_module('nodetool.config')
mock_module('nodetool.config.logging_config')
mock_module('nodetool.workflows')
mock_module('nodetool.workflows.io')
mock_module('nodetool.workflows.types')
mock_module('nodetool.workflows.processing_context')
mock_module('nodetool.workflows.base_node')
mock_module('nodetool.metadata')
mock_module('nodetool.metadata.types')
mock_module('nodetool.config.environment')
mock_module('nodetool.providers')
mock_module('nodetool.providers.types')

# Assign real classes to mocks
sys.modules['nodetool.metadata.types'].VideoRef = VideoRef
sys.modules['nodetool.metadata.types'].AudioChunk = AudioChunk
sys.modules['nodetool.metadata.types'].FontRef = FontRef
sys.modules['nodetool.metadata.types'].ColorRef = ColorRef
sys.modules['nodetool.metadata.types'].ImageRef = ImageRef
sys.modules['nodetool.metadata.types'].FolderRef = FolderRef
sys.modules['nodetool.metadata.types'].VideoModel = VideoModel
sys.modules['nodetool.metadata.types'].Provider = Provider

sys.modules['nodetool.workflows.base_node'].BaseNode = BaseNode
sys.modules['nodetool.workflows.processing_context'].ProcessingContext = ProcessingContext
sys.modules['nodetool.workflows.processing_context'].create_file_uri = lambda x: f"file://{x}"

# Setup environment mock
sys.modules['nodetool.config.environment'].Environment.is_production.return_value = False

# Import the target module
# Use absolute import assuming src is in python path
try:
    import nodetool.nodes.nodetool.video as video_module
except ImportError:
    import src.nodetool.nodes.nodetool.video as video_module

# Test instantiation
node = video_module.AddSubtitles()
print("AddSubtitles instantiated successfully")
