from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode


class SaveAudio(GraphNode):
    """
    Save an audio file to a specified folder.
    audio, folder, name

    Use cases:
    - Save generated audio files with timestamps
    - Organize outputs into specific folders
    - Create backups of generated audio
    """

    audio: types.AudioRef | GraphNode | tuple[GraphNode, str] = Field(default=types.AudioRef(type='audio', uri='', asset_id=None, data=None), description=None)
    folder: types.FolderRef | GraphNode | tuple[GraphNode, str] = Field(default=types.FolderRef(type='folder', uri='', asset_id=None, data=None), description='The folder to save the audio file to. ')
    name: str | GraphNode | tuple[GraphNode, str] = Field(default='%Y-%m-%d-%H-%M-%S.opus', description='\n        The name of the audio file.\n        You can use time and date variables to create unique names:\n        %Y - Year\n        %m - Month\n        %d - Day\n        %H - Hour\n        %M - Minute\n        %S - Second\n        ')

    @classmethod
    def get_node_type(cls): return "nodetool.audio.SaveAudio"


