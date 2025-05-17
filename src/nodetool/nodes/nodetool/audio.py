from datetime import datetime
import io
from pydantic import Field
from nodetool.metadata.types import AudioRef
from nodetool.metadata.types import FolderRef
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext


class LoadAudioFolder(BaseNode):
    """
    Load audio files from an asset folder.
    load, audio, file, import
    """

    folder: FolderRef = Field(
        default=FolderRef(),
        description="The asset folder to load the audio files from.",
    )

    @classmethod
    def get_title(cls):
        return "Load Audio Folder"

    @classmethod
    def return_type(cls):
        return {
            "audio": AudioRef,
            "name": str,
        }

    async def gen_process(self, context: ProcessingContext):
        if self.folder.is_empty():
            raise ValueError("Please select an asset folder.")

        parent_id = self.folder.asset_id
        list_assets = await context.list_assets(parent_id=parent_id, mime_type="audio")
        for asset in list_assets.assets:
            if asset.content_type.startswith("audio/"):
                yield "name", asset.name
                yield "audio", AudioRef(
                    type="audio",
                    uri=await context.get_asset_url(asset.id),
                    asset_id=asset.id,
                )


class SaveAudio(BaseNode):
    """
    Save an audio file to a specified asset folder.
    audio, folder, name

    Use cases:
    - Save generated audio files with timestamps
    - Organize outputs into specific folders
    - Create backups of generated audio
    """

    audio: AudioRef = AudioRef()
    folder: FolderRef = Field(
        FolderRef(), description="The asset folder to save the audio file to. "
    )
    name: str = Field(
        default="%Y-%m-%d-%H-%M-%S.opus",
        description="""
        The name of the audio file.
        You can use time and date variables to create unique names:
        %Y - Year
        %m - Month
        %d - Day
        %H - Hour
        %M - Minute
        %S - Second
        """,
    )

    @classmethod
    def get_title(cls):
        return "Save Audio Asset"

    def required_inputs(self):
        return ["audio"]

    async def process(self, context: ProcessingContext) -> AudioRef:
        audio = await context.audio_to_audio_segment(self.audio)
        file = io.BytesIO()
        audio.export(file)
        file.seek(0)
        parent_id = self.folder.asset_id if self.folder.is_set() else None
        name = datetime.now().strftime(self.name)
        return await context.audio_from_segment(audio, name, parent_id=parent_id)
