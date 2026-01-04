import io
import os
import datetime
import base64
from typing import AsyncGenerator, TypedDict, ClassVar
from nodetool.config.environment import Environment
from nodetool.config.logging_config import get_logger
from nodetool.io.uri_utils import create_file_uri
from pydantic import Field
from nodetool.metadata.types import AudioRef, TTSModel, Provider
from nodetool.metadata.types import FolderRef
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import Chunk, SaveUpdate
from nodetool.workflows.io import NodeInputs, NodeOutputs
from nodetool.media.audio.audio_helpers import normalize_audio, remove_silence

import numpy as np
from pydub import AudioSegment
from nodetool.metadata.types import NPArray

log = get_logger(__name__)


class LoadAudioAssets(BaseNode):
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
        return "Load Audio Assets"

    class OutputType(TypedDict):
        audio: AudioRef
        name: str

    async def gen_process(
        self, context: ProcessingContext
    ) -> AsyncGenerator[OutputType, None]:
        if self.folder.is_empty():
            raise ValueError("Please select an asset folder.")

        parent_id = self.folder.asset_id
        list_assets, _ = await context.list_assets(
            parent_id=parent_id, content_type="audio"
        )
        for asset in list_assets:
            yield {
                "name": asset.name,
                "audio": AudioRef(
                    type="audio",
                    uri=await context.get_asset_url(asset.id),
                    asset_id=asset.id,
                ),
            }


class LoadAudioFile(BaseNode):
    """
    Read an audio file from disk.
    audio, input, load, file

    Use cases:
    - Load audio for processing
    - Import sound files for editing
    - Read audio assets for a workflow
    """

    path: str = Field(default="", description="Path to the audio file to read")

    async def process(self, context: ProcessingContext) -> AudioRef:
        if Environment.is_production():
            raise ValueError("This node is not available in production")
        if not self.path:
            raise ValueError("path cannot be empty")
        expanded_path = os.path.expanduser(self.path)
        if not os.path.exists(expanded_path):
            raise ValueError(f"Audio file not found: {expanded_path}")

        with open(expanded_path, "rb") as f:
            audio_data = f.read()

        audio = await context.audio_from_bytes(audio_data)
        audio.uri = create_file_uri(expanded_path)
        return audio


class LoadAudioFolder(BaseNode):
    """
    Load all audio files from a folder, optionally including subfolders.
    audio, load, folder, files

    Use cases:
    - Batch import audio for processing
    - Build datasets from a directory tree
    - Iterate over audio collections
    """

    folder: str = Field(default="", description="Folder to scan for audio files")
    include_subdirectories: bool = Field(
        default=False, description="Include audio in subfolders"
    )
    extensions: list[str] = Field(
        default=[".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"],
        description="Audio file extensions to include",
    )

    @classmethod
    def get_title(cls):
        return "Load Audio Folder"

    class OutputType(TypedDict):
        audio: AudioRef
        path: str

    async def gen_process(
        self, context: ProcessingContext
    ) -> AsyncGenerator[OutputType, None]:
        if Environment.is_production():
            raise ValueError("This node is not available in production")
        if not self.folder:
            raise ValueError("folder cannot be empty")

        expanded_folder = os.path.expanduser(self.folder)
        if not os.path.isdir(expanded_folder):
            raise ValueError(f"Folder does not exist: {expanded_folder}")

        allowed_exts = {ext.lower() for ext in self.extensions}

        def iter_files(base_folder: str):
            if self.include_subdirectories:
                for root, _, files in os.walk(base_folder):
                    for f in files:
                        yield os.path.join(root, f)
            else:
                for f in os.listdir(base_folder):
                    yield os.path.join(base_folder, f)

        for file_path in iter_files(expanded_folder):
            if not os.path.isfile(file_path):
                continue

            _, ext = os.path.splitext(file_path)
            if ext.lower() not in allowed_exts:
                continue

            with open(file_path, "rb") as f:
                audio_data = f.read()

            audio = await context.audio_from_bytes(audio_data)
            audio.uri = create_file_uri(file_path)
            yield {"path": file_path, "audio": audio}


class SaveAudio(BaseNode):
    """
    Save an audio file to a specified asset folder.
    audio, folder, name

    Use cases:
    - Save generated audio files with timestamps
    - Organize outputs into specific folders
    - Create backups of generated audio
    """

    _expose_as_tool = True

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
        name = datetime.datetime.now().strftime(self.name)
        result = await context.audio_from_segment(audio, name, parent_id=parent_id)

        # Emit SaveUpdate event
        context.post_message(SaveUpdate(
            node_id=self.id,
            name=name,
            value=result,
            output_type="audio"
        ))

        return result


class SaveAudioFile(BaseNode):
    """
    Write an audio file to disk.
    audio, output, save, file

    The filename can include time and date variables:
    %Y - Year, %m - Month, %d - Day
    %H - Hour, %M - Minute, %S - Second
    """

    audio: AudioRef = Field(default=AudioRef(), description="The audio to save")
    folder: str = Field(default="", description="Folder where the file will be saved")
    filename: str = Field(
        default="",
        description="""
        Name of the file to save.
        You can use time and date variables to create unique names:
        %Y - Year
        %m - Month
        %d - Day
        %H - Hour
        %M - Minute
        %S - Second
        """,
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        if Environment.is_production():
            raise ValueError("This node is not available in production")
        if not self.folder:
            raise ValueError("folder cannot be empty")
        if not self.filename:
            raise ValueError("filename cannot be empty")

        expanded_folder = os.path.expanduser(self.folder)
        if not os.path.exists(expanded_folder):
            raise ValueError(f"Folder does not exist: {expanded_folder}")

        filename = datetime.datetime.now().strftime(self.filename)
        expanded_path = os.path.join(expanded_folder, filename)
        os.makedirs(os.path.dirname(expanded_path), exist_ok=True)

        audio_io = await context.asset_to_io(self.audio)
        audio_data = audio_io.read()
        with open(expanded_path, "wb") as f:
            f.write(audio_data)
        result = AudioRef(uri=create_file_uri(expanded_path), data=audio_data)

        # Emit SaveUpdate event
        context.post_message(SaveUpdate(
            node_id=self.id,
            name=filename,
            value=result,
            output_type="audio"
        ))

        return result


class Normalize(BaseNode):
    """
    Normalizes the volume of an audio file.
    audio, fix, dynamics, volume

    Use cases:
    - Ensure consistent volume across multiple audio files
    - Adjust overall volume level before further processing
    """

    _expose_as_tool = True

    audio: AudioRef = Field(
        default=AudioRef(), description="The audio file to normalize."
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        audio = await context.audio_to_audio_segment(self.audio)
        res = normalize_audio(audio)
        return await context.audio_from_segment(res)


class OverlayAudio(BaseNode):
    """
    Overlays two audio files together.
    audio, edit, transform

    Use cases:
    - Mix background music with voice recording
    - Layer sound effects over an existing audio track
    """

    _expose_as_tool = True

    a: AudioRef = Field(default=AudioRef(), description="The first audio file.")
    b: AudioRef = Field(default=AudioRef(), description="The second audio file.")

    async def process(self, context: ProcessingContext) -> AudioRef:
        audio_a = await context.audio_to_audio_segment(self.a)
        audio_b = await context.audio_to_audio_segment(self.b)
        res = audio_a.overlay(audio_b)
        return await context.audio_from_segment(res)


class RemoveSilence(BaseNode):
    """
    Removes or shortens silence in an audio file with smooth transitions.
    audio, edit, clean

    Use cases:
    - Trim silent parts from beginning/end of recordings
    - Remove or shorten long pauses between speech segments
    - Apply crossfade for smooth transitions
    """

    _expose_as_tool = True

    audio: AudioRef = Field(
        default=AudioRef(), description="The audio file to process."
    )
    min_length: int = Field(
        default=200,
        description="Minimum length of silence to be processed (in milliseconds).",
        ge=0,
        le=10000,
    )
    threshold: int = Field(
        default=-40,
        description="Silence threshold in dB (relative to full scale). Higher values detect more silence.",
        ge=-60.0,
        le=0,
    )
    reduction_factor: float = Field(
        default=1.0,
        description="Factor to reduce silent parts (0.0 to 1.0). 0.0 keeps silence as is, 1.0 removes it completely.",
        ge=0.0,
        le=1.0,
    )
    crossfade: int = Field(
        default=10,
        description="Duration of crossfade in milliseconds to apply between segments for smooth transitions.",
        ge=0,
        le=50,
    )
    min_silence_between_parts: int = Field(
        default=100,
        description="Minimum silence duration in milliseconds to maintain between non-silent segments",
        ge=0,
        le=500,
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        audio = await context.audio_to_audio_segment(self.audio)
        res = remove_silence(
            audio,
            min_length=self.min_length,
            threshold=self.threshold,
            reduction_factor=self.reduction_factor,
            crossfade=self.crossfade,
            min_silence_between_parts=self.min_silence_between_parts,
        )
        return await context.audio_from_segment(res)


class SliceAudio(BaseNode):
    """
    Extracts a section of an audio file.
    audio, edit, trim

    Use cases:
    - Cut out a specific clip from a longer audio file
    - Remove unwanted portions from beginning or end
    """

    _expose_as_tool = True

    audio: AudioRef = Field(default=AudioRef(), description="The audio file.")
    start: float = Field(default=0.0, description="The start time in seconds.", ge=0.0)
    end: float = Field(default=1.0, description="The end time in seconds.", ge=0.0)

    async def process(self, context: ProcessingContext) -> AudioRef:
        import pydub

        audio = await context.audio_to_audio_segment(self.audio)
        res = audio[(self.start * 1000) : (self.end * 1000)]
        assert isinstance(res, pydub.AudioSegment)
        return await context.audio_from_segment(res)


class MonoToStereo(BaseNode):
    """
    Converts a mono audio signal to stereo.
    audio, convert, channels

    Use cases:
    - Expand mono recordings for stereo playback systems
    - Prepare audio for further stereo processing
    """

    _expose_as_tool = True

    audio: AudioRef = Field(
        default=AudioRef(), description="The mono audio file to convert."
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        audio = await context.audio_to_audio_segment(self.audio)

        if audio.channels == 1:
            stereo_audio = audio.set_channels(2)
        else:
            # If already stereo or multi-channel, return as is
            stereo_audio = audio

        return await context.audio_from_segment(stereo_audio)


class StereoToMono(BaseNode):
    """
    Converts a stereo audio signal to mono.
    audio, convert, channels

    Use cases:
    - Reduce file size for mono-only applications
    - Simplify audio for certain processing tasks
    """

    _expose_as_tool = True

    audio: AudioRef = Field(
        default=AudioRef(), description="The stereo audio file to convert."
    )
    method: str = Field(
        default="average",
        description="Method to use for conversion: 'average', 'left', or 'right'.",
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        audio = await context.audio_to_audio_segment(self.audio)

        if audio.channels > 1:
            if self.method == "average":
                mono_audio = audio.set_channels(1)
            elif self.method == "left":
                mono_audio = audio.split_to_mono()[0]
            elif self.method == "right":
                mono_audio = audio.split_to_mono()[1]
            else:
                raise ValueError(
                    "Invalid method. Choose 'average', 'left', or 'right'."
                )
        else:
            # If already mono, return as is
            mono_audio = audio

        return await context.audio_from_segment(mono_audio)


class Reverse(BaseNode):
    """
    Reverses an audio file.
    audio, edit, transform

    Use cases:
    - Create reverse audio effects
    - Generate backwards speech or music
    """

    _expose_as_tool = True

    audio: AudioRef = Field(
        default=AudioRef(), description="The audio file to reverse."
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        audio = await context.audio_to_audio_segment(self.audio)
        reversed_audio = audio.reverse()
        return await context.audio_from_segment(reversed_audio)


class FadeIn(BaseNode):
    """
    Applies a fade-in effect to the beginning of an audio file.
    audio, edit, transition

    Use cases:
    - Create smooth introductions to audio tracks
    - Gradually increase volume at the start of a clip
    """

    _expose_as_tool = True

    audio: AudioRef = Field(
        default=AudioRef(), description="The audio file to apply fade-in to."
    )
    duration: float = Field(
        default=1.0, description="Duration of the fade-in effect in seconds.", ge=0.0
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        audio = await context.audio_to_audio_segment(self.audio)
        faded_audio = audio.fade_in(duration=int(self.duration * 1000))
        return await context.audio_from_segment(faded_audio)


class FadeOut(BaseNode):
    """
    Applies a fade-out effect to the end of an audio file.
    audio, edit, transition

    Use cases:
    - Create smooth endings to audio tracks
    - Gradually decrease volume at the end of a clip
    """

    _expose_as_tool = True

    audio: AudioRef = Field(
        default=AudioRef(), description="The audio file to apply fade-out to."
    )
    duration: float = Field(
        default=1.0, description="Duration of the fade-out effect in seconds.", ge=0.0
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        audio = await context.audio_to_audio_segment(self.audio)
        faded_audio = audio.fade_out(duration=int(self.duration * 1000))
        return await context.audio_from_segment(faded_audio)


class Repeat(BaseNode):
    """
    Loops an audio file a specified number of times.
    audio, edit, repeat

    Use cases:
    - Create repeating background sounds or music
    - Extend short audio clips to fill longer durations
    - Generate rhythmic patterns from short samples
    """

    _expose_as_tool = True

    audio: AudioRef = Field(default=AudioRef(), description="The audio file to loop.")
    loops: int = Field(
        default=2,
        ge=1,
        le=100,
        description="Number of times to loop the audio. Minimum 1 (plays once), maximum 100.",
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        audio = await context.audio_to_audio_segment(self.audio)

        # Create the looped audio
        looped_audio = audio * self.loops

        return await context.audio_from_segment(looped_audio)


class AudioMixer(BaseNode):
    """
    Mix up to 5 audio tracks together with individual volume controls.
    audio, mix, volume, combine, blend, layer, add, overlay

    Use cases:
    - Mix multiple audio tracks into a single output
    - Create layered soundscapes
    - Combine music, voice, and sound effects
    - Adjust individual track volumes
    """

    track1: AudioRef = Field(
        default=AudioRef(), description="First audio track to mix."
    )
    track2: AudioRef = Field(
        default=AudioRef(), description="Second audio track to mix."
    )
    track3: AudioRef = Field(
        default=AudioRef(), description="Third audio track to mix."
    )
    track4: AudioRef = Field(
        default=AudioRef(), description="Fourth audio track to mix."
    )
    track5: AudioRef = Field(
        default=AudioRef(), description="Fifth audio track to mix."
    )
    volume1: float = Field(
        default=1.0,
        description="Volume for track 1. 1.0 is original volume.",
        ge=0,
        le=2,
    )
    volume2: float = Field(
        default=1.0,
        description="Volume for track 2. 1.0 is original volume.",
        ge=0,
        le=2,
    )
    volume3: float = Field(
        default=1.0,
        description="Volume for track 3. 1.0 is original volume.",
        ge=0,
        le=2,
    )
    volume4: float = Field(
        default=1.0,
        description="Volume for track 4. 1.0 is original volume.",
        ge=0,
        le=2,
    )
    volume5: float = Field(
        default=1.0,
        description="Volume for track 5. 1.0 is original volume.",
        ge=0,
        le=2,
    )

    async def process(self, context: ProcessingContext) -> AudioRef:

        # Initialize mixed track
        mixed_track = None

        # List of tracks and their volumes
        tracks = [
            (self.track1, self.volume1),
            (self.track2, self.volume2),
            (self.track3, self.volume3),
            (self.track4, self.volume4),
            (self.track5, self.volume5),
        ]

        # Process each track
        for track, volume in tracks:
            if track.is_empty():
                continue

            # Load and adjust volume of current track
            current = await context.audio_to_audio_segment(track)
            if volume != 1.0:
                current = current.apply_gain(volume - 1.0)

            # Add to mixed track
            if mixed_track is None:
                mixed_track = current
            else:
                mixed_track = mixed_track.overlay(current)

        if mixed_track is None:
            raise ValueError("At least one audio track must be provided")

        return await context.audio_from_segment(mixed_track)


class AudioToNumpy(BaseNode):
    """
    Convert audio to numpy array for processing.
    audio, numpy, convert, array

    Use cases:
    - Prepare audio for custom processing
    - Convert audio for machine learning models
    - Extract raw audio data for analysis
    """

    audio: AudioRef = Field(
        default=AudioRef(), description="The audio to convert to numpy."
    )

    class OutputType(TypedDict):
        array: NPArray
        sample_rate: int
        channels: int

    async def process(self, context: ProcessingContext) -> OutputType:
        array, sample_rate, channels = await context.audio_to_numpy(self.audio)
        return {
            "array": NPArray(value=array.tolist(), dtype=str(array.dtype)),
            "sample_rate": sample_rate,
            "channels": channels,
        }


class NumpyToAudio(BaseNode):
    """
    Convert numpy array to audio.
    audio, numpy, convert

    Use cases:
    - Convert processed audio data back to audio format
    - Create audio from machine learning model outputs
    - Generate audio from synthesized waveforms
    """

    array: NPArray = Field(
        default=NPArray(), description="The numpy array to convert to audio."
    )
    sample_rate: int = Field(default=44100, description="Sample rate in Hz.")
    channels: int = Field(default=1, description="Number of audio channels (1 or 2).")

    async def process(self, context: ProcessingContext) -> AudioRef:
        samples = np.array(self.array.value, dtype=self.array.dtype or "int16")

        # Create audio segment from numpy array
        audio_segment = AudioSegment(
            samples.tobytes(),
            frame_rate=self.sample_rate,
            sample_width=samples.dtype.itemsize,
            channels=self.channels,
        )

        # Export to bytes
        buffer = io.BytesIO()
        audio_segment.export(buffer, format="wav")
        audio_bytes = buffer.getvalue()

        return await context.audio_from_bytes(audio_bytes)


class Trim(BaseNode):
    """
    Trim an audio file to a specified duration.
    audio, trim, cut

    Use cases:
    - Remove silence from the beginning or end of audio files
    - Extract specific segments from audio files
    - Prepare audio data for machine learning models
    """

    _expose_as_tool = True

    audio: AudioRef = Field(default=AudioRef(), description="The audio file to trim.")
    start: float = Field(
        default=0.0,
        ge=0.0,
        description="The start time of the trimmed audio in seconds.",
    )
    end: float = Field(
        default=0.0, ge=0.0, description="The end time of the trimmed audio in seconds."
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        audio = await context.audio_to_audio_segment(self.audio)
        audio = audio[self.start * 1000 : self.end * 1000]
        return await context.audio_from_segment(audio)  # type: ignore


class ConvertToArray(BaseNode):
    """
    Converts an audio file to a Array for further processing.
    audio, conversion, tensor

    Use cases:
    - Prepare audio data for machine learning models
    - Enable signal processing operations on audio
    - Convert audio to a format suitable for spectral analysisr
    """

    audio: AudioRef = Field(
        default=AudioRef(), description="The audio file to convert to a tensor."
    )

    async def process(self, context: ProcessingContext) -> NPArray:
        audio = await context.audio_to_audio_segment(self.audio)
        samples = np.array(audio.get_array_of_samples().tolist())
        return NPArray.from_numpy(samples)


class CreateSilence(BaseNode):
    """
    Creates a silent audio file with a specified duration.
    audio, silence, empty

    Use cases:
    - Generate placeholder audio files
    - Create audio segments for padding or spacing
    - Add silence to the beginning or end of audio files
    """

    duration: float = Field(
        default=1.0, ge=0.0, description="The duration of the silence in seconds."
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        audio = AudioSegment.silent(duration=int(self.duration * 1000))
        return await context.audio_from_segment(audio)


class Concat(BaseNode):
    """
    Concatenates two audio files together.
    audio, edit, join, +

    Use cases:
    - Combine multiple audio clips into a single file
    - Create longer audio tracks from shorter segments
    """

    _expose_as_tool = True

    a: AudioRef = Field(default=AudioRef(), description="The first audio file.")
    b: AudioRef = Field(default=AudioRef(), description="The second audio file.")

    async def process(self, context: ProcessingContext) -> AudioRef:
        audio_a = await context.audio_to_audio_segment(self.a)
        audio_b = await context.audio_to_audio_segment(self.b)
        res = audio_a + audio_b
        return await context.audio_from_segment(res)


class ConcatList(BaseNode):
    """
    Concatenates multiple audio files together in sequence.
    audio, edit, join, multiple, +

    Use cases:
    - Combine multiple audio clips into a single file
    - Create longer audio tracks from multiple segments
    - Chain multiple audio files in order
    """

    _expose_as_tool = True

    audio_files: list[AudioRef] = Field(
        default=[], description="List of audio files to concatenate in sequence."
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        if not self.audio_files:
            return AudioRef()

        if len(self.audio_files) == 0:
            raise ValueError("No audio files provided")

        if len(self.audio_files) == 1:
            return self.audio_files[0]

        # Convert first file to base segment
        result = await context.audio_to_audio_segment(self.audio_files[0])

        # Concatenate remaining files in sequence
        for audio_ref in self.audio_files[1:]:
            next_segment = await context.audio_to_audio_segment(audio_ref)
            result = result + next_segment

        return await context.audio_from_segment(result)


class TextToSpeech(BaseNode):
    """
    Generate speech audio from text using any supported TTS provider. Automatically routes to the appropriate backend (OpenAI, HuggingFace, MLX).
    audio, generation, AI, text-to-speech, tts, voice

    Use cases:
    - Create voiceovers for videos and presentations
    - Generate natural-sounding narration for content
    - Build voice assistants and chatbots
    - Convert written content to audio format
    - Create accessible audio versions of text
    """

    _expose_as_tool: ClassVar[bool] = True

    model: TTSModel = Field(
        default=TTSModel(
            provider=Provider.OpenAI,
            id="tts-1",
            name="TTS 1",
            voices=["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
        ),
        description="The text-to-speech model to use",
    )
    text: str = Field(
        default="Hello! This is a text-to-speech demonstration.",
        description="Text to convert to speech",
    )
    speed: float = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
        description="Speech speed multiplier (0.25 to 4.0)",
    )

    class OutputType(TypedDict):
        audio: AudioRef | None
        chunk: Chunk

    async def gen_process(
        self, context: ProcessingContext
    ) -> AsyncGenerator[OutputType, None]:
        # Get the TTS provider for this model
        provider_instance = await context.get_provider(self.model.provider)

        # Use the first voice from the model if no voice is specified
        voice_to_use = self.model.selected_voice
        if not voice_to_use and self.model.voices:
            voice_to_use = self.model.voices[0]

        # Generate speech - provider now yields int16 numpy arrays at 24kHz
        audio_chunks: list[np.ndarray] = []
        async for audio_chunk_array in provider_instance.text_to_speech(
            text=self.text,
            model=self.model.id,
            voice=voice_to_use if voice_to_use else None,
            speed=self.speed,
            context=context,
        ):
            # Store chunk for final audio
            audio_chunks.append(audio_chunk_array)

            # Yield audio chunk as base64-encoded int16 data
            audio_base64 = base64.b64encode(audio_chunk_array.tobytes()).decode(
                "utf-8"
            )
            chunk = Chunk(
                content=audio_base64,
                content_type="audio",
                content_metadata={
                    "sample_rate": 24000,
                    "channels": 1,
                    "dtype": "int16",
                },
                done=False,
            )
            yield {"chunk": chunk, "audio": None}

        # Combine all chunks and create final AudioRef
        if not audio_chunks:
            raise ValueError("No audio data generated")

        # Concatenate all int16 numpy arrays
        combined_array = np.concatenate(audio_chunks) if len(audio_chunks) > 1 else audio_chunks[0]

        # Yield final audio using audio_from_numpy at 24kHz
        yield {
            "audio": await context.audio_from_numpy(combined_array, 24000),
            "chunk": Chunk(content="", done=True, content_type="audio"),
        }

    @classmethod
    def get_basic_fields(cls):
        return ["model", "text", "voice", "speed"]


# =======================================================
# MUST GO INTO OWN PACKAGE BECAUSE OF TORCH DEPENDENCY

# class RealtimeWhisper(BaseNode):
#     """
#     Stream audio input to WhisperLive and emit real-time transcription.
#     realtime, whisper, transcription, streaming, audio-to-text, speech-to-text

#     Emits:
#       - `chunk` Chunk(content=..., done=False) for transcript deltas
#       - `chunk` Chunk(content="", done=True) to mark segment end
#       - `text` final aggregated transcript when input ends
#     """

#     class WhisperModel(str, Enum):
#         TINY = "tiny"
#         BASE = "base"
#         SMALL = "small"
#         MEDIUM = "medium"
#         LARGE = "large"
#         LARGE_V2 = "large-v2"
#         LARGE_V3 = "large-v3"

#     class Language(str, Enum):
#         AUTO = "auto"
#         ENGLISH = "en"
#         SPANISH = "es"
#         FRENCH = "fr"
#         GERMAN = "de"
#         ITALIAN = "it"
#         PORTUGUESE = "pt"
#         DUTCH = "nl"
#         RUSSIAN = "ru"
#         CHINESE = "zh"
#         JAPANESE = "ja"
#         KOREAN = "ko"
#         ARABIC = "ar"
#         HINDI = "hi"
#         TURKISH = "tr"
#         POLISH = "pl"
#         UKRAINIAN = "uk"
#         VIETNAMESE = "vi"

#     model: WhisperModel = Field(
#         default=WhisperModel.TINY,
#         description="Whisper model size - larger models are more accurate but slower",
#     )
#     language: Language = Field(
#         default=Language.ENGLISH,
#         description="Language code for transcription, or 'auto' for automatic detection",
#     )
#     chunk: Chunk = Field(
#         default=Chunk(),
#         description="The audio chunk to transcribe",
#     )
#     temperature: float = Field(
#         default=0.0,
#         ge=0.0,
#         le=1.0,
#         description="Sampling temperature for transcription",
#     )
#     initial_prompt: str = Field(
#         default="",
#         description="Optional initial prompt to guide transcription style",
#     )

#     @classmethod
#     def is_cacheable(cls) -> bool:
#         return False

#     @classmethod
#     def is_streaming_output(cls) -> bool:
#         return True

#     @classmethod
#     def is_streaming_input(cls) -> bool:
#         return True

#     @classmethod
#     def return_type(cls):
#         return cls.OutputType

#     class OutputType(TypedDict):
#         start: float
#         end: float
#         text: str
#         chunk: Chunk
#         speaker: int
#         detected_language: str
#         translation: str

#     async def run(
#         self,
#         context: ProcessingContext,
#         inputs: NodeInputs,
#         outputs: NodeOutputs,
#     ) -> None:
#         """Process streaming audio input and emit real-time transcription.

#         Args:
#             context: Processing context for the workflow
#             inputs: Streaming audio chunks
#             outputs: Output emitter for transcription chunks and final text
#         """
#         from whisperlivekit import TranscriptionEngine, AudioProcessor

#         log.info(f"Starting RealtimeWhisper with model: {self.model.value}")

#         # Initialize transcription engine
#         transcription_engine = TranscriptionEngine(
#             model=self.model.value,
#             language=self.language.value,
#             temperature=self.temperature,
#             initial_prompt=self.initial_prompt if self.initial_prompt else None,
#             min_chunk_size=0.04,
#             pcm_input=True,
#         )
#         log.debug("TranscriptionEngine initialized")

#         # Create audio processor
#         audio_processor = AudioProcessor(transcription_engine=transcription_engine)
#         log.debug("AudioProcessor created")

#         # Create tasks for results processing
#         results_generator = await audio_processor.create_tasks()
#         log.debug("Results generator created")

#         async def producer_loop():
#             """Read audio chunks from input and feed to processor."""
#             log.debug("Producer loop started")
#             try:
#                 async for handle, item in inputs.any():
#                     if handle != "chunk":
#                         log.error(f"Unknown handle: {handle}")
#                         raise ValueError(f"Unknown handle: {handle}")

#                     assert isinstance(item, Chunk)

#                     # Only process audio chunks
#                     if item.content_type == "audio" and item.content:
#                         # Decode base64 audio if needed
#                         if isinstance(item.content, str):
#                             audio_bytes = base64.b64decode(item.content)
#                         else:
#                             audio_bytes = item.content

#                         log.debug(f"Processing audio chunk: {len(audio_bytes)} bytes")
#                         await audio_processor.process_audio(audio_bytes)

#                 log.debug("Producer loop finished - input stream ended")
#             except Exception as e:
#                 log.error(f"Error in producer loop: {e}", exc_info=True)
#                 raise
#             finally:
#                 # Signal end of audio input
#                 await audio_processor.cleanup()

#         asyncio.create_task(producer_loop())

#         """Consume transcription results and emit chunks."""
#         log.debug("Consume transcription results and emit chunks")
#         aggregated_text = ""

#         async for response in results_generator:
#             if response.error:
#                 raise RuntimeError(f"Transcription error: {response.error}")

#             if not response.lines:
#                 continue

#             # Process each line in the response, but skip the last one (it may still be updating)
#             # Only emit lines that are truly finalized (not the current in-progress line)
#             lines_to_process = response.lines[:-1] if len(response.lines) > 1 else []

#             for line in lines_to_process:
#                 # Skip dummy lines
#                 if getattr(line, 'is_dummy', False):
#                     continue

#                 # Skip placeholder/empty lines
#                 if line.speaker == -2 or not line.text or line.text.strip() == "":
#                     continue

#                 line_text = line.text

#                 # Find overlap: check if line_text is already contained in aggregated_text
#                 if line_text in aggregated_text:
#                     log.debug(f"Skipping duplicate text: {line_text[:50]}...")
#                     continue

#                 # Find if there's an overlap at the end of aggregated_text with the start of line_text
#                 new_text = line_text
#                 max_overlap = min(len(aggregated_text), len(line_text))

#                 for overlap_len in range(max_overlap, 0, -1):
#                     if aggregated_text.endswith(line_text[:overlap_len]):
#                         # Found overlap, only emit the new part
#                         new_text = line_text[overlap_len:]
#                         break

#                 if not new_text.strip():
#                     continue

#                 # Add to aggregated text
#                 aggregated_text += new_text

#                 log.debug(f"Emitting new text: start={line.start}, text={new_text[:50]}...")

#                 await outputs.emit("chunk", Chunk(content=new_text, done=False))
#                 await outputs.emit("start", line.start)
#                 await outputs.emit("end", line.end)
#                 await outputs.emit("speaker", line.speaker)
#                 await outputs.emit("detected_language", line.detected_language or "en")
#                 await outputs.emit("translation", line.translation or "")


#         log.debug("Consumer loop finished - results generator ended")


#         # Emit final aggregated text
#         final_text = " ".join(aggregated_text).strip()
#         if final_text:
#             log.info(f"Emitting final transcript: {len(final_text)} characters")
#             await outputs.emit("text", final_text)

class ChunkToAudio(BaseNode):
    """
    Aggregates audio chunks from an input stream into AudioRef objects.
    audio, stream, chunk, aggregate, collect, batch

    Use cases:
    - Collect streaming audio chunks into larger files for processing
    - buffer realtime audio streams
    """

    chunk: Chunk = Field(default=Chunk(), description="Stream of audio chunks")
    batch_size: int = Field(
        default=50, description="Number of chunks to aggregate per output"
    )

    @classmethod
    def is_streaming_input(cls) -> bool:
        return True

    class OutputType(TypedDict):
        audio: AudioRef

    async def run(
        self, context: ProcessingContext, inputs: NodeInputs, outputs: NodeOutputs
    ) -> None:
        buffer = AudioSegment.empty()
        count = 0

        async for chunk in inputs.stream("chunk"):
            log.info(f"ChunkToAudio received chunk: content_type={chunk.content_type}, metadata={chunk.content_metadata}")
            if chunk.content_type == "audio" and chunk.content:
                try:
                    # Check if content is base64 encoded string
                    if not chunk.content:
                        continue

                    data = base64.b64decode(chunk.content)
                    # Create AudioSegment from bytes
                    # Use metadata if available to handle raw PCM
                    meta = chunk.content_metadata or {}
                    fmt = meta.get("format")
                    
                    if fmt == "pcm16le" or meta.get("encoding") == "pcm16le":
                        segment = AudioSegment(
                            data=data,
                            sample_width=2, # 16-bit
                            frame_rate=meta.get("sample_rate", 44100),
                            channels=meta.get("channels", 1)
                        )
                    else:
                        # Fallback for container formats (mp3, wav, etc.) or unknown
                        segment = AudioSegment.from_file(io.BytesIO(data))
                        
                    buffer += segment
                    count += 1
                except Exception as e:
                    log.error(f"Error decoding chunk: {e}")
                    continue

            if count >= self.batch_size:
                # Flush
                if len(buffer) > 0:
                    audio = await context.audio_from_segment(buffer)
                    await outputs.emit("audio", audio)

                # Reset
                buffer = AudioSegment.empty()
                count = 0

        # Flush remaining
        if count > 0 and len(buffer) > 0:
            audio = await context.audio_from_segment(buffer)
            await outputs.emit("audio", audio)


