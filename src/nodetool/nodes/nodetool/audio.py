from datetime import datetime
import io
from pydantic import Field
from nodetool.metadata.types import AudioRef
from nodetool.metadata.types import FolderRef
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.common.audio_helpers import normalize_audio, remove_silence

import numpy as np
from pydub import AudioSegment
from nodetool.metadata.types import NPArray


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
        list_assets = await context.list_assets(
            parent_id=parent_id, content_type="audio"
        )
        for asset in list_assets.assets:
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


class Concat(BaseNode):
    """
    Concatenates two audio files together.
    audio, edit, join, +

    Use cases:
    - Combine multiple audio clips into a single file
    - Create longer audio tracks from shorter segments
    """

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


class Normalize(BaseNode):
    """
    Normalizes the volume of an audio file.
    audio, fix, dynamics, volume

    Use cases:
    - Ensure consistent volume across multiple audio files
    - Adjust overall volume level before further processing
    """

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


class Trim(BaseNode):
    """
    Trim an audio file to a specified duration.
    audio, trim, cut

    Use cases:
    - Remove silence from the beginning or end of audio files
    - Extract specific segments from audio files
    - Prepare audio data for machine learning models
    """

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
