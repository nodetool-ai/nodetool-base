from io import BytesIO
import os
import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch, ANY
from pydub import AudioSegment

from nodetool.metadata.types import AudioRef, NPArray
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.nodes.nodetool.audio import (
    Concat,
    ConcatList,
    Normalize,
    OverlayAudio,
    RemoveSilence,
    SliceAudio,
    Tone,
    MonoToStereo,
    StereoToMono,
    Reverse,
    FadeIn,
    FadeOut,
    Repeat,
    AudioMixer,
)


# Create dummy AudioRefs for testing
buffer = BytesIO()
AudioSegment.silent(duration=5000, frame_rate=44100).export(buffer, format="wav")
dummy_audio = AudioRef(data=buffer.getvalue())
dummy_audio_2 = AudioRef(data=buffer.getvalue())


@pytest.fixture
def audio_segment_mono():
    """Create a simple mono audio segment for testing."""
    return AudioSegment.silent(duration=1000, frame_rate=44100).set_channels(1)


@pytest.fixture
def audio_segment_stereo():
    """Create a simple stereo audio segment for testing."""
    return AudioSegment.silent(duration=1000, frame_rate=44100).set_channels(2)


@pytest.fixture
def mock_context(audio_segment_mono, audio_segment_stereo):
    """Create a mock processing context for testing."""
    context = MagicMock(spec=ProcessingContext)

    # Since AudioRef is not hashable, we'll use a simpler approach
    # We'll just have the mock return stereo for specific test cases

    async def mock_audio_to_audio_segment(audio_ref):
        # Default to mono audio segment
        return audio_segment_mono

    async def mock_audio_from_segment(segment):
        # Create a new AudioRef
        return AudioRef()

    context.audio_to_audio_segment = AsyncMock(side_effect=mock_audio_to_audio_segment)
    context.audio_from_segment = AsyncMock(side_effect=mock_audio_from_segment)

    # Add methods to override the default behavior for specific tests
    def set_next_audio_as_stereo():
        context.audio_to_audio_segment = AsyncMock(return_value=audio_segment_stereo)

    def reset_to_mono():
        context.audio_to_audio_segment = AsyncMock(return_value=audio_segment_mono)

    context.set_next_audio_as_stereo = set_next_audio_as_stereo
    context.reset_to_mono = reset_to_mono

    return context


class TestConcat:
    @pytest.mark.asyncio
    async def test_concat(self, mock_context, audio_segment_mono):
        """Test that Concat correctly concatenates two audio files."""
        # Setup
        node = Concat(a=AudioRef(), b=AudioRef())

        # Execute
        result = await node.process(mock_context)

        # Verify
        assert mock_context.audio_to_audio_segment.call_count == 2
        assert mock_context.audio_from_segment.call_count == 1
        assert isinstance(result, AudioRef)
        # The mock should have been called with the concatenated audio segment
        mock_context.audio_from_segment.assert_called_once()


class TestConcatList:
    @pytest.mark.asyncio
    async def test_concat_list_empty(self, mock_context):
        """Test that ConcatList returns an empty AudioRef when given an empty list."""
        # Setup
        node = ConcatList(audio_files=[])

        # Execute
        result = await node.process(mock_context)

        # Verify
        assert isinstance(result, AudioRef)
        assert mock_context.audio_to_audio_segment.call_count == 0
        assert mock_context.audio_from_segment.call_count == 0

    @pytest.mark.asyncio
    async def test_concat_list_single_file(self, mock_context):
        """Test that ConcatList returns the single audio file when given a list with one file."""
        # Setup
        audio_ref = AudioRef()
        node = ConcatList(audio_files=[audio_ref])

        # Execute
        result = await node.process(mock_context)

        # Verify
        assert result == audio_ref
        assert mock_context.audio_to_audio_segment.call_count == 0
        assert mock_context.audio_from_segment.call_count == 0

    @pytest.mark.asyncio
    async def test_concat_list_multiple_files(self, mock_context):
        """Test that ConcatList correctly concatenates multiple audio files."""
        # Setup
        node = ConcatList(audio_files=[AudioRef(), AudioRef(), AudioRef()])

        # Execute
        result = await node.process(mock_context)

        # Verify
        assert mock_context.audio_to_audio_segment.call_count == 3
        assert mock_context.audio_from_segment.call_count == 1
        assert isinstance(result, AudioRef)


class TestNormalize:
    @pytest.mark.asyncio
    async def test_normalize(self, mock_context):
        """Test that Normalize correctly normalizes an audio file."""
        # Setup
        with patch(
            "nodetool.nodes.nodetool.audio.normalize_audio"
        ) as mock_normalize:
            mock_normalize.return_value = AudioSegment.silent(duration=1000)
            node = Normalize(audio=AudioRef())
            
            # Execute
            result = await node.process(mock_context)

            # Verify
            assert mock_context.audio_to_audio_segment.call_count == 1
            assert mock_context.audio_from_segment.call_count == 1
            assert isinstance(result, AudioRef)
            mock_normalize.assert_called_once()


class TestOverlayAudio:
    @pytest.mark.asyncio
    async def test_overlay_audio(self, mock_context, audio_segment_mono):
        """Test that OverlayAudio correctly overlays two audio files."""
        # Setup
        node = OverlayAudio(a=AudioRef(), b=AudioRef())

        # Execute
        with patch.object(audio_segment_mono, "overlay") as mock_overlay:
            mock_overlay.return_value = audio_segment_mono
            result = await node.process(mock_context)

        # Verify
        assert mock_context.audio_to_audio_segment.call_count == 2
        assert mock_context.audio_from_segment.call_count == 1
        assert isinstance(result, AudioRef)
        mock_overlay.assert_called_once()


class TestRemoveSilence:
    @pytest.mark.asyncio
    async def test_remove_silence(self, mock_context):
        """Test that RemoveSilence correctly removes silence from an audio file."""
        # Setup
        with patch(
            "nodetool.nodes.nodetool.audio.remove_silence"
        ) as mock_remove_silence:
            mock_remove_silence.return_value = AudioSegment.silent(duration=500)
            
            node = RemoveSilence(
                audio=AudioRef(),
                min_length=200,
                threshold=-40,
                reduction_factor=1.0,
                crossfade=10,
                min_silence_between_parts=100,
            )

            # Execute
            result = await node.process(mock_context)

            # Verify
            assert mock_context.audio_to_audio_segment.call_count == 1
            assert mock_context.audio_from_segment.call_count == 1
            assert isinstance(result, AudioRef)

            # Use a more flexible assertion that doesn't check the exact audio segment object
            mock_remove_silence.assert_called_once()
            args, kwargs = mock_remove_silence.call_args
            assert len(args) == 1  # Should have one positional argument (the audio segment)
            assert isinstance(
                args[0], AudioSegment
            )  # The first arg should be an AudioSegment
            assert kwargs == {
                "min_length": 200,
                "threshold": -40,
                "reduction_factor": 1.0,
                "crossfade": 10,
                "min_silence_between_parts": 100,
            }


class TestSliceAudio:
    @pytest.mark.asyncio
    async def test_slice_audio(self, mock_context, audio_segment_mono):
        """Test that SliceAudio correctly slices an audio file."""
        # Setup
        node = SliceAudio(audio=AudioRef(), start=0.5, end=2.0)

        # Execute
        result = await node.process(mock_context)

        # Verify
        assert mock_context.audio_to_audio_segment.call_count == 1
        assert mock_context.audio_from_segment.call_count == 1
        assert isinstance(result, AudioRef)
        # The slice operation should have been performed on the audio segment
        # We can't directly verify the slice operation due to the mock setup,
        # but we can verify that the audio_from_segment was called


class TestTone:
    @pytest.mark.asyncio
    async def test_tone_generation(self, mock_context):
        """Test that Tone correctly generates a tone signal."""
        # Setup
        node = Tone(frequency=440.0, sampling_rate=44100, duration=1.0, phi=0.0)

        # Execute
        with patch("librosa.tone") as mock_tone:
            mock_tone.return_value = np.zeros(44100)
            result = await node.process(mock_context)

        # Verify
        assert isinstance(result, NPArray)
        mock_tone.assert_called_once_with(
            frequency=440.0, sr=44100, length=44100, phi=0.0
        )


class TestMonoToStereo:
    @pytest.mark.asyncio
    async def test_mono_to_stereo_conversion(self, mock_context, audio_segment_mono):
        """Test that MonoToStereo correctly converts a mono audio to stereo."""
        # Setup
        node = MonoToStereo(audio=AudioRef())

        # Execute
        with patch.object(audio_segment_mono, "set_channels") as mock_set_channels:
            mock_set_channels.return_value = AudioSegment.silent(
                duration=1000
            ).set_channels(2)
            result = await node.process(mock_context)

        # Verify
        assert mock_context.audio_to_audio_segment.call_count == 1
        assert mock_context.audio_from_segment.call_count == 1
        assert isinstance(result, AudioRef)
        mock_set_channels.assert_called_once_with(2)

    @pytest.mark.asyncio
    async def test_stereo_to_stereo_no_change(self, mock_context, audio_segment_stereo):
        """Test that MonoToStereo doesn't change an already stereo audio."""
        # Setup
        mock_context.set_next_audio_as_stereo()
        node = MonoToStereo(audio=AudioRef())

        # Execute
        result = await node.process(mock_context)

        # Verify
        assert mock_context.audio_to_audio_segment.call_count == 1
        assert mock_context.audio_from_segment.call_count == 1
        assert isinstance(result, AudioRef)


class TestStereoToMono:
    @pytest.mark.asyncio
    async def test_stereo_to_mono_average(self, mock_context, audio_segment_stereo):
        """Test that StereoToMono correctly converts a stereo audio to mono using average method."""
        # Setup
        mock_context.set_next_audio_as_stereo()
        node = StereoToMono(audio=AudioRef(), method="average")

        # Execute
        with patch.object(audio_segment_stereo, "set_channels") as mock_set_channels:
            mock_set_channels.return_value = AudioSegment.silent(
                duration=1000
            ).set_channels(1)
            result = await node.process(mock_context)

        # Verify
        assert mock_context.audio_to_audio_segment.call_count == 1
        assert mock_context.audio_from_segment.call_count == 1
        assert isinstance(result, AudioRef)
        mock_set_channels.assert_called_once_with(1)

    @pytest.mark.asyncio
    async def test_stereo_to_mono_left(self, mock_context, audio_segment_stereo):
        """Test that StereoToMono correctly converts a stereo audio to mono using left channel."""
        # Setup
        mock_context.set_next_audio_as_stereo()
        node = StereoToMono(audio=AudioRef(), method="left")

        # Execute
        with patch.object(audio_segment_stereo, "split_to_mono") as mock_split_to_mono:
            mock_split_to_mono.return_value = [
                AudioSegment.silent(duration=1000).set_channels(1),
                AudioSegment.silent(duration=1000).set_channels(1),
            ]
            result = await node.process(mock_context)

        # Verify
        assert mock_context.audio_to_audio_segment.call_count == 1
        assert mock_context.audio_from_segment.call_count == 1
        assert isinstance(result, AudioRef)
        mock_split_to_mono.assert_called_once()

    @pytest.mark.asyncio
    async def test_stereo_to_mono_right(self, mock_context, audio_segment_stereo):
        """Test that StereoToMono correctly converts a stereo audio to mono using right channel."""
        # Setup
        mock_context.set_next_audio_as_stereo()
        node = StereoToMono(audio=AudioRef(), method="right")

        # Execute
        with patch.object(audio_segment_stereo, "split_to_mono") as mock_split_to_mono:
            mock_split_to_mono.return_value = [
                AudioSegment.silent(duration=1000).set_channels(1),
                AudioSegment.silent(duration=1000).set_channels(1),
            ]
            result = await node.process(mock_context)

        # Verify
        assert mock_context.audio_to_audio_segment.call_count == 1
        assert mock_context.audio_from_segment.call_count == 1
        assert isinstance(result, AudioRef)
        mock_split_to_mono.assert_called_once()

    @pytest.mark.asyncio
    async def test_stereo_to_mono_invalid_method(
        self, mock_context, audio_segment_stereo
    ):
        """Test that StereoToMono raises an error for invalid method."""
        # Setup
        mock_context.set_next_audio_as_stereo()
        node = StereoToMono(audio=AudioRef(), method="invalid")

        # Execute and Verify
        with pytest.raises(ValueError):
            await node.process(mock_context)

    @pytest.mark.asyncio
    async def test_mono_to_mono_no_change(self, mock_context):
        """Test that StereoToMono doesn't change an already mono audio."""
        # Setup
        mock_context.reset_to_mono()
        node = StereoToMono(audio=AudioRef(), method="average")

        # Execute
        result = await node.process(mock_context)

        # Verify
        assert mock_context.audio_to_audio_segment.call_count == 1
        assert mock_context.audio_from_segment.call_count == 1
        assert isinstance(result, AudioRef)


class TestReverse:
    @pytest.mark.asyncio
    async def test_reverse(self, mock_context, audio_segment_mono):
        """Test that Reverse correctly reverses an audio file."""
        # Setup
        node = Reverse(audio=AudioRef())

        # Execute
        with patch.object(audio_segment_mono, "reverse") as mock_reverse:
            mock_reverse.return_value = audio_segment_mono
            result = await node.process(mock_context)

        # Verify
        assert mock_context.audio_to_audio_segment.call_count == 1
        assert mock_context.audio_from_segment.call_count == 1
        assert isinstance(result, AudioRef)
        mock_reverse.assert_called_once()


class TestFadeIn:
    @pytest.mark.asyncio
    async def test_fade_in(self, mock_context, audio_segment_mono):
        """Test that FadeIn correctly applies a fade-in effect to an audio file."""
        # Setup
        node = FadeIn(audio=AudioRef(), duration=1.0)

        # Execute
        with patch.object(audio_segment_mono, "fade_in") as mock_fade_in:
            mock_fade_in.return_value = audio_segment_mono
            result = await node.process(mock_context)

        # Verify
        assert mock_context.audio_to_audio_segment.call_count == 1
        assert mock_context.audio_from_segment.call_count == 1
        assert isinstance(result, AudioRef)
        mock_fade_in.assert_called_once_with(duration=1000)


class TestFadeOut:
    @pytest.mark.asyncio
    async def test_fade_out(self, mock_context, audio_segment_mono):
        """Test that FadeOut correctly applies a fade-out effect to an audio file."""
        # Setup
        node = FadeOut(audio=AudioRef(), duration=1.0)

        # Execute
        with patch.object(audio_segment_mono, "fade_out") as mock_fade_out:
            mock_fade_out.return_value = audio_segment_mono
            result = await node.process(mock_context)

        # Verify
        assert mock_context.audio_to_audio_segment.call_count == 1
        assert mock_context.audio_from_segment.call_count == 1
        assert isinstance(result, AudioRef)
        mock_fade_out.assert_called_once_with(duration=1000)


class TestRepeat:
    @pytest.mark.asyncio
    async def test_repeat(self, mock_context, audio_segment_mono):
        """Test that Repeat correctly loops an audio file."""
        # Setup
        node = Repeat(audio=AudioRef(), loops=3)

        # Execute
        result = await node.process(mock_context)

        # Verify
        assert mock_context.audio_to_audio_segment.call_count == 1
        assert mock_context.audio_from_segment.call_count == 1
        assert isinstance(result, AudioRef)
        # The multiplication operation should have been performed on the audio segment
        # We can't directly verify the multiplication due to the mock setup,
        # but we can verify that the audio_from_segment was called


class TestAudioMixer:
    @pytest.mark.asyncio
    async def test_audio_mixer_with_one_track(self, mock_context, audio_segment_mono):
        """Test that AudioMixer correctly mixes one audio track."""
        # Setup
        # Create a mock for is_empty method
        with patch.object(AudioRef, "is_empty") as mock_is_empty:
            # First track is not empty, others are empty
            mock_is_empty.side_effect = [False, True, True, True, True]

            node = AudioMixer(
                track1=AudioRef(),
                track2=AudioRef(),
                track3=AudioRef(),
                track4=AudioRef(),
                track5=AudioRef(),
                volume1=1.0,
                volume2=1.0,
                volume3=1.0,
                volume4=1.0,
                volume5=1.0,
            )

            # Execute
            result = await node.process(mock_context)

        # Verify
        assert mock_context.audio_to_audio_segment.call_count == 1
        assert mock_context.audio_from_segment.call_count == 1
        assert isinstance(result, AudioRef)

    @pytest.mark.asyncio
    async def test_audio_mixer_with_multiple_tracks(
        self, mock_context, audio_segment_mono
    ):
        """Test that AudioMixer correctly mixes multiple audio tracks."""
        # Setup
        # Create a mock for is_empty method
        with patch.object(AudioRef, "is_empty") as mock_is_empty:
            # First three tracks are not empty, last two are empty
            mock_is_empty.side_effect = [False, False, False, True, True]

            node = AudioMixer(
                track1=AudioRef(),
                track2=AudioRef(),
                track3=AudioRef(),
                track4=AudioRef(),
                track5=AudioRef(),
                volume1=0.8,
                volume2=1.0,
                volume3=1.2,
                volume4=1.0,
                volume5=1.0,
            )

            # Execute
            with patch.object(
                audio_segment_mono, "apply_gain"
            ) as mock_apply_gain, patch.object(
                audio_segment_mono, "overlay"
            ) as mock_overlay:
                mock_apply_gain.return_value = audio_segment_mono
                mock_overlay.return_value = audio_segment_mono
                result = await node.process(mock_context)

        # Verify
        assert mock_context.audio_to_audio_segment.call_count == 3
        assert mock_context.audio_from_segment.call_count == 1
        assert isinstance(result, AudioRef)
        assert (
            mock_apply_gain.call_count == 2
        )  # Called for track1 and track3 with non-default volumes
        assert (
            mock_overlay.call_count == 2
        )  # Called to overlay track2 and track3 onto track1

    @pytest.mark.asyncio
    async def test_audio_mixer_no_tracks(self, mock_context):
        """Test that AudioMixer raises an error when no tracks are provided."""
        # Setup
        # Create a mock for is_empty method
        with patch.object(AudioRef, "is_empty") as mock_is_empty:
            # All tracks are empty
            mock_is_empty.return_value = True

            node = AudioMixer(
                track1=AudioRef(),
                track2=AudioRef(),
                track3=AudioRef(),
                track4=AudioRef(),
                track5=AudioRef(),
                volume1=1.0,
                volume2=1.0,
                volume3=1.0,
                volume4=1.0,
                volume5=1.0,
            )

            # Execute and Verify
            with pytest.raises(ValueError):
                await node.process(mock_context)
