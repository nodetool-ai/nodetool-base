from __future__ import annotations

import asyncio
import base64
import logging
import os
from enum import Enum
from typing import Any

import numpy as np
from pydantic import Field
from huggingface_hub import try_to_load_from_cache, _CACHED_NO_EXIST

from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import AudioRef, HuggingFaceModel
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.io import NodeInputs, NodeOutputs

# Optional: used for streaming string chunks
from nodetool.chat.providers import Chunk

# New library: pywhispercpp
from pywhispercpp import constants as pwconstants
from pywhispercpp.model import Model as PWModel, Segment

log = get_logger(__name__)

REPO_ID = "ggerganov/whisper.cpp"


def _resolve_model_path(model_name: str) -> str:
    """Map model short name (e.g., 'tiny.en') to ggml file in HF cache.

    Uses Hugging Face cache to avoid network downloads.
    """
    # pywhispercpp uses whisper.cpp ggml files named ggml-<name>.bin
    filename = f"ggml-{model_name}.bin"
    filepath = try_to_load_from_cache(REPO_ID, filename)
    if isinstance(filepath, str):
        return filepath
    if filepath is _CACHED_NO_EXIST:
        raise FileNotFoundError(
            f"Model file not found in HF cache: {REPO_ID}:{filename}"
        )
    raise FileNotFoundError(f"Model file not found in HF cache: {REPO_ID}:{filename}")


def _audio_segment_to_float32_mono_16k(
    context: ProcessingContext, audio: AudioRef
) -> np.ndarray:
    """
    Convert an AudioRef to float32 mono PCM at 16 kHz suitable for whispercpp.
    """
    # This function is sync by design; the heavy lifting happens in the awaited caller
    raise NotImplementedError


class WhisperCpp(BaseNode):
    """
    Transcribe an audio asset using whispercpp (whisper.cpp bindings) and stream strings.
    whisper, whispercpp, asr, speech-to-text, streaming, huggingface-cache

    - Model file is loaded from the local Hugging Face cache (repo + filename)
    - Emits streaming text deltas on `chunk` and final transcript on `text`
    """

    class Model(str, Enum):
        # tiny
        TINY = "tiny"
        TINY_Q5_1 = "tiny-q5_1"
        TINY_Q8_0 = "tiny-q8_0"
        TINY_EN = "tiny.en"
        TINY_EN_Q5_1 = "tiny.en-q5_1"
        TINY_EN_Q8_0 = "tiny.en-q8_0"

        # base
        BASE = "base"
        BASE_Q5_1 = "base-q5_1"
        BASE_Q8_0 = "base-q8_0"
        BASE_EN = "base.en"
        BASE_EN_Q5_1 = "base.en-q5_1"
        BASE_EN_Q8_0 = "base.en-q8_0"

        # small
        SMALL = "small"
        SMALL_Q5_1 = "small-q5_1"
        SMALL_Q8_0 = "small-q8_0"
        SMALL_EN = "small.en"
        SMALL_EN_Q5_1 = "small.en-q5_1"
        SMALL_EN_Q8_0 = "small.en-q8_0"

        # medium
        MEDIUM = "medium"
        MEDIUM_Q5_0 = "medium-q5_0"
        MEDIUM_Q8_0 = "medium-q8_0"
        MEDIUM_EN = "medium.en"
        MEDIUM_EN_Q5_0 = "medium.en-q5_0"
        MEDIUM_EN_Q8_0 = "medium.en-q8_0"

        # large variants
        LARGE_V1 = "large-v1"
        LARGE_V2 = "large-v2"
        LARGE_V2_Q5_0 = "large-v2-q5_0"
        LARGE_V2_Q8_0 = "large-v2-q8_0"
        LARGE_V3 = "large-v3"
        LARGE_V3_Q5_0 = "large-v3-q5_0"
        LARGE_V3_TURBO = "large-v3-turbo"
        LARGE_V3_TURBO_Q5_0 = "large-v3-turbo-q5_0"
        LARGE_V3_TURBO_Q8_0 = "large-v3-turbo-q8_0"

    model: Model = Field(
        default=Model.TINY_EN,
        description="Model to use from ggerganov/whisper.cpp",
    )

    # Decoding parameters (subset, expanded as needed)
    n_threads: int = Field(default=4, description="Decoder CPU threads")
    length_ms: int = Field(
        default=5000, description="Chunk length in milliseconds for pseudo-streaming"
    )

    audio: AudioRef = Field(default=AudioRef(), description="Audio to transcribe")
    chunk: Chunk = Field(default=Chunk(), description="Chunk to transcribe")

    @classmethod
    def is_cacheable(cls) -> bool:
        return False

    @classmethod
    def is_streaming_output(cls) -> bool:
        return True

    @classmethod
    def is_streaming_input(cls) -> bool:
        return True

    @classmethod
    def return_type(cls):
        return {
            "text": str,
            "chunk": Chunk,
            "t0": float,
            "t1": float,
            "probability": float,
        }

    async def _load_whisper(self) -> PWModel:
        # Resolve local path from HF cache and instantiate pywhispercpp model
        model_path = _resolve_model_path(self.model.value)
        return PWModel(
            model_path,
            print_realtime=False,
            print_progress=False,
            print_timestamps=False,
            single_segment=True,
            n_threads=self.n_threads,
        )

    async def _audio_to_float32(
        self, context: ProcessingContext, audio: AudioRef
    ) -> tuple[np.ndarray, int]:
        """Convert AudioRef to float32 mono array at 16kHz and return (array, sample_rate)."""
        if not audio or audio.is_empty():
            raise ValueError("Audio input is empty; please connect an audio source")

        audio_segment = await context.audio_to_audio_segment(audio)
        # Ensure mono, 16kHz, 16-bit
        audio_segment = (
            audio_segment.set_channels(1).set_frame_rate(16000).set_sample_width(2)
        )
        # Convert to float32 in range [-1.0, 1.0]
        pcm = np.frombuffer(audio_segment.raw_data, dtype=np.int16)
        arr = (pcm.astype(np.float32) / 32768.0).flatten()
        return arr, 16000

    async def run(
        self, context: ProcessingContext, inputs: NodeInputs, outputs: NodeOutputs
    ) -> None:
        model = await self._load_whisper()

        # Queues for streaming audio samples and decoded text
        input_q: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=32)
        done_flag = {"done": False}

        async def producer() -> None:
            # Accept either streaming audio chunks or a single AudioRef
            async for handle, item in inputs.any():
                if handle == "chunk" and isinstance(item, Chunk):
                    if item.content_type == "audio" and item.content:
                        raw: bytes | None = None
                        if isinstance(item.content, str):
                            try:
                                raw = base64.b64decode(item.content)
                            except Exception:
                                raw = None
                        elif isinstance(item.content, (bytes, bytearray)):
                            raw = bytes(item.content)
                        if raw:
                            # Assume PCM16 little-endian; convert to float32 mono 16k
                            pcm16 = np.frombuffer(raw, dtype=np.int16)
                            arr = (pcm16.astype(np.float32) / 32768.0).flatten()
                            await input_q.put(arr)
                    if getattr(item, "done", False):
                        # Segment boundary hint; push empty marker
                        await input_q.put(np.array([], dtype=np.float32))
                elif handle == "audio" and isinstance(item, AudioRef):
                    arr, _sr = await self._audio_to_float32(context, item)
                    # Push full audio in fixed-size chunks into the queue
                    chunk_len = max(
                        1,
                        int(self.length_ms * pwconstants.WHISPER_SAMPLE_RATE / 1000),
                    )
                    total = arr.shape[0]
                    pos = 0
                    while pos < total:
                        await input_q.put(arr[pos : min(pos + chunk_len, total)])
                        pos += chunk_len
                    # Signal boundary to force decode of any remaining buffer
                    await input_q.put(np.array([], dtype=np.float32))
            done_flag["done"] = True

        async def consumer() -> None:
            buffer = np.array([], dtype=np.float32)
            min_samples = max(
                1, int(self.length_ms * pwconstants.WHISPER_SAMPLE_RATE / 1000)
            )
            aggregated: list[str] = []
            last_text = ""

            loop = asyncio.get_running_loop()

            while not (done_flag["done"] and input_q.empty()):
                try:
                    chunk = await asyncio.wait_for(input_q.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    chunk = None  # type: ignore

                if chunk is not None:
                    if chunk.size == 0:
                        # explicit boundary: force decode if we have content
                        if buffer.size > 0:
                            arr = buffer.astype(np.float32, copy=True)
                            buffer = np.array([], dtype=np.float32)
                        else:
                            arr = None
                    else:
                        buffer = (
                            np.concatenate([buffer, chunk]) if buffer.size else chunk
                        )
                        arr = buffer if buffer.size >= min_samples else None
                        if arr is not None:
                            # keep small tail to overlap a bit (simple VAD-ish)
                            tail_keep = min_samples // 4
                            buffer = (
                                buffer[-tail_keep:]
                                if buffer.size > tail_keep
                                else np.array([], dtype=np.float32)
                            )
                    if arr is None:
                        await asyncio.sleep(0)
                        continue

                    # Blocking transcription in executor
                    def _do_transcribe(a: np.ndarray) -> list[Segment]:
                        return model.transcribe(a, n_processors=None)

                    try:
                        result = await loop.run_in_executor(None, _do_transcribe, arr)
                    except Exception as e:
                        raise RuntimeError(
                            f"Whisper (pywhispercpp) transcription failed: {e}"
                        ) from e

                    # Normalize to string
                    for segment in result:
                        # Try emit only the delta when prefix grows
                        aggregated.append(segment.text)
                        await outputs.emit(
                            "chunk", Chunk(content=segment.text, done=False)
                        )
                        await outputs.emit("t0", segment.t0)
                        await outputs.emit("t1", segment.t1)
                        await outputs.emit("probability", segment.probability)

                await asyncio.sleep(0)

            # Flush
            await outputs.emit("chunk", Chunk(content="", done=True))
            final_text = "".join(aggregated).strip()
            if final_text:
                await outputs.emit("text", final_text)
            outputs.complete("text")

        await asyncio.gather(producer(), consumer())

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        """Recommend ggml Whisper models from ggerganov/whisper.cpp for local cache use.

        These correspond to files listed on the HF repo page and are suitable
        for whisper.cpp bindings.
        """
        paths = [f"ggml-{m.value}.bin" for m in cls.Model]
        return [HuggingFaceModel(repo_id=REPO_ID, path=p) for p in paths]
