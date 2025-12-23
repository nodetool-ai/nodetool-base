"""Kie.ai video generation nodes.

This module provides nodes for generating videos using Kie.ai's various APIs:
- Veo 3.1 (Google DeepMind text-to-video)
- Wan 2.6 (Alibaba multi-shot HD video)
- Sora 2 (OpenAI text-to-video)
- Seedance 1.0 (ByteDance video generation)
- Hailuo (MiniMax video generation)
- Kling AI Avatar (Kuaishou talking-head generator)
- Topaz Video Upscaler (AI video enhancement)
"""

from enum import Enum
from typing import Any, ClassVar

from pydantic import Field

from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import ImageRef, VideoRef, AudioRef
from nodetool.workflows.processing_context import ProcessingContext

from .image import KieBaseNode

log = get_logger(__name__)


class KieVideoBaseNode(KieBaseNode):
    """Base class for Kie.ai video generation nodes.

    kie, ai, video generation, api

    Extends KieBaseNode with video-specific result handling.
    """

    @classmethod
    def is_visible(cls) -> bool:
        return cls is not KieVideoBaseNode

    async def _execute_video_task(self, context: ProcessingContext) -> bytes:
        """Execute the full task workflow for video: submit, poll, download."""
        return await self._execute_task(context)


class Veo31Generate(KieVideoBaseNode):
    """Generate videos using Google DeepMind's Veo 3.1 model via Kie.ai.

    kie, veo, google, deepmind, video generation, ai, text-to-video, 1080p

    Veo 3.1 creates 1080p videos with synchronized native audio, extended clip
    durations beyond 8 seconds, and fine-grained frame control.

    Use cases:
    - Generate videos from text descriptions
    - Create 1080p content with native audio
    - Generate cinematic short videos
    - Image-to-video generation
    """

    _expose_as_tool: ClassVar[bool] = True

    prompt: str = Field(
        default="",
        description="The text prompt describing the video to generate.",
    )

    image: ImageRef = Field(
        default=ImageRef(),
        description="Optional reference image for image-to-video generation.",
    )

    class Mode(str, Enum):
        FAST = "fast"
        QUALITY = "quality"

    mode: Mode = Field(
        default=Mode.QUALITY,
        description="Generation mode: 'fast' for speed, 'quality' for higher fidelity.",
    )

    duration: int = Field(
        default=8,
        description="Video duration in seconds.",
        ge=4,
        le=15,
    )

    def _get_base_endpoint(self) -> str:
        return "/v1/market/veo-3-1"

    def _get_submit_payload(self) -> dict[str, Any]:
        if not self.prompt:
            raise ValueError("Prompt cannot be empty")
        payload: dict[str, Any] = {
            "prompt": self.prompt,
            "mode": self.mode.value,
            "duration": self.duration,
        }
        if self.image.is_set():
            payload["image"] = self.image.to_dict()
        return payload

    async def process(self, context: ProcessingContext) -> VideoRef:
        video_bytes = await self._execute_video_task(context)
        return await context.video_from_bytes(video_bytes)


class Wan26Generate(KieVideoBaseNode):
    """Generate videos using Alibaba's Wan 2.6 model via Kie.ai.

    kie, wan, alibaba, video generation, ai, text-to-video, multi-shot, 1080p

    Wan 2.6 supports text-to-video, image-to-video, and video-to-video modes
    with up to 1080p resolution. Can produce coherent multi-scene videos.

    Use cases:
    - Generate 15-second HD videos
    - Create multi-shot videos with scene transitions
    - Image-to-video animation
    - Video-to-video transformations
    """

    _expose_as_tool: ClassVar[bool] = True

    prompt: str = Field(
        default="",
        description="The text prompt describing the video to generate.",
    )

    image: ImageRef = Field(
        default=ImageRef(),
        description="Optional reference image for image-to-video generation.",
    )

    multi_shots: bool = Field(
        default=False,
        description="Enable multi-shot mode for coherent scene transitions.",
    )

    class Resolution(str, Enum):
        HD_720P = "720p"
        FULL_HD_1080P = "1080p"

    resolution: Resolution = Field(
        default=Resolution.FULL_HD_1080P,
        description="Video resolution.",
    )

    def _get_base_endpoint(self) -> str:
        return "/v1/market/wan-2-6"

    def _get_submit_payload(self) -> dict[str, Any]:
        if not self.prompt:
            raise ValueError("Prompt cannot be empty")
        payload: dict[str, Any] = {
            "prompt": self.prompt,
            "resolution": self.resolution.value,
            "multi_shots": self.multi_shots,
        }
        if self.image.is_set():
            payload["image"] = self.image.to_dict()
        return payload

    async def process(self, context: ProcessingContext) -> VideoRef:
        video_bytes = await self._execute_video_task(context)
        return await context.video_from_bytes(video_bytes)


class Sora2Generate(KieVideoBaseNode):
    """Generate videos using OpenAI's Sora 2 model via Kie.ai.

    kie, sora, openai, video generation, ai, text-to-video, realistic

    Sora 2 generates short videos (up to 10 seconds) from text or images,
    emphasizing realistic motion, physics consistency, and native audio.

    Use cases:
    - Generate realistic videos from text
    - Create videos with native audio (dialogue/ambient sound)
    - Image-to-video generation
    - Cinematic content creation
    """

    _expose_as_tool: ClassVar[bool] = True

    prompt: str = Field(
        default="",
        description="The text prompt describing the video to generate.",
    )

    image: ImageRef = Field(
        default=ImageRef(),
        description="Optional reference image for image-to-video generation.",
    )

    class Mode(str, Enum):
        STANDARD = "standard"
        PRO = "pro"

    mode: Mode = Field(
        default=Mode.STANDARD,
        description="Generation mode: 'standard' or 'pro' for higher quality.",
    )

    def _get_base_endpoint(self) -> str:
        return "/v1/market/sora-2"

    def _get_submit_payload(self) -> dict[str, Any]:
        if not self.prompt:
            raise ValueError("Prompt cannot be empty")
        payload: dict[str, Any] = {
            "prompt": self.prompt,
            "mode": self.mode.value,
        }
        if self.image.is_set():
            payload["image"] = self.image.to_dict()
        return payload

    async def process(self, context: ProcessingContext) -> VideoRef:
        video_bytes = await self._execute_video_task(context)
        return await context.video_from_bytes(video_bytes)


class Sora2ProGenerate(KieVideoBaseNode):
    """Generate videos using OpenAI's Sora 2 Pro model via Kie.ai.

    kie, sora, openai, video generation, ai, text-to-video, pro, 1080p

    Sora 2 Pro supports higher resolution (HD) and longer clips (15s),
    with improved realism in motion and physics.

    Use cases:
    - Generate professional-grade HD videos
    - Create longer 15-second clips
    - High-fidelity motion and physics
    - Premium content creation
    """

    _expose_as_tool: ClassVar[bool] = True

    prompt: str = Field(
        default="",
        description="The text prompt describing the video to generate.",
    )

    image: ImageRef = Field(
        default=ImageRef(),
        description="Optional reference image for image-to-video generation.",
    )

    class Resolution(str, Enum):
        STANDARD_720P = "720p"
        HD_1080P = "1080p"

    resolution: Resolution = Field(
        default=Resolution.HD_1080P,
        description="Video resolution.",
    )

    def _get_base_endpoint(self) -> str:
        return "/v1/market/sora-2-pro"

    def _get_submit_payload(self) -> dict[str, Any]:
        if not self.prompt:
            raise ValueError("Prompt cannot be empty")
        payload: dict[str, Any] = {
            "prompt": self.prompt,
            "resolution": self.resolution.value,
        }
        if self.image.is_set():
            payload["image"] = self.image.to_dict()
        return payload

    async def process(self, context: ProcessingContext) -> VideoRef:
        video_bytes = await self._execute_video_task(context)
        return await context.video_from_bytes(video_bytes)


class Seedance10Generate(KieVideoBaseNode):
    """Generate videos using ByteDance's Seedance 1.0 model via Kie.ai.

    kie, seedance, bytedance, video generation, ai, text-to-video, multi-shot

    Seedance 1.0 supports text and/or image to video generation with
    Lite (faster) and Pro (higher quality) modes.

    Use cases:
    - Generate videos from text/image
    - Create multi-shot videos with scene transitions
    - Fast video prototyping with Lite mode
    - High-quality output with Pro mode
    """

    _expose_as_tool: ClassVar[bool] = True

    prompt: str = Field(
        default="",
        description="The text prompt describing the video to generate.",
    )

    image: ImageRef = Field(
        default=ImageRef(),
        description="Optional reference image for image-to-video generation.",
    )

    class Mode(str, Enum):
        LITE = "lite"
        PRO = "pro"

    mode: Mode = Field(
        default=Mode.PRO,
        description="Generation mode: 'lite' for speed, 'pro' for quality.",
    )

    def _get_base_endpoint(self) -> str:
        return "/v1/market/seedance-1-0"

    def _get_submit_payload(self) -> dict[str, Any]:
        if not self.prompt:
            raise ValueError("Prompt cannot be empty")
        payload: dict[str, Any] = {
            "prompt": self.prompt,
            "mode": self.mode.value,
        }
        if self.image.is_set():
            payload["image"] = self.image.to_dict()
        return payload

    async def process(self, context: ProcessingContext) -> VideoRef:
        video_bytes = await self._execute_video_task(context)
        return await context.video_from_bytes(video_bytes)


class Hailuo23Generate(KieVideoBaseNode):
    """Generate videos using MiniMax's Hailuo 2.3 model via Kie.ai.

    kie, hailuo, minimax, video generation, ai, text-to-video, cinematic

    Hailuo 2.3 specializes in realistic character motion, expressive facial
    micro-expressions, and cinematic quality for text-to-video and image-to-video.

    Use cases:
    - Generate cinematic videos with realistic characters
    - Create content with expressive facial animations
    - High-fidelity text-to-video generation
    - Image-to-video with complex movements
    """

    _expose_as_tool: ClassVar[bool] = True

    prompt: str = Field(
        default="",
        description="The text prompt describing the video to generate.",
    )

    image: ImageRef = Field(
        default=ImageRef(),
        description="Optional reference image for image-to-video generation.",
    )

    class Resolution(str, Enum):
        HD_720P = "720p"
        FULL_HD_1080P = "1080p"

    resolution: Resolution = Field(
        default=Resolution.FULL_HD_1080P,
        description="Video resolution.",
    )

    class Mode(str, Enum):
        STANDARD = "standard"
        FAST = "fast"

    mode: Mode = Field(
        default=Mode.STANDARD,
        description="Generation mode: 'standard' for quality, 'fast' for speed.",
    )

    def _get_base_endpoint(self) -> str:
        return "/v1/market/hailuo-2-3"

    def _get_submit_payload(self) -> dict[str, Any]:
        if not self.prompt:
            raise ValueError("Prompt cannot be empty")
        payload: dict[str, Any] = {
            "prompt": self.prompt,
            "resolution": self.resolution.value,
            "mode": self.mode.value,
        }
        if self.image.is_set():
            payload["image"] = self.image.to_dict()
        return payload

    async def process(self, context: ProcessingContext) -> VideoRef:
        video_bytes = await self._execute_video_task(context)
        return await context.video_from_bytes(video_bytes)


class KlingAIAvatar(KieVideoBaseNode):
    """Generate talking avatar videos using Kuaishou's Kling AI via Kie.ai.

    kie, kling, kuaishou, avatar, video generation, ai, talking-head, lip-sync

    Transforms a photo plus audio track into a lip-synced talking avatar video
    with natural-looking speech animation and consistent identity.

    Use cases:
    - Create virtual influencer content
    - Generate educational presenters
    - Lip-synced avatar videos
    - Virtual spokesperson creation
    """

    _expose_as_tool: ClassVar[bool] = True

    image: ImageRef = Field(
        default=ImageRef(),
        description="The face/character image to animate.",
    )

    audio: AudioRef = Field(
        default=AudioRef(),
        description="The audio track for lip-syncing.",
    )

    prompt: str = Field(
        default="",
        description="Optional text to guide emotions and expressions.",
    )

    class Mode(str, Enum):
        STANDARD = "standard"
        PRO = "pro"

    mode: Mode = Field(
        default=Mode.STANDARD,
        description="Generation mode: 'standard' or 'pro' for higher quality.",
    )

    def _get_base_endpoint(self) -> str:
        return "/v1/market/kling-ai-avatar"

    def _get_submit_payload(self) -> dict[str, Any]:
        if not self.image.is_set():
            raise ValueError("Image is required")
        if not self.audio.is_set():
            raise ValueError("Audio is required")
        payload: dict[str, Any] = {
            "image": self.image.to_dict(),
            "audio": self.audio.to_dict(),
            "mode": self.mode.value,
        }
        if self.prompt:
            payload["prompt"] = self.prompt
        return payload

    async def process(self, context: ProcessingContext) -> VideoRef:
        video_bytes = await self._execute_video_task(context)
        return await context.video_from_bytes(video_bytes)


class TopazVideoUpscale(KieVideoBaseNode):
    """Upscale and enhance videos using Topaz Labs AI via Kie.ai.

    kie, topaz, upscale, enhance, video, ai, 4k

    Uses Topaz Labs' video enhancement AI to upscale videos to 1080p or 4K,
    sharpen frames, reduce noise, and interpolate frames for smoother motion.

    Use cases:
    - Upscale old or low-quality footage to HD/4K
    - Enhance YouTube content or game recordings
    - Restore and improve home videos
    - Prepare videos for high-resolution displays
    """

    _expose_as_tool: ClassVar[bool] = True

    video: VideoRef = Field(
        default=VideoRef(),
        description="The video to upscale.",
    )

    class Resolution(str, Enum):
        HD_1080P = "1080p"
        UHD_4K = "4k"

    resolution: Resolution = Field(
        default=Resolution.HD_1080P,
        description="Target resolution for upscaling.",
    )

    denoise: bool = Field(
        default=True,
        description="Apply denoising to reduce artifacts.",
    )

    def _get_base_endpoint(self) -> str:
        return "/v1/market/topaz-video-upscaler"

    def _get_submit_payload(self) -> dict[str, Any]:
        if not self.video.is_set():
            raise ValueError("Video is required")
        return {
            "video": self.video.to_dict(),
            "resolution": self.resolution.value,
            "denoise": self.denoise,
        }

    async def process(self, context: ProcessingContext) -> VideoRef:
        video_bytes = await self._execute_video_task(context)
        return await context.video_from_bytes(video_bytes)
