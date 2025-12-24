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

import asyncio
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

    _poll_interval: float = 8.0
    _max_poll_attempts: int = 180

    @classmethod
    def is_visible(cls) -> bool:
        return cls is not KieVideoBaseNode

    async def _execute_video_task(self, context: ProcessingContext) -> bytes:
        """Execute the full task workflow for video: submit, poll, download."""
        return await self._execute_task(context)


class Sora2ProTextToVideo(KieVideoBaseNode):
    """Generate videos from text using OpenAI's Sora 2 Pro model via Kie.ai.

    kie, sora, openai, video generation, ai, text-to-video, pro, 1080p

    Sora 2 Pro Text-to-Video generates professional-grade HD videos from text prompts
    with improved realism in motion and physics. Supports higher resolution (HD) and
    longer clips (15s).

    Use cases:
    - Generate professional-grade HD videos from text
    - Create longer 15-second clips
    - High-fidelity motion and physics
    - Premium text-to-video content creation
    """

    _expose_as_tool: ClassVar[bool] = True

    prompt: str = Field(
        default="",
        description="The text prompt describing the video to generate.",
    )

    class AspectRatio(str, Enum):
        LANDSCAPE = "landscape"
        PORTRAIT = "portrait"
        SQUARE = "square"

    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.LANDSCAPE,
        description="The aspect ratio of the generated video.",
    )

    n_frames: int = Field(
        default=10,
        description="Number of frames for the video.",
        ge=1,
        le=60,
    )

    remove_watermark: bool = Field(
        default=True,
        description="Whether to remove the watermark from the generated video.",
    )

    def _get_model(self) -> str:
        return "sora-2-pro-text-to-video"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if not self.prompt:
            raise ValueError("Prompt cannot be empty")
        return {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "n_frames": str(self.n_frames),
            "remove_watermark": self.remove_watermark,
        }

    async def process(self, context: ProcessingContext) -> VideoRef:
        video_bytes = await self._execute_video_task(context)
        return await context.video_from_bytes(video_bytes)


class Sora2ProImageToVideo(KieVideoBaseNode):
    """Generate videos from images using OpenAI's Sora 2 Pro model via Kie.ai.

    kie, sora, openai, video generation, ai, image-to-video, pro, 1080p

    Sora 2 Pro Image-to-Video transforms reference images into professional-grade
    HD videos with improved realism in motion and physics. Supports higher resolution
    (HD) and longer clips (15s).

    Use cases:
    - Generate professional-grade HD videos from images
    - Create longer 15-second video clips
    - High-fidelity motion and physics from images
    - Premium image-to-video content creation
    """

    _expose_as_tool: ClassVar[bool] = True

    image: ImageRef = Field(
        default=ImageRef(),
        description="The reference image to animate into a video.",
    )

    prompt: str = Field(
        default="",
        description="Optional text to guide the video generation.",
    )

    class AspectRatio(str, Enum):
        LANDSCAPE = "landscape"
        PORTRAIT = "portrait"
        SQUARE = "square"

    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.LANDSCAPE,
        description="The aspect ratio of the generated video.",
    )

    n_frames: int = Field(
        default=10,
        description="Number of frames for the video.",
        ge=1,
        le=60,
    )

    remove_watermark: bool = Field(
        default=True,
        description="Whether to remove the watermark from the generated video.",
    )

    def _get_model(self) -> str:
        return "sora-2-pro-image-to-video"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if not self.image.is_set():
            raise ValueError("Image is required")
        if context is None:
            raise ValueError("Context is required for image upload")
        image_url = await self._upload_image(context, self.image)
        payload: dict[str, Any] = {
            "image_urls": [image_url],
            "aspect_ratio": self.aspect_ratio.value,
            "n_frames": str(self.n_frames),
            "remove_watermark": self.remove_watermark,
        }
        if self.prompt:
            payload["prompt"] = self.prompt
        return payload

    async def _get_submit_payload(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        return {
            "model": self._get_model(),
            "input": await self._get_input_params(context),
        }

    async def process(self, context: ProcessingContext) -> VideoRef:
        video_bytes = await self._execute_video_task(context)
        return await context.video_from_bytes(video_bytes)


class Sora2ProStoryboard(KieVideoBaseNode):
    """Generate storyboard videos using OpenAI's Sora 2 Pro model via Kie.ai.

    kie, sora, openai, video generation, ai, storyboard, pro, 1080p

    Sora 2 Pro Storyboard creates professional-grade storyboard videos with
    scene transitions, cinematic framing, and professional editing techniques.

    Use cases:
    - Generate cinematic storyboards from text
    - Create professional video previews
    - Plan video shoots with visual storyboards
    - Develop video concepts with scene sequences
    """

    _expose_as_tool: ClassVar[bool] = True

    prompt: str = Field(
        default="",
        description="The text prompt describing the storyboard sequence.",
    )

    class AspectRatio(str, Enum):
        LANDSCAPE = "landscape"
        PORTRAIT = "portrait"
        SQUARE = "square"

    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.LANDSCAPE,
        description="The aspect ratio of the generated video.",
    )

    n_frames: int = Field(
        default=10,
        description="Number of frames for the video.",
        ge=1,
        le=60,
    )

    remove_watermark: bool = Field(
        default=True,
        description="Whether to remove the watermark from the generated video.",
    )

    def _get_model(self) -> str:
        return "sora-2-pro-storyboard"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if not self.prompt:
            raise ValueError("Prompt cannot be empty")
        return {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "n_frames": str(self.n_frames),
            "remove_watermark": self.remove_watermark,
        }

    async def process(self, context: ProcessingContext) -> VideoRef:
        video_bytes = await self._execute_video_task(context)
        return await context.video_from_bytes(video_bytes)


class Sora2TextToVideo(KieVideoBaseNode):
    """Generate videos from text using OpenAI's Sora 2 model via Kie.ai.

    kie, sora, openai, video generation, ai, text-to-video, realistic

    Sora 2 Text-to-Video generates short videos (up to 10 seconds) from text prompts,
    emphasizing realistic motion, physics consistency, and native audio. Supports
    standard and pro modes for different quality/speed tradeoffs.

    Use cases:
    - Generate realistic videos from text
    - Create videos with native audio (dialogue/ambient sound)
    - Quick text-to-video prototyping
    - Cinematic content creation
    """

    _expose_as_tool: ClassVar[bool] = True

    prompt: str = Field(
        default="",
        description="The text prompt describing the video to generate.",
    )

    class AspectRatio(str, Enum):
        LANDSCAPE = "landscape"
        PORTRAIT = "portrait"
        SQUARE = "square"

    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.LANDSCAPE,
        description="The aspect ratio of the generated video.",
    )

    n_frames: int = Field(
        default=10,
        description="Number of frames for the video.",
        ge=1,
        le=60,
    )

    remove_watermark: bool = Field(
        default=True,
        description="Whether to remove the watermark from the generated video.",
    )

    class Mode(str, Enum):
        STANDARD = "standard"
        PRO = "pro"

    mode: Mode = Field(
        default=Mode.STANDARD,
        description="Generation mode: 'standard' or 'pro' for higher quality.",
    )

    def _get_model(self) -> str:
        return "sora-2-image-to-video"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if context is None:
            raise ValueError("Context is required for upload handling")
        payload: dict[str, Any] = {
            "aspect_ratio": self.aspect_ratio.value,
            "n_frames": str(self.n_frames),
            "remove_watermark": self.remove_watermark,
            "mode": self.mode.value,
        }
        if self.prompt:
            payload["prompt"] = self.prompt
        return payload

    async def process(self, context: ProcessingContext) -> VideoRef:
        video_bytes = await self._execute_video_task(context)
        return await context.video_from_bytes(video_bytes)


class SeedanceV1LiteTextToVideo(KieVideoBaseNode):
    """Generate videos from text using ByteDance's Seedance V1 Lite model via Kie.ai.

    kie, seedance, bytedance, video generation, ai, text-to-video, lite, fast

    Seedance V1 Lite offers fast text-to-video generation with efficient processing,
    supporting multi-shot videos and cinematic scene transitions.

    Use cases:
    - Fast video prototyping from text
    - Create multi-shot videos with scene transitions
    - Quick concept visualization
    - Rapid iteration on video ideas
    """

    _expose_as_tool: ClassVar[bool] = True

    prompt: str = Field(
        default="",
        description="The text prompt describing the video to generate. Supports shot descriptions with [Cut to] notation.",
    )

    class AspectRatio(str, Enum):
        LANDSCAPE = "16:9"
        PORTRAIT = "9:16"
        SQUARE = "1:1"

    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.LANDSCAPE,
        description="The aspect ratio of the generated video.",
    )

    class Resolution(str, Enum):
        HD_720P = "720p"
        FULL_HD_1080P = "1080p"

    resolution: Resolution = Field(
        default=Resolution.HD_720P,
        description="Video resolution.",
    )

    duration: int = Field(
        default=5,
        description="Video duration in seconds.",
        ge=1,
        le=10,
    )

    camera_fixed: bool = Field(
        default=False,
        description="Whether to keep the camera fixed or allow camera movement.",
    )

    seed: int = Field(
        default=-1,
        description="Random seed for reproducible results. Use -1 for random seed.",
    )

    enable_safety_checker: bool = Field(
        default=True,
        description="Enable safety checker to filter inappropriate content.",
    )

    def _get_model(self) -> str:
        return "bytedance/v1-lite-text-to-video"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if not self.prompt:
            raise ValueError("Prompt cannot be empty")
        return {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "resolution": self.resolution.value,
            "duration": str(self.duration),
            "camera_fixed": self.camera_fixed,
            "seed": self.seed,
            "enable_safety_checker": self.enable_safety_checker,
        }

    async def process(self, context: ProcessingContext) -> VideoRef:
        video_bytes = await self._execute_video_task(context)
        return await context.video_from_bytes(video_bytes)


class SeedanceV1ProTextToVideo(KieVideoBaseNode):
    """Generate videos from text using ByteDance's Seedance V1 Pro model via Kie.ai.

    kie, seedance, bytedance, video generation, ai, text-to-video, pro, high-quality

    Seedance V1 Pro offers high-quality text-to-video generation with improved
    fidelity, supporting multi-shot videos and cinematic scene transitions.

    Use cases:
    - Generate high-quality videos from text
    - Create professional multi-shot videos
    - Cinematic content creation
    - High-fidelity video production
    """

    _expose_as_tool: ClassVar[bool] = True

    prompt: str = Field(
        default="",
        description="The text prompt describing the video to generate. Supports shot descriptions with [Cut to] notation.",
    )

    class AspectRatio(str, Enum):
        LANDSCAPE = "16:9"
        PORTRAIT = "9:16"
        SQUARE = "1:1"

    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.LANDSCAPE,
        description="The aspect ratio of the generated video.",
    )

    class Resolution(str, Enum):
        HD_720P = "720p"
        FULL_HD_1080P = "1080p"

    resolution: Resolution = Field(
        default=Resolution.HD_720P,
        description="Video resolution.",
    )

    duration: int = Field(
        default=5,
        description="Video duration in seconds.",
        ge=1,
        le=10,
    )

    camera_fixed: bool = Field(
        default=False,
        description="Whether to keep the camera fixed or allow camera movement.",
    )

    seed: int = Field(
        default=-1,
        description="Random seed for reproducible results. Use -1 for random seed.",
    )

    enable_safety_checker: bool = Field(
        default=True,
        description="Enable safety checker to filter inappropriate content.",
    )

    def _get_model(self) -> str:
        return "bytedance/v1-pro-text-to-video"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if not self.prompt:
            raise ValueError("Prompt cannot be empty")
        return {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "resolution": self.resolution.value,
            "duration": str(self.duration),
            "camera_fixed": self.camera_fixed,
            "seed": self.seed,
            "enable_safety_checker": self.enable_safety_checker,
        }

    async def process(self, context: ProcessingContext) -> VideoRef:
        video_bytes = await self._execute_video_task(context)
        return await context.video_from_bytes(video_bytes)


class SeedanceV1LiteImageToVideo(KieVideoBaseNode):
    """Generate videos from images using ByteDance's Seedance V1 Lite model via Kie.ai.

    kie, seedance, bytedance, video generation, ai, image-to-video, lite, fast

    Seedance V1 Lite offers fast image-to-video generation with efficient processing,
    transforming reference images into dynamic video sequences.

    Use cases:
    - Fast image animation from reference images
    - Quick video prototyping
    - Animate static images efficiently
    - Rapid iteration on image-to-video concepts
    """

    _expose_as_tool: ClassVar[bool] = True

    image: ImageRef = Field(
        default=ImageRef(),
        description="The reference image to animate into a video.",
    )

    prompt: str = Field(
        default="",
        description="Optional text to guide the video generation.",
    )

    class Resolution(str, Enum):
        HD_720P = "720p"
        FULL_HD_1080P = "1080p"

    resolution: Resolution = Field(
        default=Resolution.HD_720P,
        description="Video resolution.",
    )

    duration: int = Field(
        default=5,
        description="Video duration in seconds.",
        ge=1,
        le=10,
    )

    camera_fixed: bool = Field(
        default=False,
        description="Whether to keep the camera fixed or allow camera movement.",
    )

    seed: int = Field(
        default=-1,
        description="Random seed for reproducible results. Use -1 for random seed.",
    )

    enable_safety_checker: bool = Field(
        default=True,
        description="Enable safety checker to filter inappropriate content.",
    )

    def _get_model(self) -> str:
        return "bytedance/v1-lite-image-to-video"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if not self.image.is_set():
            raise ValueError("Image is required")
        if context is None:
            raise ValueError("Context is required for image upload")
        image_url = await self._upload_image(context, self.image)
        payload: dict[str, Any] = {
            "image_url": image_url,
            "resolution": self.resolution.value,
            "duration": str(self.duration),
            "camera_fixed": self.camera_fixed,
            "seed": self.seed,
            "enable_safety_checker": self.enable_safety_checker,
        }
        if self.prompt:
            payload["prompt"] = self.prompt
        return payload

    async def process(self, context: ProcessingContext) -> VideoRef:
        video_bytes = await self._execute_video_task(context)
        return await context.video_from_bytes(video_bytes)


class SeedanceV1ProImageToVideo(KieVideoBaseNode):
    """Generate videos from images using ByteDance's Seedance V1 Pro model via Kie.ai.

    kie, seedance, bytedance, video generation, ai, image-to-video, pro, high-quality

    Seedance V1 Pro offers high-quality image-to-video generation with improved
    fidelity, transforming reference images into professional video sequences.

    Use cases:
    - Generate high-quality videos from images
    - Create professional image animations
    - Cinematic image-to-video conversion
    - High-fidelity video production from images
    """

    _expose_as_tool: ClassVar[bool] = True

    image: ImageRef = Field(
        default=ImageRef(),
        description="The reference image to animate into a video.",
    )

    prompt: str = Field(
        default="",
        description="Optional text to guide the video generation.",
    )

    class Resolution(str, Enum):
        HD_720P = "720p"
        FULL_HD_1080P = "1080p"

    resolution: Resolution = Field(
        default=Resolution.HD_720P,
        description="Video resolution.",
    )

    duration: int = Field(
        default=5,
        description="Video duration in seconds.",
        ge=1,
        le=10,
    )

    camera_fixed: bool = Field(
        default=False,
        description="Whether to keep the camera fixed or allow camera movement.",
    )

    seed: int = Field(
        default=-1,
        description="Random seed for reproducible results. Use -1 for random seed.",
    )

    enable_safety_checker: bool = Field(
        default=True,
        description="Enable safety checker to filter inappropriate content.",
    )

    def _get_model(self) -> str:
        return "bytedance/v1-pro-image-to-video"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if not self.image.is_set():
            raise ValueError("Image is required")
        if context is None:
            raise ValueError("Context is required for image upload")
        image_url = await self._upload_image(context, self.image)
        payload: dict[str, Any] = {
            "image_url": image_url,
            "resolution": self.resolution.value,
            "duration": str(self.duration),
            "camera_fixed": self.camera_fixed,
            "seed": self.seed,
            "enable_safety_checker": self.enable_safety_checker,
        }
        if self.prompt:
            payload["prompt"] = self.prompt
        return payload

    async def process(self, context: ProcessingContext) -> VideoRef:
        video_bytes = await self._execute_video_task(context)
        return await context.video_from_bytes(video_bytes)


class SeedanceV1ProFastImageToVideo(KieVideoBaseNode):
    """Generate videos from images using ByteDance's Seedance V1 Pro Fast model via Kie.ai.

    kie, seedance, bytedance, video generation, ai, image-to-video, pro, fast

    Seedance V1 Pro Fast offers balanced image-to-video generation with both
    quality and speed, transforming reference images into high-quality video sequences
    with optimized processing time.

    Use cases:
    - Generate high-quality videos from images efficiently
    - Fast professional image animations
    - Balanced quality and speed image-to-video conversion
    - Production-ready image animation
    """

    _expose_as_tool: ClassVar[bool] = True

    image: ImageRef = Field(
        default=ImageRef(),
        description="The reference image to animate into a video.",
    )

    prompt: str = Field(
        default="",
        description="Optional text to guide the video generation.",
    )

    class Resolution(str, Enum):
        HD_720P = "720p"
        FULL_HD_1080P = "1080p"

    resolution: Resolution = Field(
        default=Resolution.HD_720P,
        description="Video resolution.",
    )

    duration: int = Field(
        default=5,
        description="Video duration in seconds.",
        ge=1,
        le=10,
    )

    camera_fixed: bool = Field(
        default=False,
        description="Whether to keep the camera fixed or allow camera movement.",
    )

    seed: int = Field(
        default=-1,
        description="Random seed for reproducible results. Use -1 for random seed.",
    )

    enable_safety_checker: bool = Field(
        default=True,
        description="Enable safety checker to filter inappropriate content.",
    )

    def _get_model(self) -> str:
        return "bytedance/v1-pro-fast-image-to-video"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if not self.image.is_set():
            raise ValueError("Image is required")
        if context is None:
            raise ValueError("Context is required for image upload")
        image_url = await self._upload_image(context, self.image)
        payload: dict[str, Any] = {
            "image_url": image_url,
            "resolution": self.resolution.value,
            "duration": str(self.duration),
            "camera_fixed": self.camera_fixed,
            "seed": self.seed,
            "enable_safety_checker": self.enable_safety_checker,
        }
        if self.prompt:
            payload["prompt"] = self.prompt
        return payload

    async def process(self, context: ProcessingContext) -> VideoRef:
        video_bytes = await self._execute_video_task(context)
        return await context.video_from_bytes(video_bytes)


class HailuoImageToVideoPro(KieVideoBaseNode):
    """Generate videos from images using MiniMax's Hailuo 2.3 Pro model via Kie.ai.

    kie, hailuo, minimax, video generation, ai, image-to-video, pro, high-quality

    Hailuo 2.3 Pro specializes in realistic character motion, expressive facial
    micro-expressions, and cinematic quality for image-to-video generation.

    Use cases:
    - Generate high-quality cinematic videos from images
    - Create content with expressive facial animations
    - Professional image-to-video production
    - Complex character movements from static images
    """

    _expose_as_tool: ClassVar[bool] = True

    image: ImageRef = Field(
        default=ImageRef(),
        description="The reference image to animate into a video.",
    )

    prompt: str = Field(
        default="",
        description="Optional text to guide the video generation.",
    )

    class Resolution(str, Enum):
        HD_720P = "720p"
        FULL_HD_1080P = "1080p"

    resolution: Resolution = Field(
        default=Resolution.FULL_HD_1080P,
        description="Video resolution.",
    )

    def _get_model(self) -> str:
        return "hailuo/2-3-image-to-video-pro"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if not self.image.is_set():
            raise ValueError("Image is required")
        if context is None:
            raise ValueError("Context is required for image upload")
        image_url = await self._upload_image(context, self.image)
        payload: dict[str, Any] = {
            "image_url": image_url,
            "resolution": self.resolution.value,
        }
        if self.prompt:
            payload["prompt"] = self.prompt
        return payload

    async def process(self, context: ProcessingContext) -> VideoRef:
        video_bytes = await self._execute_video_task(context)
        return await context.video_from_bytes(video_bytes)


class HailuoImageToVideoStandard(KieVideoBaseNode):
    """Generate videos from images using MiniMax's Hailuo 2.3 Standard model via Kie.ai.

    kie, hailuo, minimax, video generation, ai, image-to-video, standard, fast

    Hailuo 2.3 Standard offers efficient image-to-video generation with good quality
    and faster processing times for practical use cases.

    Use cases:
    - Generate quality videos from images efficiently
    - Quick image animation with realistic motion
    - Fast image-to-video prototyping
    - Practical video content creation
    """

    _expose_as_tool: ClassVar[bool] = True

    image: ImageRef = Field(
        default=ImageRef(),
        description="The reference image to animate into a video.",
    )

    prompt: str = Field(
        default="",
        description="Optional text to guide the video generation.",
    )

    class Resolution(str, Enum):
        HD_720P = "720p"
        FULL_HD_1080P = "1080p"

    resolution: Resolution = Field(
        default=Resolution.HD_720P,
        description="Video resolution.",
    )

    def _get_model(self) -> str:
        return "hailuo/2-3-image-to-video-standard"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if not self.image.is_set():
            raise ValueError("Image is required")
        if context is None:
            raise ValueError("Context is required for image upload")
        image_url = await self._upload_image(context, self.image)
        payload: dict[str, Any] = {
            "image_url": image_url,
            "resolution": self.resolution.value,
        }
        if self.prompt:
            payload["prompt"] = self.prompt
        return payload

    async def process(self, context: ProcessingContext) -> VideoRef:
        video_bytes = await self._execute_video_task(context)
        return await context.video_from_bytes(video_bytes)


class KlingTextToVideo(KieVideoBaseNode):
    """Generate videos from text using Kuaishou's Kling 2.6 model via Kie.ai.

    kie, kling, kuaishou, video generation, ai, text-to-video, realistic

    Kling 2.6 offers high-quality text-to-video generation with realistic motion,
    physics consistency, and cinematic quality output.

    Use cases:
    - Generate high-quality videos from text
    - Create realistic motion and physics
    - Cinematic content creation
    - Professional video production
    """

    _expose_as_tool: ClassVar[bool] = True

    prompt: str = Field(
        default="",
        description="The text prompt describing the video to generate.",
    )

    class AspectRatio(str, Enum):
        LANDSCAPE = "16:9"
        PORTRAIT = "9:16"
        SQUARE = "1:1"

    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.LANDSCAPE,
        description="The aspect ratio of the generated video.",
    )

    class Resolution(str, Enum):
        HD_720P = "720p"
        FULL_HD_1080P = "1080p"

    resolution: Resolution = Field(
        default=Resolution.HD_720P,
        description="Video resolution.",
    )

    duration: int = Field(
        default=5,
        description="Video duration in seconds.",
        ge=1,
        le=10,
    )

    seed: int = Field(
        default=-1,
        description="Random seed for reproducible results. Use -1 for random seed.",
    )

    def _get_model(self) -> str:
        return "kling-2.6/text-to-video"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if not self.prompt:
            raise ValueError("Prompt cannot be empty")
        return {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "resolution": self.resolution.value,
            "duration": str(self.duration),
            "seed": self.seed,
        }

    async def process(self, context: ProcessingContext) -> VideoRef:
        video_bytes = await self._execute_video_task(context)
        return await context.video_from_bytes(video_bytes)


class KlingImageToVideo(KieVideoBaseNode):
    """Generate videos from images using Kuaishou's Kling 2.6 model via Kie.ai.

    kie, kling, kuaishou, video generation, ai, image-to-video, realistic

    Kling 2.6 Image-to-Video transforms reference images into high-quality videos
    with realistic motion, physics consistency, and cinematic quality output.

    Use cases:
    - Generate high-quality videos from images
    - Create realistic motion and physics from images
    - Animate static images with cinematic quality
    - Professional image-to-video production
    """

    _expose_as_tool: ClassVar[bool] = True

    image: ImageRef = Field(
        default=ImageRef(),
        description="The reference image to animate into a video.",
    )

    prompt: str = Field(
        default="",
        description="Optional text to guide the video generation.",
    )

    class AspectRatio(str, Enum):
        LANDSCAPE = "16:9"
        PORTRAIT = "9:16"
        SQUARE = "1:1"

    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.LANDSCAPE,
        description="The aspect ratio of the generated video.",
    )

    class Resolution(str, Enum):
        HD_720P = "720p"
        FULL_HD_1080P = "1080p"

    resolution: Resolution = Field(
        default=Resolution.HD_720P,
        description="Video resolution.",
    )

    duration: int = Field(
        default=5,
        description="Video duration in seconds.",
        ge=1,
        le=10,
    )

    seed: int = Field(
        default=-1,
        description="Random seed for reproducible results. Use -1 for random seed.",
    )

    sound: bool = Field(
        default=False,
        description="Whether the generated video includes sound.",
    )

    def _get_model(self) -> str:
        return "kling-2.6/image-to-video"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if not self.image.is_set():
            raise ValueError("Image is required")
        if context is None:
            raise ValueError("Context is required for image upload")
        image_url = await self._upload_image(context, self.image)
        payload: dict[str, Any] = {
            "image_url": image_url,
            "aspect_ratio": self.aspect_ratio.value,
            "resolution": self.resolution.value,
            "duration": "10" if self.duration >= 10 else "5",
            "seed": self.seed,
            "sound": self.sound,
        }
        if self.prompt:
            payload["prompt"] = self.prompt
        return payload

    async def process(self, context: ProcessingContext) -> VideoRef:
        video_bytes = await self._execute_video_task(context)
        return await context.video_from_bytes(video_bytes)


class KlingAIAvatarStandard(KieVideoBaseNode):
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

    def _get_model(self) -> str:
        return "kling/v1-avatar-standard"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if not self.image.is_set():
            raise ValueError("Image is required")
        if not self.audio.is_set():
            raise ValueError("Audio is required")
        if context is None:
            raise ValueError("Context is required for media upload")
        image_url, audio_url = await asyncio.gather(
            self._upload_image(context, self.image),
            self._upload_audio(context, self.audio),
        )
        payload: dict[str, Any] = {
            "image_url": image_url,
            "audio_url": audio_url,
            "mode": self.mode.value,
        }
        if self.prompt:
            payload["prompt"] = self.prompt
        return payload

    async def process(self, context: ProcessingContext) -> VideoRef:
        video_bytes = await self._execute_video_task(context)
        return await context.video_from_bytes(video_bytes)


class KlingAIAvatarPro(KieVideoBaseNode):
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

    def _get_model(self) -> str:
        return "kling/v1-avatar-pro"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if not self.image.is_set():
            raise ValueError("Image is required")
        if not self.audio.is_set():
            raise ValueError("Audio is required")
        if context is None:
            raise ValueError("Context is required for media upload")
        image_url, audio_url = await asyncio.gather(
            self._upload_image(context, self.image),
            self._upload_audio(context, self.audio),
        )
        payload: dict[str, Any] = {
            "image_url": image_url,
            "audio_url": audio_url,
            "mode": self.mode.value,
        }
        if self.prompt:
            payload["prompt"] = self.prompt
        return payload

    async def process(self, context: ProcessingContext) -> VideoRef:
        video_bytes = await self._execute_video_task(context)
        return await context.video_from_bytes(video_bytes)


class GrokImagineImageToVideo(KieVideoBaseNode):
    """Generate videos from images using xAI's Grok Imagine model via Kie.ai.

    kie, grok, xai, video generation, ai, image-to-video, multimodal

    Grok Imagine can transform images into videos with coherent motion and
    synchronized background elements.

    Use cases:
    - Animate static images
    - Create videos from photos
    - Generate motion from images
    """

    _expose_as_tool: ClassVar[bool] = True

    image: ImageRef = Field(
        default=ImageRef(),
        description="The source image to animate.",
    )

    prompt: str = Field(
        default="",
        description="Optional text to guide the animation.",
    )

    class Duration(str, Enum):
        SHORT = "short"
        MEDIUM = "medium"
        LONG = "long"

    duration: Duration = Field(
        default=Duration.MEDIUM,
        description="Duration of the generated video.",
    )

    def _get_model(self) -> str:
        return "grok-imagine/image-to-video"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if not self.image.is_set():
            raise ValueError("Image is required")
        if context is None:
            raise ValueError("Context is required for image upload")
        image_url = await self._upload_image(context, self.image)
        payload: dict[str, Any] = {
            "image": image_url,
            "duration": self.duration.value,
        }
        if self.prompt:
            payload["prompt"] = self.prompt
        return payload

    async def process(self, context: ProcessingContext) -> VideoRef:
        video_bytes = await self._execute_video_task(context)
        return await context.video_from_bytes(video_bytes)


class GrokImagineTextToVideo(KieVideoBaseNode):
    """Generate videos from text using xAI's Grok Imagine model via Kie.ai.

    kie, grok, xai, video generation, ai, text-to-video, multimodal

    Grok Imagine can generate videos from text prompts with coherent motion
    and synchronized background elements.

    Use cases:
    - Generate videos from text descriptions
    - Create cinematic content from prompts
    - Text-to-video generation
    """

    _expose_as_tool: ClassVar[bool] = True

    prompt: str = Field(
        default="",
        description="The text prompt describing the video to generate.",
    )

    class Duration(str, Enum):
        SHORT = "short"
        MEDIUM = "medium"
        LONG = "long"

    duration: Duration = Field(
        default=Duration.MEDIUM,
        description="Duration of the generated video.",
    )

    class Resolution(str, Enum):
        HD_720P = "720p"
        FULL_HD_1080P = "1080p"

    resolution: Resolution = Field(
        default=Resolution.FULL_HD_1080P,
        description="Video resolution.",
    )

    def _get_model(self) -> str:
        return "grok-imagine/text-to-video"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if not self.prompt:
            raise ValueError("Prompt cannot be empty")
        return {
            "prompt": self.prompt,
            "duration": self.duration.value,
            "resolution": self.resolution.value,
        }

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

    def _get_model(self) -> str:
        return "topaz-video-upscaler"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if not self.video.is_set():
            raise ValueError("Video is required")
        if context is None:
            raise ValueError("Context is required for video upload")
        video_url = await self._upload_video(context, self.video)
        return {
            "video": video_url,
            "resolution": self.resolution.value,
            "denoise": self.denoise,
        }

    async def process(self, context: ProcessingContext) -> VideoRef:
        video_bytes = await self._execute_video_task(context)
        return await context.video_from_bytes(video_bytes)
