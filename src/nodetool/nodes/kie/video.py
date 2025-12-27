"""Kie.ai video generation nodes.

This module provides nodes for generating videos using Kie.ai's various APIs.
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
    _max_poll_attempts: int = 240

    @classmethod
    def is_visible(cls) -> bool:
        return cls is not KieVideoBaseNode

    async def _execute_video_task(self, context: ProcessingContext) -> bytes:
        """Execute the full task workflow for video: submit, poll, download."""
        return await self._execute_task(context)

    async def process(self, context: ProcessingContext) -> VideoRef:
        video_bytes = await self._execute_video_task(context)
        return await context.video_from_bytes(video_bytes)


class KlingTextToVideo(KieVideoBaseNode):
    """Generate videos from text using Kuaishou's Kling 2.6 model via Kie.ai."""

    _expose_as_tool: ClassVar[bool] = True

    prompt: str = Field(..., description="The text prompt describing the video.")

    class AspectRatio(str, Enum):
        V16_9 = "16:9"
        V9_16 = "9:16"
        V1_1 = "1:1"

    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.V16_9,
        description="The aspect ratio of the generated video.",
    )

    duration: int = Field(
        default=5,
        description="Video duration in seconds.",
        ge=1,
        le=10,
    )

    class Resolution(str, Enum):
        R768P = "768P"

    resolution: Resolution = Field(
        default=Resolution.R768P,
        description="Video resolution.",
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


class KlingImageToVideo(KieVideoBaseNode):
    """Generate videos from images using Kuaishou's Kling 2.6 model via Kie.ai."""

    _expose_as_tool: ClassVar[bool] = True

    prompt: str = Field(
        default="",
        description="Optional text prompt to guide the video generation.",
    )

    image_input: list[ImageRef] = Field(
        default=[],
        description="Source images for the video generation.",
    )

    sound: bool = Field(
        default=False,
        description="Whether to generate sound for the video.",
    )

    duration: int = Field(
        default=5,
        description="Video duration in seconds.",
    )

    def _get_model(self) -> str:
        return "kling-2.6/image-to-video"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if context is None:
            raise ValueError("Context is required for image upload")

        image_urls = []
        for img in self.image_input:
            if img.is_set():
                url = await self._upload_image(context, img)
                image_urls.append(url)

        return {
            "prompt": self.prompt,
            "image_urls": image_urls,
            "sound": self.sound,
            "duration": self.duration,
        }



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


class GrokImagineTextToVideo(KieVideoBaseNode):
    """Generate videos from text using xAI's Grok Imagine model via Kie.ai."""

    _expose_as_tool: ClassVar[bool] = True

    prompt: str = Field(..., description="The text prompt describing the video.")

    class Resolution(str, Enum):
        R720P = "720p"
        R1080P = "1080p"

    resolution: Resolution = Field(
        default=Resolution.R1080P,
        description="The resolution of the video.",
    )

    class Duration(str, Enum):
        SHORT = "short"
        MEDIUM = "medium"
        LONG = "long"

    duration: Duration = Field(
        default=Duration.MEDIUM,
        description="The duration tier of the video.",
    )

    def _get_model(self) -> str:
        return "grok-imagine/text-to-video"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "resolution": self.resolution.value,
            "duration": self.duration.value,
        }


class GrokImagineImageToVideo(KieVideoBaseNode):
    """Generate videos from images using xAI's Grok Imagine model via Kie.ai."""

    _expose_as_tool: ClassVar[bool] = True

    prompt: str = Field(
        default="",
        description="Optional text guide for the animation.",
    )

    image: ImageRef = Field(..., description="The source image to animate.")

    class Duration(str, Enum):
        SHORT = "short"
        MEDIUM = "medium"
        LONG = "long"

    duration: Duration = Field(
        default=Duration.MEDIUM,
        description="The duration tier of the video.",
    )

    def _get_model(self) -> str:
        return "grok-imagine/image-to-video"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if context is None:
            raise ValueError("Context is required for image upload")

        image_url = await self._upload_image(context, self.image)
        return {
            "prompt": self.prompt,
            "image": image_url,
            "duration": self.duration.value,
        }


class SeedanceBaseNode(KieVideoBaseNode):
    """Base class for Seedance (Bytedance) video generation nodes."""

    class AspectRatio(str, Enum):
        V1_1 = "1:1"
        V16_9 = "16:9"
        V9_16 = "9:16"
        V4_3 = "4:3"
        V3_4 = "3:4"
        V21_9 = "21:9"
        V9_21 = "9:21"

    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.V16_9,
        description="The aspect ratio of the generated video.",
    )

    class Resolution(str, Enum):
        R720P = "720p"

    resolution: Resolution = Field(
        default=Resolution.R720P,
        description="The resolution of the video.",
    )

    class Duration(str, Enum):
        D5 = "5"
        D10 = "10"

    duration: Duration = Field(
        default=Duration.D5,
        description="The duration of the video in seconds.",
    )

    remove_watermark: bool = Field(
        default=True,
        description="Whether to remove the watermark from the video.",
    )

    def _get_common_params(self) -> dict[str, Any]:
        return {
            "aspect_ratio": self.aspect_ratio.value,
            "resolution": self.resolution.value,
            "duration": self.duration.value,
            "remove_watermark": self.remove_watermark,
        }


class SeedanceV1LiteTextToVideo(SeedanceBaseNode):
    """Bytedance 1.0 - text-to-video-lite via Kie.ai."""

    _expose_as_tool: ClassVar[bool] = True

    prompt: str = Field(..., description="The text prompt describing the video.")

    def _get_model(self) -> str:
        return "seedance/v1-lite-text-to-video"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        params = self._get_common_params()
        params["prompt"] = self.prompt
        return params


class SeedanceV1ProTextToVideo(SeedanceBaseNode):
    """Bytedance 1.0 - text-to-video-pro via Kie.ai."""

    _expose_as_tool: ClassVar[bool] = True

    prompt: str = Field(..., description="The text prompt describing the video.")

    def _get_model(self) -> str:
        return "seedance/v1-pro-text-to-video"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        params = self._get_common_params()
        params["prompt"] = self.prompt
        return params


class SeedanceV1LiteImageToVideo(SeedanceBaseNode):
    """Bytedance 1.0 - image-to-video-lite via Kie.ai."""

    _expose_as_tool: ClassVar[bool] = True

    prompt: str = Field(
        default="",
        description="Optional text guide for the video generation.",
    )

    image_input: list[ImageRef] = Field(
        default=[],
        description="Source images for the video generation.",
    )

    def _get_model(self) -> str:
        return "seedance/v1-lite-image-to-video"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if context is None:
            raise ValueError("Context is required for image upload")

        image_urls = []
        for img in self.image_input:
            if img.is_set():
                url = await self._upload_image(context, img)
                image_urls.append(url)

        params = self._get_common_params()
        params["prompt"] = self.prompt
        params["image_urls"] = image_urls
        return params


class SeedanceV1ProImageToVideo(SeedanceBaseNode):
    """Bytedance 1.0 - image-to-video-pro via Kie.ai."""

    _expose_as_tool: ClassVar[bool] = True

    prompt: str = Field(
        default="",
        description="Optional text guide for the video generation.",
    )

    image_input: list[ImageRef] = Field(
        default=[],
        description="Source images for the video generation.",
    )

    def _get_model(self) -> str:
        return "seedance/v1-pro-image-to-video"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if context is None:
            raise ValueError("Context is required for image upload")

        image_urls = []
        for img in self.image_input:
            if img.is_set():
                url = await self._upload_image(context, img)
                image_urls.append(url)

        params = self._get_common_params()
        params["prompt"] = self.prompt
        params["image_urls"] = image_urls
        return params


class SeedanceV1ProFastImageToVideo(SeedanceBaseNode):
    """Bytedance 1.0 - fast-image-to-video-pro via Kie.ai."""

    _expose_as_tool: ClassVar[bool] = True

    image_input: list[ImageRef] = Field(
        default=[],
        description="Source images for the fast video generation.",
    )

    def _get_model(self) -> str:
        return "seedance/v1-pro-fast-image-to-video"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if context is None:
            raise ValueError("Context is required for image upload")

        image_urls = []
        for img in self.image_input:
            if img.is_set():
                url = await self._upload_image(context, img)
                image_urls.append(url)

        params = self._get_common_params()
        params["image_urls"] = image_urls
        return params



class HailuoImageToVideoPro(KieVideoBaseNode):
    """Generate videos from images using MiniMax's Hailuo 2.3 Pro model via Kie.ai.

    kie, hailuo, minimax, video generation, ai, image-to-video, pro

    Hailuo 2.3 Pro offers the highest quality image-to-video generation with
    realistic motion, detailed textures, and cinematic quality.

    Use cases:
    - Generate high-quality cinematic videos from images
    - Create realistic motion and physics
    - Professional video production
    - High-fidelity image animation
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
        R768P = "768P"

    resolution: Resolution = Field(
        default=Resolution.R768P,
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
        R768P = "768P"

    resolution: Resolution = Field(
        default=Resolution.R768P,
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


class HailuoImageToVideo(KieVideoBaseNode):
    """Generate videos from images using MiniMax's Hailuo model via Kie.ai."""

    _expose_as_tool: ClassVar[bool] = True

    prompt: str = Field(
        default="",
        description="Optional text to guide the video generation.",
    )

    image: ImageRef = Field(..., description="The source image to animate.")

    class ModelType(str, Enum):
        PRO = "pro"
        STANDARD = "standard"

    model_type: ModelType = Field(
        default=ModelType.PRO,
        description="The model tier to use.",
    )

    class Resolution(str, Enum):
        R768P = "768P"

    resolution: Resolution = Field(
        default=Resolution.R768P,
        description="The resolution of the video.",
    )

    def _get_model(self) -> str:
        return f"hailuo/2-3-image-to-video-{self.model_type.value}"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if context is None:
            raise ValueError("Context is required for image upload")

        image_url = await self._upload_image(context, self.image)
        return {
            "prompt": self.prompt,
            "image_url": image_url,
            "resolution": self.resolution.value,
        }


class Sora2BaseNode(KieVideoBaseNode):
    """Base class for Sora 2 nodes via Kie.ai."""

    class AspectRatio(str, Enum):
        LANDSCAPE = "landscape"
        PORTRAIT = "portrait"
        SQUARE = "square"

    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.LANDSCAPE,
        description="The aspect ratio of the generated video.",
    )

    remove_watermark: bool = Field(
        default=True,
        description="Whether to remove the watermark from the video.",
    )

    n_frames: int = Field(
        default=10,
        description="Number of frames for the video output.",
        ge=1,
        le=60,
    )


class Sora2ProTextToVideo(Sora2BaseNode):
    """Generate videos from text using Sora 2 Pro via Kie.ai."""

    _expose_as_tool: ClassVar[bool] = True

    prompt: str = Field(..., description="The text prompt describing the video.")

    def _get_model(self) -> str:
        return "sora-2-pro-text-to-video"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "n_frames": self.n_frames,
            "remove_watermark": self.remove_watermark,
        }


class Sora2ProImageToVideo(Sora2BaseNode):
    """Generate videos from images using Sora 2 Pro via Kie.ai."""

    _expose_as_tool: ClassVar[bool] = True

    prompt: str = Field(
        default="",
        description="Optional text guide for the video generation.",
    )

    image: ImageRef = Field(..., description="The source image to animate.")

    def _get_model(self) -> str:
        return "sora-2-pro-image-to-video"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if context is None:
            raise ValueError("Context is required for image upload")

        image_url = await self._upload_image(context, self.image)
        return {
            "prompt": self.prompt,
            "image_url": image_url,
            "n_frames": self.n_frames,
            "remove_watermark": self.remove_watermark,
        }


class Sora2ProStoryboard(Sora2BaseNode):
    """Generate videos from storyboards using Sora 2 Pro via Kie.ai."""

    _expose_as_tool: ClassVar[bool] = True

    prompt: str = Field(..., description="The text prompt describing the video.")

    image_input: list[ImageRef] = Field(
        default=[],
        description="Source images for the storyboard animation.",
    )

    def _get_model(self) -> str:
        return "sora-2-pro-story-board"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if context is None:
            raise ValueError("Context is required for image upload")

        image_urls = []
        for img in self.image_input:
            if img.is_set():
                url = await self._upload_image(context, img)
                image_urls.append(url)

        return {
            "prompt": self.prompt,
            "image_urls": image_urls,
            "n_frames": self.n_frames,
            "remove_watermark": self.remove_watermark,
        }


class Sora2TextToVideo(Sora2BaseNode):
    """Generate videos from text using Sora 2 Standard via Kie.ai."""

    _expose_as_tool: ClassVar[bool] = True

    prompt: str = Field(..., description="The text prompt describing the video.")

    n_frames: int = Field(
        default=10,
        description="Number of frames for the video output.",
        ge=1,
        le=60,
    )

    def _get_model(self) -> str:
        return "sora-2-text-to-video"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "n_frames": self.n_frames,
            "remove_watermark": self.remove_watermark,
        }


class WanMultiShotTextToVideoPro(KieVideoBaseNode):
    """Generate videos from text using Alibaba's Wan 2.1 model via Kie.ai."""

    _expose_as_tool: ClassVar[bool] = True

    prompt: str = Field(..., description="The text prompt describing the video.")

    class AspectRatio(str, Enum):
        V16_9 = "16:9"
        V9_16 = "9:16"
        V1_1 = "1:1"
        V4_3 = "4:3"
        V3_4 = "3:4"

    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.V16_9,
        description="The aspect ratio of the generated video.",
    )

    class Resolution(str, Enum):
        R720P = "720p"
        R1080P = "1080p"

    resolution: Resolution = Field(
        default=Resolution.R1080P,
        description="The resolution of the video.",
    )

    class Duration(str, Enum):
        D5 = "5"
        D10 = "10"

    duration: Duration = Field(
        default=Duration.D5,
        description="The duration of the video in seconds.",
    )

    remove_watermark: bool = Field(
        default=True,
        description="Whether to remove the watermark from the video.",
    )

    def _get_model(self) -> str:
        return "wan/v2-1-multi-shot-text-to-video-pro"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "resolution": self.resolution.value,
            "duration": self.duration.value,
            "remove_watermark": self.remove_watermark,
        }


class Wan26TextToVideo(KieVideoBaseNode):
    """Generate videos from text using Alibaba's Wan 2.6 model via Kie.ai."""

    _expose_as_tool: ClassVar[bool] = True

    prompt: str = Field(..., description="The text prompt describing the video.")

    class Duration(str, Enum):
        D5 = "5"
        D10 = "10"

    duration: Duration = Field(
        default=Duration.D5,
        description="The duration of the video in seconds.",
    )

    class Resolution(str, Enum):
        R1080P = "1080p"
        R720P = "720p"

    resolution: Resolution = Field(
        default=Resolution.R1080P,
        description="The resolution of the video.",
    )

    def _get_model(self) -> str:
        return "wan/2-6-text-to-video"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "duration": self.duration.value,
            "resolution": self.resolution.value,
        }


class Wan26ImageToVideo(KieVideoBaseNode):
    """Generate videos from images using Alibaba's Wan 2.6 model via Kie.ai."""

    _expose_as_tool: ClassVar[bool] = True

    prompt: str = Field(..., description="The text prompt describing the video.")

    image_input: list[ImageRef] = Field(
        default=[],
        description="Source images for the video generation.",
    )

    class Duration(str, Enum):
        D5 = "5"
        D10 = "10"

    duration: Duration = Field(
        default=Duration.D5,
        description="The duration of the video in seconds.",
    )

    class Resolution(str, Enum):
        R1080P = "1080p"
        R720P = "720p"

    resolution: Resolution = Field(
        default=Resolution.R1080P,
        description="The resolution of the video.",
    )

    def _get_model(self) -> str:
        return "wan/2-6-image-to-video"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if context is None:
            raise ValueError("Context is required for image upload")

        image_urls = []
        for img in self.image_input:
            if img.is_set():
                url = await self._upload_image(context, img)
                image_urls.append(url)

        return {
            "prompt": self.prompt,
            "image_urls": image_urls,
            "duration": self.duration.value,
            "resolution": self.resolution.value,
        }


class Wan26VideoToVideo(KieVideoBaseNode):
    """Generate videos from videos using Alibaba's Wan 2.6 model via Kie.ai."""

    _expose_as_tool: ClassVar[bool] = True

    prompt: str = Field(..., description="The text prompt describing the changes.")

    video_input: list[VideoRef] = Field(
        default=[],
        description="Source videos for the video-to-video task.",
    )

    class Duration(str, Enum):
        D5 = "5"
        D10 = "10"

    duration: Duration = Field(
        default=Duration.D5,
        description="The duration of the video in seconds.",
    )

    class Resolution(str, Enum):
        R1080P = "1080p"
        R720P = "720p"

    resolution: Resolution = Field(
        default=Resolution.R1080P,
        description="The resolution of the video.",
    )

    def _get_model(self) -> str:
        return "wan/2-6-video-to-video"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if context is None:
            raise ValueError("Context is required for video upload")

        video_urls = []
        for vid in self.video_input:
            if vid.is_set():
                url = await self._upload_video(context, vid)
                video_urls.append(url)

        return {
            "prompt": self.prompt,
            "video_urls": video_urls,
            "duration": self.duration.value,
            "resolution": self.resolution.value,
        }


class TopazVideoUpscale(KieVideoBaseNode):
    """Upscale and enhance videos using Topaz Labs AI via Kie.ai."""

    _expose_as_tool: ClassVar[bool] = True

    video: VideoRef = Field(..., description="The video to upscale.")

    class Resolution(str, Enum):
        R1080P = "1080p"
        R4K = "4k"

    resolution: Resolution = Field(
        default=Resolution.R1080P,
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
        if context is None:
            raise ValueError("Context is required for video upload")

        video_url = await self._upload_video(context, self.video)
        return {
            "video": video_url,
            "resolution": self.resolution.value,
            "denoise": self.denoise,
        }


class InfinitalkV1(KieVideoBaseNode):
    """Generate videos using Infinitalk v1 (image-to-video) via Kie.ai."""

    _expose_as_tool: ClassVar[bool] = True

    prompt: str = Field(
        default="",
        description="Optional text guide for the video generation.",
    )

    image: ImageRef = Field(..., description="The source image.")

    audio: AudioRef = Field(..., description="The source audio track.")

    class Resolution(str, Enum):
        R480P = "480p"

    resolution: Resolution = Field(
        default=Resolution.R480P,
        description="Video resolution.",
    )

    def _get_model(self) -> str:
        return "infinitalk/v1"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if context is None:
            raise ValueError("Context is required for media upload")

        image_url, audio_url = await asyncio.gather(
            self._upload_image(context, self.image),
            self._upload_audio(context, self.audio),
        )

        return {
            "prompt": self.prompt,
            "image_url": image_url,
            "audio_url": audio_url,
            "resolution": self.resolution.value,
        }
