"""Kie.ai video generation nodes.

This module provides nodes for generating videos using Kie.ai's various APIs.
"""

import asyncio
from enum import Enum
from typing import Any, ClassVar

import aiohttp
from pydantic import Field

from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import AudioRef, ImageRef, VideoRef
from nodetool.workflows.processing_context import ProcessingContext

from .image import KieBaseNode, KIE_API_BASE_URL

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

    async def _execute_video_task(
        self, context: ProcessingContext
    ) -> tuple[bytes, str]:
        """Execute the full task workflow for video: submit, poll, download."""
        return await self._execute_task(context)

    async def process(self, context: ProcessingContext) -> VideoRef:
        video_bytes, task_id = await self._execute_video_task(context)
        return await context.video_from_bytes(video_bytes)


class KlingTextToVideo(KieVideoBaseNode):
    """Generate videos from text using Kuaishou's Kling 2.6 model via Kie.ai.

    kie, kling, kuaishou, video generation, ai, text-to-video, 2.6

    Kling 2.6 produces high-quality videos from text descriptions with
    realistic motion, natural lighting, and cinematic detail.
    """
    _auto_save_asset: ClassVar[bool] = True

    _expose_as_tool: ClassVar[bool] = True

    @classmethod
    def get_title(cls) -> str:
        return "Kling 2.6 Text To Video"

    prompt: str = Field(
        default="A cinematic video with smooth motion, natural lighting, and high detail.",
        description="The text prompt describing the video.",
    )

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
    """Generate videos from images using Kuaishou's Kling 2.6 model via Kie.ai.

    kie, kling, kuaishou, video generation, ai, image-to-video, 2.6

    Transforms static images into dynamic videos with realistic motion
    and temporal consistency while preserving the original visual style.
    """
    _auto_save_asset: ClassVar[bool] = True

    _expose_as_tool: ClassVar[bool] = True

    @classmethod
    def get_title(cls) -> str:
        return "Kling 2.6 Image To Video"

    prompt: str = Field(
        default="A cinematic video with smooth motion, natural lighting, and high detail.",
        description="Optional text prompt to guide the video generation.",
    )

    image1: ImageRef = Field(
        default=ImageRef(),
        description="First source image for the video generation.",
    )

    image2: ImageRef = Field(
        default=ImageRef(),
        description="Second source image (optional).",
    )

    image3: ImageRef = Field(
        default=ImageRef(),
        description="Third source image (optional).",
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
        if not self.prompt:
            raise ValueError("Prompt is required")
        if context is None:
            raise ValueError("Context is required for image upload")

        image_urls = []
        for img in [self.image1, self.image2, self.image3]:
            if img.is_set():
                url = await self._upload_image(context, img)
                image_urls.append(url)

        return {
            "prompt": self.prompt,
            "image_urls": image_urls,
            "sound": self.sound,
            "duration": str(self.duration),
        }


class KlingAIAvatarStandard(KieVideoBaseNode):
    """Generate talking avatar videos using Kuaishou's Kling AI via Kie.ai.

    kie, kling, kuaishou, avatar, video generation, ai, talking-head, lip-sync

    Transforms a photo plus audio track into a lip-synced talking avatar video
    with natural-looking speech animation and consistent identity.
    """
    _auto_save_asset: ClassVar[bool] = True

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
        default="A cinematic video with smooth motion, natural lighting, and high detail.",
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
        video_bytes, task_id = await self._execute_video_task(context)
        return await context.video_from_bytes(video_bytes)


class KlingAIAvatarPro(KieVideoBaseNode):
    """Generate talking avatar videos using Kuaishou's Kling AI via Kie.ai.

    kie, kling, kuaishou, avatar, video generation, ai, talking-head, lip-sync

    Transforms a photo plus audio track into a lip-synced talking avatar video
    with natural-looking speech animation and consistent identity.
    """
    _auto_save_asset: ClassVar[bool] = True

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
        default="A cinematic video with smooth motion, natural lighting, and high detail.",
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
        video_bytes, task_id = await self._execute_video_task(context)
        return await context.video_from_bytes(video_bytes)


class GrokImagineTextToVideo(KieVideoBaseNode):
    """Generate videos from text using xAI's Grok Imagine model via Kie.ai.

    kie, grok, xai, video generation, ai, text-to-video, multimodal

    Grok Imagine generates videos from text prompts using xAI's
    multimodal generation capabilities.
    """
    _auto_save_asset: ClassVar[bool] = True

    _expose_as_tool: ClassVar[bool] = True

    @classmethod
    def get_title(cls) -> str:
        return "Grok Imagine Text To Video"

    prompt: str = Field(
        default="A cinematic video with smooth motion, natural lighting, and high detail.",
        description="The text prompt describing the video.",
    )

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
        if not self.prompt:
            raise ValueError("Prompt is required")
        return {
            "prompt": self.prompt,
            "resolution": self.resolution.value,
            "duration": self.duration.value,
        }


class GrokImagineImageToVideo(KieVideoBaseNode):
    """Generate videos from images using xAI's Grok Imagine model via Kie.ai.

    kie, grok, xai, video generation, ai, image-to-video, multimodal

    Grok Imagine transforms images into videos using xAI's
    multimodal generation capabilities.
    """
    _auto_save_asset: ClassVar[bool] = True

    _expose_as_tool: ClassVar[bool] = True

    @classmethod
    def get_title(cls) -> str:
        return "Grok Imagine Image To Video"

    prompt: str = Field(
        default="A cinematic video with smooth motion, natural lighting, and high detail.",
        description="Optional text guide for the animation.",
    )

    image: ImageRef = Field(
        default=ImageRef(),
        description="The source image to animate.",
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
        return "grok-imagine/image-to-video"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if not self.prompt:
            raise ValueError("Prompt is required")
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
    """Bytedance 1.0 - text-to-video-lite via Kie.ai.

    kie, seedance, bytedance, video generation, ai, text-to-video, lite

    Seedance V1 Lite offers efficient text-to-video generation
    with good quality and faster processing times.
    """
    _auto_save_asset: ClassVar[bool] = True

    _expose_as_tool: ClassVar[bool] = True

    @classmethod
    def get_title(cls) -> str:
        return "Seedance V1 Lite Text To Video"

    prompt: str = Field(
        default="A cinematic video with smooth motion, natural lighting, and high detail.",
        description="The text prompt describing the video.",
    )

    def _get_model(self) -> str:
        return "seedance/v1-lite-text-to-video"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if not self.prompt:
            raise ValueError("Prompt is required")
        params = self._get_common_params()
        params["prompt"] = self.prompt
        return params


class SeedanceV1ProTextToVideo(SeedanceBaseNode):
    """Bytedance 1.0 - text-to-video-pro via Kie.ai."""
    _auto_save_asset: ClassVar[bool] = True

    _expose_as_tool: ClassVar[bool] = True

    @classmethod
    def get_title(cls) -> str:
        return "Seedance V1 Pro Text To Video"

    prompt: str = Field(
        default="A cinematic video with smooth motion, natural lighting, and high detail.",
        description="The text prompt describing the video.",
    )

    def _get_model(self) -> str:
        return "seedance/v1-pro-text-to-video"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if not self.prompt:
            raise ValueError("Prompt is required")
        params = self._get_common_params()
        params["prompt"] = self.prompt
        return params


class SeedanceV1LiteImageToVideo(SeedanceBaseNode):
    """Bytedance 1.0 - image-to-video-lite via Kie.ai."""
    _auto_save_asset: ClassVar[bool] = True

    _expose_as_tool: ClassVar[bool] = True

    @classmethod
    def get_title(cls) -> str:
        return "Seedance V1 Lite Image To Video"

    prompt: str = Field(
        default="A cinematic video with smooth motion, natural lighting, and high detail.",
        description="Optional text guide for the video generation.",
    )

    image1: ImageRef = Field(
        default=ImageRef(),
        description="First source image for the video generation.",
    )

    image2: ImageRef = Field(
        default=ImageRef(),
        description="Second source image (optional).",
    )

    image3: ImageRef = Field(
        default=ImageRef(),
        description="Third source image (optional).",
    )

    def _get_model(self) -> str:
        return "seedance/v1-lite-image-to-video"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if not self.prompt:
            raise ValueError("Prompt is required")
        if context is None:
            raise ValueError("Context is required for image upload")

        image_urls = []
        for img in [self.image1, self.image2, self.image3]:
            if img.is_set():
                url = await self._upload_image(context, img)
                image_urls.append(url)

        params = self._get_common_params()
        params["prompt"] = self.prompt
        params["image_urls"] = image_urls
        return params


class SeedanceV1ProImageToVideo(SeedanceBaseNode):
    """Bytedance 1.0 - image-to-video-pro via Kie.ai."""
    _auto_save_asset: ClassVar[bool] = True

    _expose_as_tool: ClassVar[bool] = True

    @classmethod
    def get_title(cls) -> str:
        return "Seedance V1 Pro Image To Video"

    prompt: str = Field(
        default="A cinematic video with smooth motion, natural lighting, and high detail.",
        description="Optional text guide for the video generation.",
    )

    image1: ImageRef = Field(
        default=ImageRef(),
        description="First source image for the video generation.",
    )

    image2: ImageRef = Field(
        default=ImageRef(),
        description="Second source image (optional).",
    )

    image3: ImageRef = Field(
        default=ImageRef(),
        description="Third source image (optional).",
    )

    def _get_model(self) -> str:
        return "seedance/v1-pro-image-to-video"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if not self.prompt:
            raise ValueError("Prompt is required")
        if context is None:
            raise ValueError("Context is required for image upload")

        image_urls = []
        for img in [self.image1, self.image2, self.image3]:
            if img.is_set():
                url = await self._upload_image(context, img)
                image_urls.append(url)

        params = self._get_common_params()
        params["prompt"] = self.prompt
        params["image_urls"] = image_urls
        return params


class SeedanceV1ProFastImageToVideo(SeedanceBaseNode):
    """Bytedance 1.0 - fast-image-to-video-pro via Kie.ai."""
    _auto_save_asset: ClassVar[bool] = True

    _expose_as_tool: ClassVar[bool] = True

    @classmethod
    def get_title(cls) -> str:
        return "Seedance V1 Pro Fast Image To Video"

    image1: ImageRef = Field(
        default=ImageRef(),
        description="First source image for the video generation.",
    )

    image2: ImageRef = Field(
        default=ImageRef(),
        description="Second source image (optional).",
    )

    image3: ImageRef = Field(
        default=ImageRef(),
        description="Third source image (optional).",
    )

    def _get_model(self) -> str:
        return "seedance/v1-pro-fast-image-to-video"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if context is None:
            raise ValueError("Context is required for image upload")

        image_urls = []
        for img in [self.image1, self.image2, self.image3]:
            if img.is_set():
                url = await self._upload_image(context, img)
                image_urls.append(url)

        params = self._get_common_params()
        params["image_urls"] = image_urls
        return params


class HailuoTextToVideoPro(KieVideoBaseNode):
    """Generate videos from text using MiniMax's Hailuo 2.3 Pro model via Kie.ai.

    kie, hailuo, minimax, video generation, ai, text-to-video, pro

    Hailuo 2.3 Pro offers the highest quality text-to-video generation with
    realistic motion, detailed textures, and cinematic quality.
    """
    _auto_save_asset: ClassVar[bool] = True

    _expose_as_tool: ClassVar[bool] = True

    @classmethod
    def get_title(cls) -> str:
        return "Hailuo 2.3 Pro Text To Video"

    prompt: str = Field(
        default="A cinematic video with smooth motion, natural lighting, and high detail.",
        description="The text prompt describing the video.",
    )

    class Duration(str, Enum):
        D6 = "6"
        D10 = "10"

    duration: Duration = Field(
        default=Duration.D6,
        description="The duration of the video in seconds. 10s is not supported for 1080p.",
    )

    class Resolution(str, Enum):
        R768P = "768P"
        R1080P = "1080P"

    resolution: Resolution = Field(
        default=Resolution.R768P,
        description="Video resolution.",
    )

    def _get_model(self) -> str:
        return "hailuo/2-3-text-to-video-pro"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if not self.prompt:
            raise ValueError("Prompt is required")
        if (
            self.resolution == self.Resolution.R1080P
            and self.duration == self.Duration.D10
        ):
            raise ValueError("10s duration is not supported for 1080p resolution.")

        return {
            "prompt": self.prompt,
            "resolution": self.resolution.value,
            "duration": self.duration.value,
        }

    async def process(self, context: ProcessingContext) -> VideoRef:
        video_bytes, task_id = await self._execute_video_task(context)
        return await context.video_from_bytes(video_bytes)


class HailuoTextToVideoStandard(KieVideoBaseNode):
    """Generate videos from text using MiniMax's Hailuo 2.3 Standard model via Kie.ai.

    kie, hailuo, minimax, video generation, ai, text-to-video, standard, fast
    """
    _auto_save_asset: ClassVar[bool] = True

    _expose_as_tool: ClassVar[bool] = True

    @classmethod
    def get_title(cls) -> str:
        return "Hailuo 2.3 Standard Text To Video"

    prompt: str = Field(
        default="A cinematic video with smooth motion, natural lighting, and high detail.",
        description="The text prompt describing the video.",
    )

    class Duration(str, Enum):
        D6 = "6"
        D10 = "10"

    duration: Duration = Field(
        default=Duration.D6,
        description="The duration of the video in seconds. 10s is not supported for 1080p.",
    )

    class Resolution(str, Enum):
        R768P = "768P"
        R1080P = "1080P"

    resolution: Resolution = Field(
        default=Resolution.R768P,
        description="Video resolution.",
    )

    def _get_model(self) -> str:
        return "hailuo/2-3-text-to-video-standard"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if not self.prompt:
            raise ValueError("Prompt is required")
        if (
            self.resolution == self.Resolution.R1080P
            and self.duration == self.Duration.D10
        ):
            raise ValueError("10s duration is not supported for 1080p resolution.")

        return {
            "prompt": self.prompt,
            "resolution": self.resolution.value,
            "duration": self.duration.value,
        }

    async def process(self, context: ProcessingContext) -> VideoRef:
        video_bytes, task_id = await self._execute_video_task(context)
        return await context.video_from_bytes(video_bytes)


class HailuoImageToVideoPro(KieVideoBaseNode):
    """Generate videos from images using MiniMax's Hailuo 2.3 Pro model via Kie.ai.

    kie, hailuo, minimax, video generation, ai, image-to-video, pro

    Hailuo 2.3 Pro offers the highest quality image-to-video generation with
    realistic motion, detailed textures, and cinematic quality.
    """
    _auto_save_asset: ClassVar[bool] = True

    _expose_as_tool: ClassVar[bool] = True

    @classmethod
    def get_title(cls) -> str:
        return "Hailuo 2.3 Pro Image To Video"

    image: ImageRef = Field(
        default=ImageRef(),
        description="The reference image to animate into a video.",
    )

    prompt: str = Field(
        default="A cinematic video with smooth motion, natural lighting, and high detail.",
        description="Optional text to guide the video generation.",
    )

    class Duration(str, Enum):
        D6 = "6"
        D10 = "10"

    duration: Duration = Field(
        default=Duration.D6,
        description="The duration of the video in seconds. 10s is not supported for 1080p.",
    )

    class Resolution(str, Enum):
        R768P = "768P"
        R1080P = "1080P"

    resolution: Resolution = Field(
        default=Resolution.R768P,
        description="Video resolution.",
    )

    def _get_model(self) -> str:
        return "hailuo/2-3-image-to-video-pro"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if not self.prompt:
            raise ValueError("Prompt is required")
        if not self.image.is_set():
            raise ValueError("Image is required")
        if context is None:
            raise ValueError("Context is required for image upload")
        image_url = await self._upload_image(context, self.image)

        if (
            self.resolution == self.Resolution.R1080P
            and self.duration == self.Duration.D10
        ):
            raise ValueError("10s duration is not supported for 1080p resolution.")

        payload: dict[str, Any] = {
            "image_url": image_url,
            "resolution": self.resolution.value,
            "duration": self.duration.value,
        }
        if self.prompt:
            payload["prompt"] = self.prompt
        return payload

    async def process(self, context: ProcessingContext) -> VideoRef:
        video_bytes, task_id = await self._execute_video_task(context)
        return await context.video_from_bytes(video_bytes)


class HailuoImageToVideoStandard(KieVideoBaseNode):
    """Generate videos from images using MiniMax's Hailuo 2.3 Standard model via Kie.ai.

    kie, hailuo, minimax, video generation, ai, image-to-video, standard, fast

    Hailuo 2.3 Standard offers efficient image-to-video generation with good quality
    and faster processing times for practical use cases.
    """
    _auto_save_asset: ClassVar[bool] = True

    _expose_as_tool: ClassVar[bool] = True

    @classmethod
    def get_title(cls) -> str:
        return "Hailuo 2.3 Standard Image To Video"

    image: ImageRef = Field(
        default=ImageRef(),
        description="The reference image to animate into a video.",
    )

    prompt: str = Field(
        default="A cinematic video with smooth motion, natural lighting, and high detail.",
        description="Optional text to guide the video generation.",
    )

    class Duration(str, Enum):
        D6 = "6"
        D10 = "10"

    duration: Duration = Field(
        default=Duration.D6,
        description="The duration of the video in seconds. 10s is not supported for 1080p.",
    )

    class Resolution(str, Enum):
        R768P = "768P"
        R1080P = "1080P"

    resolution: Resolution = Field(
        default=Resolution.R768P,
        description="Video resolution.",
    )

    def _get_model(self) -> str:
        return "hailuo/2-3-image-to-video-standard"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if not self.prompt:
            raise ValueError("Prompt is required")
        if not self.image.is_set():
            raise ValueError("Image is required")
        if context is None:
            raise ValueError("Context is required for image upload")
        image_url = await self._upload_image(context, self.image)

        if (
            self.resolution == self.Resolution.R1080P
            and self.duration == self.Duration.D10
        ):
            raise ValueError("10s duration is not supported for 1080p resolution.")

        payload: dict[str, Any] = {
            "image_url": image_url,
            "resolution": self.resolution.value,
            "duration": self.duration.value,
        }
        if self.prompt:
            payload["prompt"] = self.prompt
        return payload

    async def process(self, context: ProcessingContext) -> VideoRef:
        video_bytes, task_id = await self._execute_video_task(context)
        return await context.video_from_bytes(video_bytes)


class Kling25TurboTextToVideo(KieVideoBaseNode):
    """Generate videos from text using Kuaishou's Kling 2.5 Turbo model via Kie.ai.

    kie, kling, kuaishou, video generation, ai, text-to-video, turbo

    Kling 2.5 Turbo offers improved prompt adherence, fluid motion,
    consistent artistic styles, and realistic physics simulation.
    """
    _auto_save_asset: ClassVar[bool] = True

    _expose_as_tool: ClassVar[bool] = True

    @classmethod
    def get_title(cls) -> str:
        return "Kling 2.5 Turbo Text To Video"

    prompt: str = Field(
        default="A cinematic video with smooth motion, natural lighting, and high detail.",
        description="The text prompt describing the video.",
    )

    class Duration(str, Enum):
        D5 = "5"
        D10 = "10"

    duration: Duration = Field(
        default=Duration.D5,
        description="Video duration in seconds.",
    )

    class AspectRatio(str, Enum):
        V16_9 = "16:9"
        V9_16 = "9:16"
        V1_1 = "1:1"

    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.V16_9,
        description="The aspect ratio of the generated video.",
    )

    negative_prompt: str = Field(
        default="",
        description="Things to avoid in the generated video.",
    )

    cfg_scale: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="The CFG scale for prompt adherence. Lower values allow more creativity.",
    )

    def _get_model(self) -> str:
        return "kling/v2-5-turbo-text-to-video-pro"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if not self.prompt:
            raise ValueError("Prompt is required")

        payload: dict[str, Any] = {
            "prompt": self.prompt,
            "duration": self.duration.value,
            "aspect_ratio": self.aspect_ratio.value,
            "cfg_scale": self.cfg_scale,
        }
        if self.negative_prompt:
            payload["negative_prompt"] = self.negative_prompt
        return payload


class Kling25TurboImageToVideo(KieVideoBaseNode):
    """Generate videos from images using Kuaishou's Kling 2.5 Turbo model via Kie.ai.

    kie, kling, kuaishou, video generation, ai, image-to-video, turbo

    Transforms a static image into a dynamic video while preserving
    visual style, colors, lighting, and texture.
    """
    _auto_save_asset: ClassVar[bool] = True

    _expose_as_tool: ClassVar[bool] = True

    @classmethod
    def get_title(cls) -> str:
        return "Kling 2.5 Turbo Image To Video"

    prompt: str = Field(
        default="A cinematic video with smooth motion, natural lighting, and high detail.",
        description="Text description to guide the video generation.",
    )

    image: ImageRef = Field(
        default=ImageRef(),
        description="The source image to animate.",
    )

    tail_image: ImageRef = Field(
        default=ImageRef(),
        description="Tail frame image for the video (optional).",
    )

    class Duration(str, Enum):
        D5 = "5"
        D10 = "10"

    duration: Duration = Field(
        default=Duration.D5,
        description="Video duration in seconds.",
    )

    negative_prompt: str = Field(
        default="",
        description="Elements to avoid in the video.",
    )

    cfg_scale: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="The CFG scale for prompt adherence. Lower values allow more creativity.",
    )

    def _get_model(self) -> str:
        return "kling/v2-5-turbo-image-to-video-pro"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if not self.prompt:
            raise ValueError("Prompt is required")
        if not self.image.is_set():
            raise ValueError("Image is required")
        if context is None:
            raise ValueError("Context is required for image upload")

        image_url = await self._upload_image(context, self.image)

        payload: dict[str, Any] = {
            "prompt": self.prompt,
            "image_url": image_url,
            "duration": self.duration.value,
            "cfg_scale": self.cfg_scale,
        }
        if self.tail_image.is_set():
            tail_image_url = await self._upload_image(context, self.tail_image)
            payload["tail_image_url"] = tail_image_url
        if self.negative_prompt:
            payload["negative_prompt"] = self.negative_prompt
        return payload


class Sora2Frames(str, Enum):
    _10s = "10"
    _15s = "15"


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

    n_frames: Sora2Frames = Field(
        default=Sora2Frames._10s,
        description="Number of frames for the video output.",
    )


class Sora2ProTextToVideo(Sora2BaseNode):
    """Generate videos from text using Sora 2 Pro via Kie.ai.

    kie, sora, openai, video generation, ai, text-to-video, pro

    Sora 2 Pro generates high-quality videos from text descriptions
    with advanced motion and temporal consistency.
    """
    _auto_save_asset: ClassVar[bool] = True

    _expose_as_tool: ClassVar[bool] = True

    @classmethod
    def get_title(cls) -> str:
        return "Sora 2 Pro Text To Video"

    prompt: str = Field(
        default="A cinematic video with smooth motion, natural lighting, and high detail.",
        description="The text prompt describing the video.",
    )

    def _get_model(self) -> str:
        return "sora-2-pro-text-to-video"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if not self.prompt:
            raise ValueError("Prompt is required")
        return {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "n_frames": self.n_frames.value,
            "remove_watermark": self.remove_watermark,
        }


class Sora2ProImageToVideo(Sora2BaseNode):
    """Generate videos from images using Sora 2 Pro via Kie.ai.

    kie, sora, openai, video generation, ai, image-to-video, pro

    Sora 2 Pro transforms images into high-quality videos with
    realistic motion and temporal consistency.
    """
    _auto_save_asset: ClassVar[bool] = True

    _expose_as_tool: ClassVar[bool] = True

    @classmethod
    def get_title(cls) -> str:
        return "Sora 2 Pro Image To Video"

    prompt: str = Field(
        default="A cinematic video with smooth motion, natural lighting, and high detail.",
        description="Optional text guide for the video generation.",
    )

    image: ImageRef = Field(
        default=ImageRef(),
        description="The source image to animate.",
    )

    def _get_model(self) -> str:
        return "sora-2-pro-image-to-video"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if not self.prompt:
            raise ValueError("Prompt is required")
        if context is None:
            raise ValueError("Context is required for image upload")

        image_url = await self._upload_image(context, self.image)
        return {
            "prompt": self.prompt,
            "image_url": image_url,
            "n_frames": self.n_frames.value,
            "remove_watermark": self.remove_watermark,
        }


class Sora2ProStoryboard(Sora2BaseNode):
    """Generate videos from storyboards using Sora 2 Pro via Kie.ai.

    kie, sora, openai, video generation, ai, storyboard, pro

    Sora 2 Pro creates videos from storyboard sequences with
    consistent characters and scenes across frames.
    """
    _auto_save_asset: ClassVar[bool] = True

    _expose_as_tool: ClassVar[bool] = True

    @classmethod
    def get_title(cls) -> str:
        return "Sora 2 Pro Storyboard"

    prompt: str = Field(
        default="A cinematic video with smooth motion, natural lighting, and high detail.",
        description="The text prompt describing the video.",
    )

    image1: ImageRef = Field(
        default=ImageRef(),
        description="First source image for the video generation.",
    )

    image2: ImageRef = Field(
        default=ImageRef(),
        description="Second source image (optional).",
    )

    image3: ImageRef = Field(
        default=ImageRef(),
        description="Third source image (optional).",
    )

    def _get_model(self) -> str:
        return "sora-2-pro-story-board"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if not self.prompt:
            raise ValueError("Prompt is required")
        if context is None:
            raise ValueError("Context is required for image upload")

        image_urls = []
        for img in [self.image1, self.image2, self.image3]:
            if img.is_set():
                url = await self._upload_image(context, img)
                image_urls.append(url)

        return {
            "prompt": self.prompt,
            "image_urls": image_urls,
            "n_frames": self.n_frames.value,
            "remove_watermark": self.remove_watermark,
        }


class Sora2TextToVideo(Sora2BaseNode):
    """Generate videos from text using Sora 2 Standard via Kie.ai.

    kie, sora, openai, video generation, ai, text-to-video, standard

    Sora 2 Standard generates quality videos from text descriptions
    with efficient processing and good visual quality.
    """
    _auto_save_asset: ClassVar[bool] = True

    _expose_as_tool: ClassVar[bool] = True

    @classmethod
    def get_title(cls) -> str:
        return "Sora 2 Text To Video"

    prompt: str = Field(
        default="A cinematic video with smooth motion, natural lighting, and high detail.",
        description="The text prompt describing the video.",
    )

    def _get_model(self) -> str:
        return "sora-2-text-to-video"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if not self.prompt:
            raise ValueError("Prompt is required")
        return {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "n_frames": self.n_frames.value,
            "remove_watermark": self.remove_watermark,
        }


class WanMultiShotTextToVideoPro(KieVideoBaseNode):
    """Generate videos from text using Alibaba's Wan 2.1 model via Kie.ai.

    kie, wan, alibaba, video generation, ai, text-to-video, multi-shot, 2.1

    Wan 2.1 Multi-Shot generates complex videos with multiple shots
    and scene transitions from text descriptions.
    """
    _auto_save_asset: ClassVar[bool] = True

    _expose_as_tool: ClassVar[bool] = True

    @classmethod
    def get_title(cls) -> str:
        return "Wan 2.1 Multi-Shot Text To Video"

    prompt: str = Field(
        default="A cinematic video with smooth motion, natural lighting, and high detail.",
        description="The text prompt describing the video.",
    )

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
        if not self.prompt:
            raise ValueError("Prompt is required")
        return {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "resolution": self.resolution.value,
            "duration": self.duration.value,
            "remove_watermark": self.remove_watermark,
        }


class Wan26TextToVideo(KieVideoBaseNode):
    """Generate videos from text using Alibaba's Wan 2.6 model via Kie.ai.

    kie, wan, alibaba, video generation, ai, text-to-video, 2.6

    Wan 2.6 generates high-quality videos from text descriptions
    with advanced motion and visual fidelity.
    """
    _auto_save_asset: ClassVar[bool] = True

    _expose_as_tool: ClassVar[bool] = True

    @classmethod
    def get_title(cls) -> str:
        return "Wan 2.6 Text To Video"

    prompt: str = Field(
        default="A cinematic video with smooth motion, natural lighting, and high detail.",
        description="The text prompt describing the video.",
    )

    class Duration(str, Enum):
        D5 = "5s"
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
        if not self.prompt:
            raise ValueError("Prompt is required")
        return {
            "prompt": self.prompt,
            "duration": self.duration.value,
            "resolution": self.resolution.value,
        }


class Wan26ImageToVideo(KieVideoBaseNode):
    """Generate videos from images using Alibaba's Wan 2.6 model via Kie.ai."""
    _auto_save_asset: ClassVar[bool] = True

    @classmethod
    def get_title(cls) -> str:
        return "Wan 2.6 Image To Video"

    _expose_as_tool: ClassVar[bool] = True

    prompt: str = Field(
        default="A cinematic video with smooth motion, natural lighting, and high detail.",
        description="The text prompt describing the video.",
    )

    image1: ImageRef = Field(
        default=ImageRef(),
        description="First source image for the video generation.",
    )

    image2: ImageRef = Field(
        default=ImageRef(),
        description="Second source image (optional).",
    )

    image3: ImageRef = Field(
        default=ImageRef(),
        description="Third source image (optional).",
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
        if not self.prompt:
            raise ValueError("Prompt is required")
        if context is None:
            raise ValueError("Context is required for image upload")

        image_urls = []
        for img in [self.image1, self.image2, self.image3]:
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
    """Generate videos from videos using Alibaba's Wan 2.6 model via Kie.ai.

    kie, wan, alibaba, video generation, ai, video-to-video, 2.6

    Wan 2.6 transforms and enhances existing videos with AI-powered
    editing and style transfer capabilities.
    """

    _expose_as_tool: ClassVar[bool] = True

    @classmethod
    def get_title(cls) -> str:
        return "Wan 2.6 Video To Video"

    prompt: str = Field(
        default="A cinematic video with smooth motion, natural lighting, and high detail.",
        description="The text prompt describing the changes.",
    )

    video1: VideoRef = Field(
        default=VideoRef(),
        description="First source video for the video-to-video task.",
    )

    video2: VideoRef = Field(
        default=VideoRef(),
        description="Second source video (optional).",
    )

    video3: VideoRef = Field(
        default=VideoRef(),
        description="Third source video (optional).",
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
        if not self.prompt:
            raise ValueError("Prompt is required")
        if context is None:
            raise ValueError("Context is required for video upload")

        video_urls = []
        for vid in [self.video1, self.video2, self.video3]:
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

    video: VideoRef = Field(
        default=VideoRef(),
        description="The video to upscale.",
    )

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
    _auto_save_asset: ClassVar[bool] = True

    _expose_as_tool: ClassVar[bool] = True

    prompt: str = Field(
        default="A cinematic video with smooth motion, natural lighting, and high detail.",
        description="Optional text guide for the video generation.",
    )

    image: ImageRef = Field(
        default=ImageRef(),
        description="The source image.",
    )

    audio: AudioRef = Field(
        default=AudioRef(),
        description="The source audio track.",
    )

    class Resolution(str, Enum):
        R480P = "480p"

    resolution: Resolution = Field(
        default=Resolution.R480P,
        description="Video resolution.",
    )

    def _get_model(self) -> str:
        return "infinitalk/from-audio"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if not self.prompt:
            raise ValueError("Prompt is required")
        if context is None:
            raise ValueError("Context is required for media upload")

        if not self.image.is_set():
            raise ValueError("Image is required")

        if not self.audio.is_set():
            raise ValueError("Audio is required")

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


class Veo31BaseNode(KieVideoBaseNode):
    """Base class for Google Veo 3.1 video generation nodes via Kie.ai.

    kie, google, veo, veo3, veo3.1, video generation, ai, text-to-video, image-to-video

    Veo 3.1 offers native 9:16 vertical video support, multilingual prompt processing,
    and significant cost savings (25% of Google's direct API pricing).
    """

    _poll_interval: float = 8.0
    _max_poll_attempts: int = 240

    class Model(str, Enum):
        VEO3 = "veo3"
        VEO3_FAST = "veo3_fast"

    model: Model = Field(
        default=Model.VEO3_FAST,
        description="The model to use for video generation.",
    )

    class AspectRatio(str, Enum):
        RATIO_16_9 = "16:9"
        RATIO_9_16 = "9:16"

    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9,
        description="Video aspect ratio.",
    )

    call_back_url: str = Field(
        default="",
        description="Optional callback URL for task completion.",
    )

    def _get_headers(self, api_key: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    async def _get_submit_payload(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        payload = await self._get_input_params(context)
        if self.call_back_url:
            payload["callBackUrl"] = self.call_back_url
        return payload

    def _get_success_flag(self, status_response: dict[str, Any]) -> int | None:
        success_flag = status_response.get("data", {}).get("successFlag")
        try:
            return int(success_flag)
        except (TypeError, ValueError):
            return None

    def _is_task_complete(self, status_response: dict[str, Any]) -> bool:
        return self._get_success_flag(status_response) == 1

    def _is_task_failed(self, status_response: dict[str, Any]) -> bool:
        return self._get_success_flag(status_response) in {2, 3}

    def _get_error_message(self, status_response: dict[str, Any]) -> str:
        return status_response.get("msg") or "Unknown error occurred"

    async def _submit_task(
        self,
        session: aiohttp.ClientSession,
        api_key: str,
        context: ProcessingContext | None = None,
    ) -> dict[str, Any]:
        url = f"{KIE_API_BASE_URL}/api/v1/veo/generate"
        payload = await self._get_submit_payload(context)
        headers = self._get_headers(api_key)
        log.info(f"Submitting Veo task to {url} with payload: {payload}")
        async with session.post(url, json=payload, headers=headers) as response:
            response_data = await response.json()
            if "code" in response_data:
                self._check_response_status(response_data)

            if response.status != 200:
                raise ValueError(
                    f"Failed to submit task: {response.status} - {response_data}"
                )
            return response_data

    async def _poll_status(
        self, session: aiohttp.ClientSession, api_key: str, task_id: str
    ) -> dict[str, Any]:
        url = f"{KIE_API_BASE_URL}/api/v1/veo/record-info?taskId={task_id}"
        headers = self._get_headers(api_key)

        for attempt in range(self._max_poll_attempts):
            log.debug(
                f"Polling Veo task status (attempt {attempt + 1}/{self._max_poll_attempts})"
            )
            async with session.get(url, headers=headers) as response:
                status_data = await response.json()
                print(f"Veo poll response: {status_data}")
                if "code" in status_data:
                    self._check_response_status(status_data)

                if self._is_task_complete(status_data):
                    log.debug("Veo task completed successfully")
                    return status_data

                if self._is_task_failed(status_data):
                    error_msg = self._get_error_message(status_data)
                    raise ValueError(f"Veo task failed: {error_msg}")

            await asyncio.sleep(self._poll_interval)

        raise TimeoutError(
            f"Veo task did not complete within {self._max_poll_attempts * self._poll_interval} seconds"
        )

    def _parse_result_urls(self, result_urls: Any) -> list[str]:
        import json

        if not result_urls:
            return []

        if isinstance(result_urls, list):
            return [url for url in result_urls if isinstance(url, str)]

        if isinstance(result_urls, str):
            try:
                parsed = json.loads(result_urls)
            except json.JSONDecodeError:
                return [result_urls]
            if isinstance(parsed, list):
                return [url for url in parsed if isinstance(url, str)]
            if isinstance(parsed, str):
                return [parsed]

        raise ValueError(f"Unexpected resultUrls format: {result_urls}")

    async def _download_result(  # type: ignore[override]
        self, session: aiohttp.ClientSession, api_key: str, status_data: dict[str, Any]
    ) -> bytes:
        if not self._is_task_complete(status_data):
            raise ValueError("Veo task is not complete yet")

        log.debug(f"Veo status_data: {status_data}")
        data = status_data.get("data", {})
        result_urls = self._parse_result_urls(
            data.get("resultUrls")
            or data.get("response", {}).get("resultUrls")
            or data.get("response", {}).get("originUrls")
        )
        log.debug(f"Veo resultUrls: {result_urls}")

        if not result_urls:
            raise ValueError("No resultUrls in response")

        result_url = result_urls[0]
        log.debug(f"Downloading result from {result_url}")

        async with session.get(result_url) as video_response:
            if video_response.status != 200:
                raise ValueError(f"Failed to download result from URL: {result_url}")
            return await video_response.read()

    async def _execute_task(self, context: ProcessingContext) -> tuple[bytes, str]:
        """Execute the full task workflow: submit, poll, download."""
        api_key = await self._get_api_key(context)

        async with aiohttp.ClientSession() as session:
            submit_response = await self._submit_task(session, api_key, context)
            task_id = self._extract_task_id(submit_response)
            log.info(f"Task submitted with ID: {task_id}")

            status_data = await self._poll_status(session, api_key, task_id)

            return await self._download_result(session, api_key, status_data), task_id


class Veo31TextToVideo(Veo31BaseNode):
    """Generate videos from text using Google's Veo 3.1 via Kie.ai.

    kie, google, veo, veo3, veo3.1, video generation, ai, text-to-video

    Veo 3.1 offers native 9:16 vertical video support, multilingual prompt processing,
    and significant cost savings (25% of Google's direct API pricing).
    """
    _auto_save_asset: ClassVar[bool] = True

    _expose_as_tool: ClassVar[bool] = True

    prompt: str = Field(
        default="A cinematic video with smooth motion, natural lighting, and high detail.",
        description="The text prompt describing the video.",
    )

    def _get_model(self) -> str:
        return "veo3/text-to-video"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if not self.prompt:
            raise ValueError("Prompt is required")
        return {
            "model": self.model.value,
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
        }


class RunwayBaseNode(KieVideoBaseNode):
    """Base class for Runway video generation nodes via Kie.ai.

    kie, runway, gen-3, video generation, ai

    Uses the dedicated Runway API endpoints:
    - POST /api/v1/runway/generate for task submission
    - GET /api/v1/runway/record-detail for polling
    """

    _poll_interval: float = 5.0
    _max_poll_attempts: int = 180

    @classmethod
    def is_visible(cls) -> bool:
        return cls is not RunwayBaseNode

    def _get_headers(self, api_key: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    async def _get_submit_payload(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        """Get the payload for Runway API - uses flat parameters, not nested input."""
        return await self._get_input_params(context)

    async def _submit_task(
        self,
        session: aiohttp.ClientSession,
        api_key: str,
        context: ProcessingContext | None = None,
    ) -> dict[str, Any]:
        """Submit a task using the dedicated Runway endpoint."""
        url = f"{KIE_API_BASE_URL}/api/v1/runway/generate"
        payload = await self._get_submit_payload(context)
        headers = self._get_headers(api_key)
        log.info(f"Submitting Runway task to {url} with payload: {payload}")
        async with session.post(url, json=payload, headers=headers) as response:
            response_data = await response.json()
            if "code" in response_data:
                self._check_response_status(response_data)

            if response.status != 200:
                raise ValueError(
                    f"Failed to submit Runway task: {response.status} - {response_data}"
                )
            return response_data

    def _is_task_complete(self, status_response: dict[str, Any]) -> bool:
        """Check if Runway task is complete."""
        state = status_response.get("data", {}).get("state", "")
        return state == "success"

    def _is_task_failed(self, status_response: dict[str, Any]) -> bool:
        """Check if Runway task has failed."""
        state = status_response.get("data", {}).get("state", "")
        return state == "fail"

    def _get_error_message(self, status_response: dict[str, Any]) -> str:
        """Extract error message from Runway response."""
        data = status_response.get("data", {})
        return data.get("failMsg") or status_response.get("msg") or "Unknown error occurred"

    async def _poll_status(
        self, session: aiohttp.ClientSession, api_key: str, task_id: str
    ) -> dict[str, Any]:
        """Poll for Runway task completion using dedicated endpoint."""
        url = f"{KIE_API_BASE_URL}/api/v1/runway/record-detail?taskId={task_id}"
        headers = self._get_headers(api_key)

        for attempt in range(self._max_poll_attempts):
            log.debug(
                f"Polling Runway task status (attempt {attempt + 1}/{self._max_poll_attempts})"
            )
            async with session.get(url, headers=headers) as response:
                status_data = await response.json()
                log.debug(f"Runway poll response: {status_data}")

                if "code" in status_data:
                    self._check_response_status(status_data)

                if self._is_task_complete(status_data):
                    log.debug("Runway task completed successfully")
                    return status_data

                if self._is_task_failed(status_data):
                    error_msg = self._get_error_message(status_data)
                    raise ValueError(f"Runway task failed: {error_msg}")

            await asyncio.sleep(self._poll_interval)

        raise TimeoutError(
            f"Runway task did not complete within {self._max_poll_attempts * self._poll_interval} seconds"
        )

    async def _download_result(  # type: ignore[override]
        self, session: aiohttp.ClientSession, api_key: str, status_data: dict[str, Any]
    ) -> bytes:
        """Download video from Runway result."""
        if not self._is_task_complete(status_data):
            raise ValueError("Runway task is not complete yet")

        data = status_data.get("data", {})
        video_info = data.get("videoInfo", {})
        video_url = video_info.get("videoUrl")

        if not video_url:
            raise ValueError(f"No videoUrl in Runway response: {status_data}")

        log.debug(f"Downloading Runway result from {video_url}")
        async with session.get(video_url) as video_response:
            if video_response.status != 200:
                raise ValueError(f"Failed to download video from URL: {video_url}")
            return await video_response.read()

    async def _execute_task(self, context: ProcessingContext) -> tuple[bytes, str]:
        """Execute the full Runway task workflow: submit, poll, download."""
        api_key = await self._get_api_key(context)

        async with aiohttp.ClientSession() as session:
            submit_response = await self._submit_task(session, api_key, context)
            task_id = self._extract_task_id(submit_response)
            log.info(f"Runway task submitted with ID: {task_id}")

            status_data = await self._poll_status(session, api_key, task_id)

            return await self._download_result(session, api_key, status_data), task_id


class RunwayGen3AlphaTextToVideo(RunwayBaseNode):
    """Generate videos from text using Runway's Gen-3 Alpha model via Kie.ai.

    kie, runway, gen-3, gen3alpha, video generation, ai, text-to-video

    Runway Gen-3 Alpha produces high-quality videos from text descriptions
    with advanced motion and temporal consistency.
    """
    _auto_save_asset: ClassVar[bool] = True

    _expose_as_tool: ClassVar[bool] = True

    @classmethod
    def get_title(cls) -> str:
        return "Runway Gen-3 Alpha Text To Video"

    prompt: str = Field(
        default="A cinematic video with smooth motion, natural lighting, and high detail.",
        description="The text prompt describing the video.",
        max_length=1800,
    )

    class AspectRatio(str, Enum):
        V16_9 = "16:9"
        V4_3 = "4:3"
        V1_1 = "1:1"
        V3_4 = "3:4"
        V9_16 = "9:16"

    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.V16_9,
        description="The aspect ratio of the generated video. Required for text-to-video generation.",
    )

    class Duration(Enum):
        D5 = 5
        D10 = 10

    duration: Duration = Field(
        default=Duration.D5,
        description="Video duration in seconds. If 10-second video is selected, 1080p resolution cannot be used.",
    )

    class Quality(str, Enum):
        R720P = "720p"
        R1080P = "1080p"

    quality: Quality = Field(
        default=Quality.R720P,
        description="Video resolution. If 1080p is selected, 10-second video cannot be generated.",
    )

    water_mark: str = Field(
        default="",
        description="Video watermark text content. An empty string indicates no watermark.",
    )

    call_back_url: str = Field(
        default="",
        description="Optional callback URL to receive task completion updates.",
    )

    def _get_model(self) -> str:
        return "runway/gen-3-alpha-text-to-video"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if not self.prompt:
            raise ValueError("Prompt is required")
        if self.duration == self.Duration.D10 and self.quality == self.Quality.R1080P:
            raise ValueError("10-second video cannot be generated with 1080p resolution")
        return {
            "prompt": self.prompt,
            "aspectRatio": self.aspect_ratio.value,
            "duration": self.duration.value,
            "quality": self.quality.value,
            "waterMark": self.water_mark,
            "callBackUrl": self.call_back_url,
        }


class RunwayGen3AlphaImageToVideo(RunwayBaseNode):
    """Generate videos from images using Runway's Gen-3 Alpha model via Kie.ai.

    kie, runway, gen-3, gen3alpha, video generation, ai, image-to-video

    Runway Gen-3 Alpha transforms static images into dynamic videos
    with realistic motion and temporal consistency.
    """
    _auto_save_asset: ClassVar[bool] = True

    _expose_as_tool: ClassVar[bool] = True

    @classmethod
    def get_title(cls) -> str:
        return "Runway Gen-3 Alpha Image To Video"

    image: ImageRef = Field(
        default=ImageRef(),
        description="Reference image to base the video on.",
    )

    prompt: str = Field(
        default="A cinematic video with smooth motion, natural lighting, and high detail.",
        description="Optional text to guide the video generation. Maximum length is 1800 characters.",
        max_length=1800,
    )

    class Duration(Enum):
        D5 = 5
        D10 = 10

    duration: Duration = Field(
        default=Duration.D5,
        description="Video duration in seconds. If 10-second video is selected, 1080p resolution cannot be used.",
    )

    class Quality(str, Enum):
        R720P = "720p"
        R1080P = "1080p"

    quality: Quality = Field(
        default=Quality.R720P,
        description="Video resolution. If 1080p is selected, 10-second video cannot be generated.",
    )

    water_mark: str = Field(
        default="",
        description="Video watermark text content. An empty string indicates no watermark.",
    )

    call_back_url: str = Field(
        default="",
        description="Optional callback URL to receive task completion updates.",
    )

    def _get_model(self) -> str:
        return "runway/gen-3-alpha-image-to-video"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if not self.image.is_set():
            raise ValueError("Image is required")
        if context is None:
            raise ValueError("Context is required for image upload")
        if self.duration == self.Duration.D10 and self.quality == self.Quality.R1080P:
            raise ValueError("10-second video cannot be generated with 1080p resolution")
        image_url = await self._upload_image(context, self.image)
        payload: dict[str, Any] = {
            "imageUrl": image_url,
            "duration": self.duration.value,
            "quality": self.quality.value,
            "waterMark": self.water_mark,
            "callBackUrl": self.call_back_url,
        }
        if self.prompt:
            payload["prompt"] = self.prompt
        return payload


class RunwayGen3AlphaExtendVideo(RunwayBaseNode):
    """Extend videos using Runway's Gen-3 Alpha model via Kie.ai.

    kie, runway, gen-3, gen3alpha, video generation, ai, video-extension

    Runway Gen-3 Alpha can extend existing videos with additional generated content.
    """

    _expose_as_tool: ClassVar[bool] = True

    @classmethod
    def get_title(cls) -> str:
        return "Runway Gen-3 Alpha Extend Video"

    video_url: str = Field(
        default="",
        description="The source video URL to extend.",
    )

    prompt: str = Field(
        default="Continue the motion naturally with smooth transitions.",
        description="Text prompt to guide the video extension. Maximum length is 1800 characters.",
        max_length=1800,
    )

    class Duration(Enum):
        D5 = 5
        D10 = 10

    duration: Duration = Field(
        default=Duration.D5,
        description="Duration to extend the video by in seconds. If 10-second extension is selected, 1080p resolution cannot be used.",
    )

    class Quality(str, Enum):
        R720P = "720p"
        R1080P = "1080p"

    quality: Quality = Field(
        default=Quality.R720P,
        description="Video resolution. If 1080p is selected, 10-second extension cannot be generated.",
    )

    water_mark: str = Field(
        default="",
        description="Video watermark text content. An empty string indicates no watermark.",
    )

    call_back_url: str = Field(
        default="",
        description="Optional callback URL to receive task completion updates.",
    )

    def _get_model(self) -> str:
        return "runway/gen-3-alpha-extend-video"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if not self.video_url:
            raise ValueError("video_url is required")
        if self.duration == self.Duration.D10 and self.quality == self.Quality.R1080P:
            raise ValueError("10-second extension cannot be generated with 1080p resolution")
        return {
            "video_url": self.video_url,
            "prompt": self.prompt,
            "duration": self.duration.value,
            "quality": self.quality.value,
            "waterMark": self.water_mark,
            "callBackUrl": self.call_back_url,
        }


class RunwayAlephVideo(RunwayBaseNode):
    """Generate videos using Runway's Aleph model via Kie.ai.

    kie, runway, aleph, video generation, ai, text-to-video

    Aleph is Runway's advanced video generation model offering
    high-quality output with sophisticated motion handling.
    """
    _auto_save_asset: ClassVar[bool] = True

    _expose_as_tool: ClassVar[bool] = True

    @classmethod
    def get_title(cls) -> str:
        return "Runway Aleph Video"

    prompt: str = Field(
        default="A cinematic video with smooth motion, natural lighting, and high detail.",
        description="The text prompt describing the video.",
        max_length=1800,
    )

    class AspectRatio(str, Enum):
        V16_9 = "16:9"
        V9_16 = "9:16"
        V1_1 = "1:1"

    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.V16_9,
        description="The aspect ratio of the generated video. Required for text-to-video generation.",
    )

    class Duration(Enum):
        D5 = 5
        D10 = 10

    duration: Duration = Field(
        default=Duration.D5,
        description="Video duration in seconds. If 10-second video is selected, 1080p resolution cannot be used.",
    )

    class Quality(str, Enum):
        R720P = "720p"
        R1080P = "1080p"

    quality: Quality = Field(
        default=Quality.R720P,
        description="Video resolution. If 1080p is selected, 10-second video cannot be generated.",
    )

    water_mark: str = Field(
        default="",
        description="Video watermark text content. An empty string indicates no watermark.",
    )

    call_back_url: str = Field(
        default="",
        description="Optional callback URL to receive task completion updates.",
    )

    def _get_model(self) -> str:
        return "runway/generate-aleph-video"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if not self.prompt:
            raise ValueError("Prompt is required")
        if self.duration == self.Duration.D10 and self.quality == self.Quality.R1080P:
            raise ValueError("10-second video cannot be generated with 1080p resolution")
        return {
            "prompt": self.prompt,
            "aspectRatio": self.aspect_ratio.value,
            "duration": self.duration.value,
            "quality": self.quality.value,
            "waterMark": self.water_mark,
            "callBackUrl": self.call_back_url,
        }


class LumaModifyVideo(KieVideoBaseNode):
    """Modify and enhance videos using Luma's API via Kie.ai.

    kie, luma, video modification, ai, video-editing

    Luma's video modification API allows for sophisticated video editing
    and enhancement capabilities.
    """

    _expose_as_tool: ClassVar[bool] = True
    _poll_interval: float = 5.0
    _max_poll_attempts: int = 180

    @classmethod
    def get_title(cls) -> str:
        return "Luma Modify Video"

    video: VideoRef = Field(
        default=VideoRef(),
        description="The source video to modify.",
    )

    prompt: str = Field(
        default="Enhance the video quality and add smooth motion.",
        description="Text prompt describing the modifications to make.",
    )

    class AspectRatio(str, Enum):
        RATIO_16_9 = "16:9"
        RATIO_9_16 = "9:16"
        RATIO_1_1 = "1:1"

    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9,
        description="The aspect ratio of the output video.",
    )

    class Duration(str, Enum):
        D5 = "5"
        D10 = "10"

    duration: Duration = Field(
        default=Duration.D5,
        description="Duration of the modified video segment.",
    )

    def _get_model(self) -> str:
        return "luma/generate-luma-modify-video"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if context is None:
            raise ValueError("Context is required for video upload")
        if not self.video.is_set():
            raise ValueError("Video is required")

        video_url = await self._upload_video(context, self.video)
        return {
            "video_url": video_url,
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "duration": self.duration.value,
        }


class Veo31ImageToVideo(Veo31BaseNode):
    """Generate videos from images using Google's Veo 3.1 model via Kie.ai.

    kie, google, veo, veo3, veo3.1, video generation, ai, image-to-video, i2v

    Supports single image (image comes alive) or two images (first and last frames transition).
    For two images, the first image serves as the video's first frame and the second as the last frame.
    """
    _auto_save_asset: ClassVar[bool] = True

    _expose_as_tool: ClassVar[bool] = True

    @classmethod
    def get_title(cls) -> str:
        return "Veo 3.1 Image To Video"

    prompt: str = Field(
        default="A cinematic video with smooth motion, natural lighting, and high detail.",
        description="Optional text prompt describing how the image should come alive.",
    )

    image1: ImageRef = Field(
        default=ImageRef(),
        description="First source image. Required. Serves as the video's first frame.",
    )

    image2: ImageRef = Field(
        default=ImageRef(),
        description="Second source image (optional). If provided, serves as the video's last frame.",
    )

    def _get_model(self) -> str:
        return self.model.value

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if context is None:
            raise ValueError("Context is required for image upload")

        if not self.image1.is_set():
            raise ValueError("At least one image is required")

        image_urls = []
        image_url1 = await self._upload_image(context, self.image1)
        image_urls.append(image_url1)

        if self.image2.is_set():
            image_url2 = await self._upload_image(context, self.image2)
            image_urls.append(image_url2)

        payload: dict[str, Any] = {
            "imageUrls": image_urls,
            "model": self.model.value,
            "aspect_ratio": self.aspect_ratio.value,
        }

        if self.prompt:
            payload["prompt"] = self.prompt

        return payload


class Veo31ReferenceToVideo(Veo31BaseNode):
    """Generate videos from reference images using Google's Veo 3.1 Fast model via Kie.ai.

    kie, google, veo, veo3, veo3.1, video generation, ai, reference-to-video, material-to-video

    Material-to-video generation based on reference images. Only supports veo3_fast model
    and requires 1-3 reference images.
    """
    _auto_save_asset: ClassVar[bool] = True

    _expose_as_tool: ClassVar[bool] = True

    @classmethod
    def get_title(cls) -> str:
        return "Veo 3.1 Reference To Video"

    prompt: str = Field(
        default="A cinematic video with smooth motion, natural lighting, and high detail.",
        description="Text prompt describing the desired video content.",
    )

    image1: ImageRef = Field(
        default=ImageRef(),
        description="First reference image. Required. Minimum 1, maximum 3 images.",
    )

    image2: ImageRef = Field(
        default=ImageRef(),
        description="Second reference image (optional).",
    )

    image3: ImageRef = Field(
        default=ImageRef(),
        description="Third reference image (optional).",
    )

    def _get_model(self) -> str:
        return "veo3_fast"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if context is None:
            raise ValueError("Context is required for image upload")

        if not self.image1.is_set():
            raise ValueError("At least one reference image is required")

        image_urls = []
        for img in [self.image1, self.image2, self.image3]:
            if img.is_set():
                url = await self._upload_image(context, img)
                image_urls.append(url)

        if len(image_urls) > 3:
            raise ValueError("Maximum 3 reference images allowed")

        payload: dict[str, Any] = {
            "prompt": self.prompt,
            "imageUrls": image_urls,
            "model": "veo3_fast",
            "aspect_ratio": self.aspect_ratio.value,
        }

        return payload


class KlingMotionControl(KieVideoBaseNode):
    """Generate videos with motion control using Kuaishou's Kling 2.6 model via Kie.ai.

    kie, kling, kuaishou, video generation, ai, motion-control, character-animation, 2.6

    Kling Motion Control generates videos where character actions are guided by a reference video,
    while the visual appearance is based on a reference image. Perfect for character animation
    and motion transfer tasks.
    """

    _expose_as_tool: ClassVar[bool] = True

    @classmethod
    def get_title(cls) -> str:
        return "Kling 2.6 Motion Control"

    prompt: str = Field(
        default="The cartoon character is dancing.",
        description="A text description of the desired output. Maximum 2500 characters.",
        max_length=2500,
    )

    image: ImageRef = Field(
        default=ImageRef(),
        description="Reference image. The characters, backgrounds, and other elements in the generated video are based on this image. Supports .jpg/.jpeg/.png, max 10MB, size needs to be greater than 300px, aspect ratio 2:5 to 5:2.",
    )

    video: VideoRef = Field(
        default=VideoRef(),
        description="Reference video. The character actions in the generated video will be consistent with this reference video. Supports .mp4/.mov, max 100MB, 3-30 seconds duration depending on character_orientation.",
    )

    class CharacterOrientation(str, Enum):
        IMAGE = "image"
        VIDEO = "video"

    character_orientation: CharacterOrientation = Field(
        default=CharacterOrientation.VIDEO,
        description="Generate the orientation of the characters in the video. 'image': same orientation as the person in the picture (max 10s video). 'video': consistent with the orientation of the characters in the video (max 30s video).",
    )

    class Mode(str, Enum):
        R720P = "720p"
        R1080P = "1080p"

    mode: Mode = Field(
        default=Mode.R720P,
        description="Output resolution mode. Use '720p' for 720p or '1080p' for 1080p.",
    )

    def _get_model(self) -> str:
        return "kling-2.6/motion-control"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if not self.image.is_set():
            raise ValueError("Reference image is required")
        if not self.video.is_set():
            raise ValueError("Reference video is required")
        if context is None:
            raise ValueError("Context is required for media upload")

        image_url, video_url = await asyncio.gather(
            self._upload_image(context, self.image),
            self._upload_video(context, self.video),
        )

        return {
            "prompt": self.prompt,
            "input_urls": [image_url],
            "video_urls": [video_url],
            "character_orientation": self.character_orientation.value,
            "mode": self.mode.value,
        }


class Kling21TextToVideo(KieVideoBaseNode):
    """Generate videos from text using Kuaishou's Kling 2.1 model via Kie.ai.

    kie, kling, kuaishou, video generation, ai, text-to-video, 2.1

    Kling 2.1 powers cutting-edge video generation with hyper-realistic motion,
    advanced physics, and high-resolution outputs up to 1080p.

    Use cases:
    - Generate high-quality videos from text descriptions
    - Create dynamic, professional-grade video content
    - Produce videos with realistic motion and physics
    """
    _auto_save_asset: ClassVar[bool] = True

    _expose_as_tool: ClassVar[bool] = True

    @classmethod
    def get_title(cls) -> str:
        return "Kling 2.1 Text To Video"

    prompt: str = Field(
        default="A cinematic video with smooth motion, natural lighting, and high detail.",
        description="The text prompt describing the video.",
    )

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
        R720P = "720P"
        R1080P = "1080P"

    resolution: Resolution = Field(
        default=Resolution.R720P,
        description="Video resolution.",
    )

    class Mode(str, Enum):
        STANDARD = "standard"
        PRO = "pro"

    mode: Mode = Field(
        default=Mode.STANDARD,
        description="Generation mode: standard or pro for higher quality.",
    )

    seed: int = Field(
        default=-1,
        description="Random seed for reproducible results. Use -1 for random seed.",
    )

    def _get_model(self) -> str:
        return "kling/v2-1-text-to-video"

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
            "mode": self.mode.value,
            "seed": self.seed,
        }


class Kling21ImageToVideo(KieVideoBaseNode):
    """Generate videos from images using Kuaishou's Kling 2.1 model via Kie.ai.

    kie, kling, kuaishou, video generation, ai, image-to-video, 2.1

    Kling 2.1 transforms static images into dynamic videos with hyper-realistic
    motion and advanced physics simulation.

    Use cases:
    - Animate static images with realistic motion
    - Create videos from photos and artwork
    - Produce dynamic content from still images
    """
    _auto_save_asset: ClassVar[bool] = True

    _expose_as_tool: ClassVar[bool] = True

    @classmethod
    def get_title(cls) -> str:
        return "Kling 2.1 Image To Video"

    prompt: str = Field(
        default="A cinematic video with smooth motion, natural lighting, and high detail.",
        description="Optional text prompt to guide the video generation.",
    )

    image1: ImageRef = Field(
        default=ImageRef(),
        description="First source image for the video generation.",
    )

    image2: ImageRef = Field(
        default=ImageRef(),
        description="Second source image (optional).",
    )

    image3: ImageRef = Field(
        default=ImageRef(),
        description="Third source image (optional).",
    )

    sound: bool = Field(
        default=False,
        description="Whether to generate sound for the video.",
    )

    duration: int = Field(
        default=5,
        description="Video duration in seconds.",
    )

    class Mode(str, Enum):
        STANDARD = "standard"
        PRO = "pro"

    mode: Mode = Field(
        default=Mode.STANDARD,
        description="Generation mode: standard or pro for higher quality.",
    )

    def _get_model(self) -> str:
        return "kling/v2-1-image-to-video"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if not self.prompt:
            raise ValueError("Prompt is required")
        if context is None:
            raise ValueError("Context is required for image upload")

        image_urls = []
        for img in [self.image1, self.image2, self.image3]:
            if img.is_set():
                url = await self._upload_image(context, img)
                image_urls.append(url)

        return {
            "prompt": self.prompt,
            "image_urls": image_urls,
            "sound": self.sound,
            "duration": str(self.duration),
            "mode": self.mode.value,
        }


class Wan25TextToVideo(KieVideoBaseNode):
    """Generate videos from text using Alibaba's Wan 2.5 model via Kie.ai.

    kie, wan, alibaba, video generation, ai, text-to-video, 2.5

    Wan 2.5 is designed for cinematic AI video generation with native audio
    synchronization including dialogue, ambient sound, and background music.

    Use cases:
    - Generate cinematic videos from text descriptions
    - Create videos with synchronized audio
    - Produce content for social media and advertising
    """
    _auto_save_asset: ClassVar[bool] = True

    _expose_as_tool: ClassVar[bool] = True

    @classmethod
    def get_title(cls) -> str:
        return "Wan 2.5 Text To Video"

    prompt: str = Field(
        default="A cinematic video with smooth motion, natural lighting, and high detail.",
        description="The text prompt describing the video.",
    )

    class Duration(str, Enum):
        D5 = "5s"
        D10 = "10s"

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

    class AspectRatio(str, Enum):
        V16_9 = "16:9"
        V9_16 = "9:16"
        V1_1 = "1:1"

    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.V16_9,
        description="The aspect ratio of the generated video.",
    )

    def _get_model(self) -> str:
        return "wan/2-5-text-to-video"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if not self.prompt:
            raise ValueError("Prompt is required")
        return {
            "prompt": self.prompt,
            "duration": self.duration.value,
            "resolution": self.resolution.value,
            "aspect_ratio": self.aspect_ratio.value,
        }


class Wan25ImageToVideo(KieVideoBaseNode):
    """Generate videos from images using Alibaba's Wan 2.5 model via Kie.ai.

    kie, wan, alibaba, video generation, ai, image-to-video, 2.5

    Wan 2.5 transforms images into cinematic videos with native audio
    synchronization.

    Use cases:
    - Animate static images with cinematic quality
    - Create videos from photos with audio
    - Produce dynamic content from still images
    """
    _auto_save_asset: ClassVar[bool] = True

    _expose_as_tool: ClassVar[bool] = True

    @classmethod
    def get_title(cls) -> str:
        return "Wan 2.5 Image To Video"

    prompt: str = Field(
        default="A cinematic video with smooth motion, natural lighting, and high detail.",
        description="The text prompt describing the video.",
    )

    image1: ImageRef = Field(
        default=ImageRef(),
        description="First source image for the video generation.",
    )

    image2: ImageRef = Field(
        default=ImageRef(),
        description="Second source image (optional).",
    )

    image3: ImageRef = Field(
        default=ImageRef(),
        description="Third source image (optional).",
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
        return "wan/2-5-image-to-video"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if not self.prompt:
            raise ValueError("Prompt is required")
        if context is None:
            raise ValueError("Context is required for image upload")

        image_urls = []
        for img in [self.image1, self.image2, self.image3]:
            if img.is_set():
                url = await self._upload_image(context, img)
                image_urls.append(url)

        return {
            "prompt": self.prompt,
            "image_urls": image_urls,
            "duration": self.duration.value,
            "resolution": self.resolution.value,
        }


class WanAnimate(KieVideoBaseNode):
    """Generate character animation videos using Alibaba's Wan 2.2 Animate via Kie.ai.

    kie, wan, alibaba, video generation, ai, image-to-video, animate, character

    Wan 2.2 Animate generates realistic character videos with motion, expressions,
    and lighting from static images.

    Use cases:
    - Animate character images with realistic motion
    - Create character-driven video content
    - Produce animated videos from portraits or character art
    """
    _auto_save_asset: ClassVar[bool] = True

    _expose_as_tool: ClassVar[bool] = True

    @classmethod
    def get_title(cls) -> str:
        return "Wan 2.2 Animate"

    prompt: str = Field(
        default="The character is moving naturally with realistic expressions.",
        description="The text prompt describing the character animation.",
    )

    image: ImageRef = Field(
        default=ImageRef(),
        description="Character image to animate.",
    )

    class Duration(str, Enum):
        D3 = "3"
        D5 = "5"

    duration: Duration = Field(
        default=Duration.D3,
        description="The duration of the video in seconds.",
    )

    class Resolution(str, Enum):
        R720P = "720p"
        R1080P = "1080p"

    resolution: Resolution = Field(
        default=Resolution.R720P,
        description="The resolution of the video.",
    )

    def _get_model(self) -> str:
        return "wan/animate"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if not self.image.is_set():
            raise ValueError("Image is required")
        if context is None:
            raise ValueError("Context is required for image upload")

        image_url = await self._upload_image(context, self.image)
        return {
            "prompt": self.prompt,
            "image_url": image_url,
            "duration": self.duration.value,
            "resolution": self.resolution.value,
        }


class WanSpeechToVideo(KieVideoBaseNode):
    """Generate videos from speech using Alibaba's Wan 2.2 A14B Turbo via Kie.ai.

    kie, wan, alibaba, video generation, ai, speech-to-video, lip-sync

    Wan 2.2 A14B Turbo Speech to Video turns static images and audio clips
    into dynamic, expressive videos.

    Use cases:
    - Create talking head videos from images and audio
    - Generate lip-synced content for presentations
    - Produce dynamic videos from voice recordings
    """
    _auto_save_asset: ClassVar[bool] = True

    _expose_as_tool: ClassVar[bool] = True

    @classmethod
    def get_title(cls) -> str:
        return "Wan 2.2 Speech To Video"

    image: ImageRef = Field(
        default=ImageRef(),
        description="Character/face image to animate.",
    )

    audio: AudioRef = Field(
        default=AudioRef(),
        description="Audio file for speech/lip-sync.",
    )

    class Resolution(str, Enum):
        R720P = "720p"
        R1080P = "1080p"

    resolution: Resolution = Field(
        default=Resolution.R720P,
        description="The resolution of the video.",
    )

    def _get_model(self) -> str:
        return "wan/speech-to-video-turbo"

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

        return {
            "image_url": image_url,
            "audio_url": audio_url,
            "resolution": self.resolution.value,
        }


class Wan22TextToVideo(KieVideoBaseNode):
    """Generate videos from text using Alibaba's Wan 2.2 A14B Turbo via Kie.ai.

    kie, wan, alibaba, video generation, ai, text-to-video, 2.2

    Wan 2.2 A14B Turbo delivers smooth 720p@24fps clips with cinematic quality,
    stable motion, and consistent visual style.

    Use cases:
    - Generate high-quality videos from text
    - Create content for diverse creative uses
    - Produce consistent video clips with stable motion
    """
    _auto_save_asset: ClassVar[bool] = True

    _expose_as_tool: ClassVar[bool] = True

    @classmethod
    def get_title(cls) -> str:
        return "Wan 2.2 Text To Video"

    prompt: str = Field(
        default="A cinematic video with smooth motion, natural lighting, and high detail.",
        description="The text prompt describing the video.",
    )

    class Duration(str, Enum):
        D3 = "3"
        D5 = "5"

    duration: Duration = Field(
        default=Duration.D3,
        description="The duration of the video in seconds.",
    )

    class Resolution(str, Enum):
        R720P = "720p"

    resolution: Resolution = Field(
        default=Resolution.R720P,
        description="The resolution of the video.",
    )

    class AspectRatio(str, Enum):
        V16_9 = "16:9"
        V9_16 = "9:16"
        V1_1 = "1:1"

    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.V16_9,
        description="The aspect ratio of the generated video.",
    )

    def _get_model(self) -> str:
        return "wan/v2-2-text-to-video"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if not self.prompt:
            raise ValueError("Prompt is required")
        return {
            "prompt": self.prompt,
            "duration": self.duration.value,
            "resolution": self.resolution.value,
            "aspect_ratio": self.aspect_ratio.value,
        }


class Wan22ImageToVideo(KieVideoBaseNode):
    """Generate videos from images using Alibaba's Wan 2.2 A14B Turbo via Kie.ai.

    kie, wan, alibaba, video generation, ai, image-to-video, 2.2

    Wan 2.2 A14B Turbo transforms images into smooth video clips with
    cinematic quality and stable motion.

    Use cases:
    - Animate static images with smooth motion
    - Create videos from photos or artwork
    - Produce consistent video content from images
    """
    _auto_save_asset: ClassVar[bool] = True

    _expose_as_tool: ClassVar[bool] = True

    @classmethod
    def get_title(cls) -> str:
        return "Wan 2.2 Image To Video"

    prompt: str = Field(
        default="A cinematic video with smooth motion, natural lighting, and high detail.",
        description="The text prompt describing the video.",
    )

    image: ImageRef = Field(
        default=ImageRef(),
        description="Source image for the video generation.",
    )

    class Duration(str, Enum):
        D3 = "3"
        D5 = "5"

    duration: Duration = Field(
        default=Duration.D3,
        description="The duration of the video in seconds.",
    )

    class Resolution(str, Enum):
        R720P = "720p"

    resolution: Resolution = Field(
        default=Resolution.R720P,
        description="The resolution of the video.",
    )

    def _get_model(self) -> str:
        return "wan/v2-2-image-to-video"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if not self.prompt:
            raise ValueError("Prompt is required")
        if not self.image.is_set():
            raise ValueError("Image is required")
        if context is None:
            raise ValueError("Context is required for image upload")

        image_url = await self._upload_image(context, self.image)
        return {
            "prompt": self.prompt,
            "image_url": image_url,
            "duration": self.duration.value,
            "resolution": self.resolution.value,
        }


class Hailuo02TextToVideo(KieVideoBaseNode):
    """Generate videos from text using Minimax's Hailuo 02 model via Kie.ai.

    kie, hailuo, minimax, video generation, ai, text-to-video

    Hailuo 02 is Minimax's advanced AI video generation model that produces
    short, cinematic clips with realistic motion and physics simulation.

    Use cases:
    - Generate cinematic video clips from text
    - Create videos with realistic motion and physics
    - Produce high-quality content up to 1080P
    """
    _auto_save_asset: ClassVar[bool] = True

    _expose_as_tool: ClassVar[bool] = True

    @classmethod
    def get_title(cls) -> str:
        return "Hailuo 02 Text To Video"

    prompt: str = Field(
        default="A cinematic video with smooth motion, natural lighting, and high detail.",
        description="The text prompt describing the video.",
    )

    class Duration(str, Enum):
        D5 = "5"
        D10 = "10"

    duration: Duration = Field(
        default=Duration.D5,
        description="The duration of the video in seconds.",
    )

    class Resolution(str, Enum):
        R720P = "720p"
        R1080P = "1080p"

    resolution: Resolution = Field(
        default=Resolution.R720P,
        description="The resolution of the video.",
    )

    class AspectRatio(str, Enum):
        V16_9 = "16:9"
        V9_16 = "9:16"
        V1_1 = "1:1"

    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.V16_9,
        description="The aspect ratio of the generated video.",
    )

    def _get_model(self) -> str:
        return "hailuo/02-text-to-video"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if not self.prompt:
            raise ValueError("Prompt is required")
        return {
            "prompt": self.prompt,
            "duration": self.duration.value,
            "resolution": self.resolution.value,
            "aspect_ratio": self.aspect_ratio.value,
        }


class Hailuo02ImageToVideo(KieVideoBaseNode):
    """Generate videos from images using Minimax's Hailuo 02 model via Kie.ai.

    kie, hailuo, minimax, video generation, ai, image-to-video

    Hailuo 02 transforms images into cinematic clips with realistic motion
    and physics simulation.

    Use cases:
    - Animate images with realistic motion
    - Create videos from photos with physics simulation
    - Produce dynamic content from still images
    """
    _auto_save_asset: ClassVar[bool] = True

    _expose_as_tool: ClassVar[bool] = True

    @classmethod
    def get_title(cls) -> str:
        return "Hailuo 02 Image To Video"

    prompt: str = Field(
        default="A cinematic video with smooth motion, natural lighting, and high detail.",
        description="The text prompt describing the video.",
    )

    image: ImageRef = Field(
        default=ImageRef(),
        description="Source image for the video generation.",
    )

    class Duration(str, Enum):
        D5 = "5"
        D10 = "10"

    duration: Duration = Field(
        default=Duration.D5,
        description="The duration of the video in seconds.",
    )

    class Resolution(str, Enum):
        R720P = "720p"
        R1080P = "1080p"

    resolution: Resolution = Field(
        default=Resolution.R720P,
        description="The resolution of the video.",
    )

    def _get_model(self) -> str:
        return "hailuo/02-image-to-video"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if not self.prompt:
            raise ValueError("Prompt is required")
        if not self.image.is_set():
            raise ValueError("Image is required")
        if context is None:
            raise ValueError("Context is required for image upload")

        image_url = await self._upload_image(context, self.image)
        return {
            "prompt": self.prompt,
            "image_url": image_url,
            "duration": self.duration.value,
            "resolution": self.resolution.value,
        }


class Sora2WatermarkRemover(KieVideoBaseNode):
    """Remove watermarks from Sora 2 videos using Kie.ai.

    kie, sora, openai, video editing, watermark removal

    Sora 2 Watermark Remover uses AI detection and motion tracking to remove
    dynamic watermarks from Sora 2 videos while keeping frames smooth and natural.

    Use cases:
    - Remove watermarks from generated videos
    - Clean up video content for final output
    - Prepare videos for professional use
    """
    _auto_save_asset: ClassVar[bool] = True

    _expose_as_tool: ClassVar[bool] = True

    @classmethod
    def get_title(cls) -> str:
        return "Sora 2 Watermark Remover"

    video: VideoRef = Field(
        default=VideoRef(),
        description="Video to remove watermark from. Must be publicly accessible.",
    )

    def _get_model(self) -> str:
        return "sora-2-watermark-remover"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if not self.video.is_set():
            raise ValueError("Video is required")
        if context is None:
            raise ValueError("Context is required for video upload")

        video_url = await self._upload_video(context, self.video)
        return {
            "video_url": video_url,
        }
