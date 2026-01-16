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
    """Generate videos from text using Google's Veo 3.1 model via Kie.ai.

    kie, google, veo, veo3, veo3.1, video generation, ai, text-to-video, t2v
    """

    _expose_as_tool: ClassVar[bool] = True

    @classmethod
    def get_title(cls) -> str:
        return "Veo 3.1 Text To Video"

    prompt: str = Field(
        default="A cinematic video with smooth motion, natural lighting, and high detail.",
        description="The text prompt describing the desired video content.",
    )

    def _get_model(self) -> str:
        return self.model.value

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if not self.prompt:
            raise ValueError("Prompt is required")

        return {
            "prompt": self.prompt,
            "model": self.model.value,
            "aspect_ratio": self.aspect_ratio.value,
        }


class Veo31ImageToVideo(Veo31BaseNode):
    """Generate videos from images using Google's Veo 3.1 model via Kie.ai.

    kie, google, veo, veo3, veo3.1, video generation, ai, image-to-video, i2v

    Supports single image (image comes alive) or two images (first and last frames transition).
    For two images, the first image serves as the video's first frame and the second as the last frame.
    """

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


class RunwayBaseNode(KieVideoBaseNode):
    """Base class for Runway video generation nodes via Kie.ai.

    kie, runway, video generation, ai, text-to-video, image-to-video

    Runway API uses different endpoints and response formats than other providers.
    """

    _poll_interval: float = 10.0
    _max_poll_attempts: int = 60

    def _get_headers(self, api_key: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def _is_task_complete(self, status_response: dict[str, Any]) -> bool:
        state = status_response.get("data", {}).get("state", "")
        return state == "success"

    def _is_task_failed(self, status_response: dict[str, Any]) -> bool:
        state = status_response.get("data", {}).get("state", "")
        return state == "fail"

    def _get_error_message(self, status_response: dict[str, Any]) -> str:
        return (
            status_response.get("data", {}).get("failMsg") or "Unknown error occurred"
        )

    def _extract_task_id(self, response: dict[str, Any]) -> str:
        if "data" in response and isinstance(response["data"], dict):
            if "taskId" in response["data"]:
                return response["data"]["taskId"]
        raise ValueError(f"Could not extract taskId from response: {response}")

    async def _download_result(
        self, session: aiohttp.ClientSession, api_key: str, task_id: str
    ) -> bytes:
        url = f"{KIE_API_BASE_URL}/api/v1/runway/record-detail?taskId={task_id}"
        headers = self._get_headers(api_key)

        async with session.get(url, headers=headers) as response:
            if response.status != 200:
                response_text = await response.text()
                raise ValueError(
                    f"Failed to get result: {response.status} - {response_text}"
                )

            status_data = await response.json()
            if "code" in status_data:
                self._check_response_status(status_data)

            video_info = status_data.get("data", {}).get("videoInfo", {})
            video_url = video_info.get("videoUrl")

            if not video_url:
                raise ValueError("No videoUrl in response")

            async with session.get(video_url) as video_response:
                if video_response.status != 200:
                    raise ValueError(f"Failed to download video from URL: {video_url}")
                return await video_response.read()


class RunwayTextToVideo(RunwayBaseNode):
    """Generate videos from text using Runway via Kie.ai.

    kie, runway, video generation, ai, text-to-video

    Runway generates high-quality videos from text descriptions with
    support for multiple durations, qualities, and aspect ratios.

    Use cases:
    - Generate videos from text descriptions
    - Create short cinematic clips
    - Produce content for social media
    """

    _expose_as_tool: ClassVar[bool] = True

    @classmethod
    def get_title(cls) -> str:
        return "Runway Text To Video"

    prompt: str = Field(
        default="A cinematic video with smooth motion, natural lighting, and high detail.",
        description="The text prompt describing the video.",
    )

    class Duration(str, Enum):
        D5 = 5
        D10 = 10

    duration: Duration = Field(
        default=Duration.D5,
        description="Video duration in seconds.",
    )

    class Quality(str, Enum):
        P720 = "720p"
        P1080 = "1080p"

    quality: Quality = Field(
        default=Quality.P720,
        description="Video quality resolution.",
    )

    class AspectRatio(str, Enum):
        LANDSCAPE = "16:9"
        PORTRAIT = "9:16"
        SQUARE = "1:1"
        STANDARD_4_3 = "4:3"
        PORTRAIT_3_4 = "3:4"

    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.LANDSCAPE,
        description="The aspect ratio of the generated video.",
    )

    watermark: str = Field(
        default="",
        description="Optional watermark text to display.",
    )

    def _get_model(self) -> str:
        return "runway"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if not self.prompt:
            raise ValueError("Prompt is required")

        payload: dict[str, Any] = {
            "prompt": self.prompt,
            "duration": self.duration.value,
            "quality": self.quality.value,
            "aspectRatio": self.aspect_ratio.value,
        }

        if self.watermark:
            payload["waterMark"] = self.watermark

        return payload

    async def _execute_video_task(
        self, context: ProcessingContext
    ) -> tuple[bytes, str]:
        api_key = await self._get_api_key(context)

        async with aiohttp.ClientSession() as session:
            submit_response = await self._submit_task(session, api_key, context)
            task_id = self._extract_task_id(submit_response)
            log.info(f"Runway task submitted with ID: {task_id}")

            await self._poll_status(session, api_key, task_id)

            return await self._download_result(session, api_key, task_id), task_id


class RunwayImageToVideo(RunwayBaseNode):
    """Generate videos from images using Runway via Kie.ai.

    kie, runway, video generation, ai, image-to-video

    Runway animates static images into dynamic videos based on text prompts.

    Use cases:
    - Animate static images into videos
    - Create dynamic content from reference images
    - Add motion to artwork or photos
    """

    _expose_as_tool: ClassVar[bool] = True

    @classmethod
    def get_title(cls) -> str:
        return "Runway Image To Video"

    prompt: str = Field(
        default="A cinematic video with smooth motion, natural lighting, and high detail.",
        description="Text description of how the image should animate.",
    )

    image: ImageRef = Field(
        default=ImageRef(),
        description="The source image to animate.",
    )

    class Duration(str, Enum):
        D5 = 5
        D10 = 10

    duration: Duration = Field(
        default=Duration.D5,
        description="Video duration in seconds.",
    )

    class Quality(str, Enum):
        P720 = "720p"
        P1080 = "1080p"

    quality: Quality = Field(
        default=Quality.P720,
        description="Video quality resolution.",
    )

    class AspectRatio(str, Enum):
        LANDSCAPE = "16:9"
        PORTRAIT = "9:16"
        SQUARE = "1:1"
        STANDARD_4_3 = "4:3"
        PORTRAIT_3_4 = "3:4"

    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.LANDSCAPE,
        description="The aspect ratio of the generated video.",
    )

    watermark: str = Field(
        default="",
        description="Optional watermark text to display.",
    )

    def _get_model(self) -> str:
        return "runway"

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
            "imageUrl": image_url,
            "duration": self.duration.value,
            "quality": self.quality.value,
            "aspectRatio": self.aspect_ratio.value,
        }

        if self.watermark:
            payload["waterMark"] = self.watermark

        return payload

    async def _execute_video_task(
        self, context: ProcessingContext
    ) -> tuple[bytes, str]:
        api_key = await self._get_api_key(context)

        async with aiohttp.ClientSession() as session:
            submit_response = await self._submit_task(session, api_key, context)
            task_id = self._extract_task_id(submit_response)
            log.info(f"Runway task submitted with ID: {task_id}")

            await self._poll_status(session, api_key, task_id)

            return await self._download_result(session, api_key, task_id), task_id


class RunwayExtendVideo(RunwayBaseNode):
    """Extend existing videos using Runway via Kie.ai.

    kie, runway, video generation, ai, video-extension, extend

    Runway extends existing videos to create longer sequences while
    maintaining visual consistency.

    Use cases:
    - Extend short videos into longer content
    - Create seamless video continuations
    - Build longer narratives from clips
    """

    _expose_as_tool: ClassVar[bool] = True

    @classmethod
    def get_title(cls) -> str:
        return "Runway Extend Video"

    prompt: str = Field(
        default="Continue the motion naturally.",
        description="Text description of how the video should continue.",
    )

    task_id: str = Field(
        default="",
        description="The task ID of the video to extend.",
    )

    class Quality(str, Enum):
        P720 = "720p"
        P1080 = "1080p"

    quality: Quality = Field(
        default=Quality.P720,
        description="Video quality resolution.",
    )

    watermark: str = Field(
        default="",
        description="Optional watermark text to display.",
    )

    def _get_model(self) -> str:
        return "runway"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if not self.prompt:
            raise ValueError("Prompt is required")
        if not self.task_id:
            raise ValueError("Task ID is required")

        payload: dict[str, Any] = {
            "taskId": self.task_id,
            "prompt": self.prompt,
            "quality": self.quality.value,
        }

        if self.watermark:
            payload["waterMark"] = self.watermark

        return payload

    async def _submit_task(
        self,
        session: aiohttp.ClientSession,
        api_key: str,
        context: ProcessingContext | None = None,
    ) -> dict[str, Any]:
        url = f"{KIE_API_BASE_URL}/api/v1/runway/extend"
        payload = await self._get_input_params(context)
        headers = self._get_headers(api_key)
        log.info(f"Submitting Runway extend task to {url} with payload: {payload}")

        async with session.post(url, json=payload, headers=headers) as response:
            response_data = await response.json()
            if "code" in response_data:
                self._check_response_status(response_data)

            if response.status != 200:
                raise ValueError(
                    f"Failed to submit task: {response.status} - {response_data}"
                )
            return response_data


class LumaModifyVideo(KieVideoBaseNode):
    """Modify and transform videos using Luma via Kie.ai.

    kie, luma, video modification, ai, video-to-video, transform

    Luma Modify transforms existing videos based on text prompts,
    enabling creative video editing and style transfer.

    Use cases:
    - Transform video style and content
    - Apply AI-powered edits to existing videos
    - Create video variations with different aesthetics
    """

    _expose_as_tool: ClassVar[bool] = True
    _poll_interval: float = 10.0
    _max_poll_attempts: int = 90

    prompt: str = Field(
        default="",
        description="Text description of the desired video transformation.",
    )

    video: VideoRef = Field(
        default=VideoRef(),
        description="The source video to modify.",
    )

    call_back_url: str = Field(
        default="",
        description="Optional callback URL for task completion.",
    )

    watermark: str = Field(
        default="",
        description="Optional watermark identifier.",
    )

    def _get_model(self) -> str:
        return "luma/modify"

    def _get_headers(self, api_key: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def _is_task_complete(self, status_response: dict[str, Any]) -> bool:
        success_flag = status_response.get("data", {}).get("successFlag", -1)
        return success_flag == 1

    def _is_task_failed(self, status_response: dict[str, Any]) -> bool:
        success_flag = status_response.get("data", {}).get("successFlag", -1)
        return success_flag in {2, 3}

    def _get_error_message(self, status_response: dict[str, Any]) -> str:
        return (
            status_response.get("data", {}).get("errorMessage")
            or "Unknown error occurred"
        )

    async def _submit_task(
        self,
        session: aiohttp.ClientSession,
        api_key: str,
        context: ProcessingContext | None = None,
    ) -> dict[str, Any]:
        url = f"{KIE_API_BASE_URL}/api/v1/modify/generate"
        payload = await self._get_input_params(context)
        headers = self._get_headers(api_key)
        log.info(f"Submitting Luma modify task to {url} with payload: {payload}")

        async with session.post(url, json=payload, headers=headers) as response:
            response_data = await response.json()
            if "code" in response_data:
                self._check_response_status(response_data)

            if response.status != 200:
                raise ValueError(
                    f"Failed to submit task: {response.status} - {response_data}"
                )
            return response_data

    async def _download_result(
        self, session: aiohttp.ClientSession, api_key: str, task_id: str
    ) -> bytes:
        url = f"{KIE_API_BASE_URL}/api/v1/modify/record-info?taskId={task_id}"
        headers = self._get_headers(api_key)

        async with session.get(url, headers=headers) as response:
            if response.status != 200:
                response_text = await response.text()
                raise ValueError(
                    f"Failed to get result: {response.status} - {response_text}"
                )

            status_data = await response.json()
            if "code" in status_data:
                self._check_response_status(status_data)

            response_body = status_data.get("data", {}).get("response", {})
            result_urls = response_body.get("resultUrls", [])

            if not result_urls:
                raise ValueError("No resultUrls in response")

            video_url = result_urls[0]
            log.debug(f"Downloading result from {video_url}")

            async with session.get(video_url) as video_response:
                if video_response.status != 200:
                    raise ValueError(f"Failed to download video from URL: {video_url}")
                return await video_response.read()

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if not self.prompt:
            raise ValueError("Prompt is required")
        if not self.video.is_set():
            raise ValueError("Video is required")
        if context is None:
            raise ValueError("Context is required for video upload")

        video_url = await self._upload_video(context, self.video)

        payload: dict[str, Any] = {
            "prompt": self.prompt,
            "videoUrl": video_url,
        }

        if self.call_back_url:
            payload["callBackUrl"] = self.call_back_url
        if self.watermark:
            payload["watermark"] = self.watermark

        return payload

    async def _execute_video_task(
        self, context: ProcessingContext
    ) -> tuple[bytes, str]:
        api_key = await self._get_api_key(context)

        async with aiohttp.ClientSession() as session:
            submit_response = await self._submit_task(session, api_key, context)
            task_id = self._extract_task_id(submit_response)
            log.info(f"Luma modify task submitted with ID: {task_id}")

            await self._poll_status(session, api_key, task_id)

            return await self._download_result(session, api_key, task_id), task_id
