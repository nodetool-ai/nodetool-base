"""Kie.ai image generation nodes.

This module provides nodes for generating images using Kie.ai's various APIs:
- 4O Image API (GPT-4o powered image generation)
- Seedream 4.5 (ByteDance's image generation model)
- Z-Image Turbo (Alibaba's photorealistic image generation)
- Nano Banana (Google Gemini 2.5 image model)
- Nano Banana Pro (Google Gemini 3.0 image model)
- Flux Kontext (Black Forest Labs advanced image generation)
- Flux Pro (Black Forest Labs text-to-image)
- Topaz Image Upscaler (AI image upscaling and enhancement)
- Grok Imagine (xAI multimodal image generation)
"""

import asyncio
from abc import abstractmethod
from enum import Enum
from typing import Any, ClassVar

import aiohttp
from pydantic import Field

from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import ImageRef
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext

log = get_logger(__name__)

# Kie.ai API base URL
KIE_API_BASE_URL = "https://api.kie.ai"


class KieBaseNode(BaseNode):
    """Base class for Kie.ai API nodes with polling logic for task completion.

    kie, ai, image generation, api

    This base class encapsulates the common pattern for Kie.ai APIs:
    1. Submit a task (POST request)
    2. Poll for task completion (GET request)
    3. Download the result
    """

    # Polling configuration
    poll_interval: float = Field(
        default=2.0,
        description="Interval in seconds between status checks.",
        ge=0.5,
        le=30.0,
    )
    max_poll_attempts: int = Field(
        default=120,
        description="Maximum number of polling attempts before timeout.",
        ge=1,
        le=600,
    )

    @classmethod
    def is_visible(cls) -> bool:
        return cls is not KieBaseNode

    async def _get_api_key(self, context: ProcessingContext) -> str:
        """Get the Kie.ai API key from secrets."""
        api_key = await context.get_secret("KIE_API_KEY")
        if not api_key:
            raise ValueError(
                "KIE_API_KEY secret is not configured. "
                "Please set the KIE_API_KEY secret in your configuration."
            )
        return api_key

    def _get_headers(self, api_key: str) -> dict[str, str]:
        """Get common headers for Kie.ai API requests."""
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    @abstractmethod
    def _get_base_endpoint(self) -> str:
        """Get the base API endpoint for this model.

        Subclasses only need to implement this single method.
        Other endpoints are derived from it:
        - Submit: {base_endpoint}/generate
        - Status: {base_endpoint}/{task_id}
        - Download: {base_endpoint}/{task_id}/download
        """
        ...

    def _get_submit_endpoint(self) -> str:
        """Get the API endpoint for submitting tasks."""
        return f"{self._get_base_endpoint()}/generate"

    def _get_status_endpoint(self, task_id: str) -> str:
        """Get the API endpoint for checking task status."""
        return f"{self._get_base_endpoint()}/{task_id}"

    def _get_download_endpoint(self, task_id: str) -> str:
        """Get the API endpoint for downloading results."""
        return f"{self._get_base_endpoint()}/{task_id}/download"

    @abstractmethod
    def _get_submit_payload(self) -> dict[str, Any]:
        """Get the payload for the task submission request."""
        ...

    def _extract_task_id(self, response: dict[str, Any]) -> str:
        """Extract the task ID from the submission response."""
        # Most Kie.ai APIs return task_id in the response
        if "task_id" in response:
            return response["task_id"]
        if "data" in response and isinstance(response["data"], dict):
            if "task_id" in response["data"]:
                return response["data"]["task_id"]
        raise ValueError(f"Could not extract task_id from response: {response}")

    def _is_task_complete(self, status_response: dict[str, Any]) -> bool:
        """Check if the task is complete based on the status response."""
        status = status_response.get("status") or status_response.get("data", {}).get(
            "status"
        )
        return status in ("completed", "success", "done", "finished")

    def _is_task_failed(self, status_response: dict[str, Any]) -> bool:
        """Check if the task has failed based on the status response."""
        status = status_response.get("status") or status_response.get("data", {}).get(
            "status"
        )
        return status in ("failed", "error", "cancelled")

    def _get_error_message(self, status_response: dict[str, Any]) -> str:
        """Extract error message from a failed task response."""
        return (
            status_response.get("message")
            or status_response.get("error")
            or status_response.get("data", {}).get("message")
            or "Unknown error occurred"
        )

    async def _submit_task(
        self, session: aiohttp.ClientSession, api_key: str
    ) -> dict[str, Any]:
        """Submit a task to the Kie.ai API."""
        url = f"{KIE_API_BASE_URL}{self._get_submit_endpoint()}"
        payload = self._get_submit_payload()
        headers = self._get_headers(api_key)

        log.debug(f"Submitting task to {url}")
        async with session.post(url, json=payload, headers=headers) as response:
            response_data = await response.json()
            if response.status != 200:
                raise ValueError(
                    f"Failed to submit task: {response.status} - {response_data}"
                )
            return response_data

    async def _poll_status(
        self, session: aiohttp.ClientSession, api_key: str, task_id: str
    ) -> dict[str, Any]:
        """Poll for task completion status."""
        url = f"{KIE_API_BASE_URL}{self._get_status_endpoint(task_id)}"
        headers = self._get_headers(api_key)

        for attempt in range(self.max_poll_attempts):
            log.debug(
                f"Polling task status (attempt {attempt + 1}/{self.max_poll_attempts})"
            )
            async with session.get(url, headers=headers) as response:
                status_data = await response.json()

                if self._is_task_complete(status_data):
                    log.debug("Task completed successfully")
                    return status_data

                if self._is_task_failed(status_data):
                    error_msg = self._get_error_message(status_data)
                    raise ValueError(f"Task failed: {error_msg}")

            await asyncio.sleep(self.poll_interval)

        raise TimeoutError(
            f"Task did not complete within {self.max_poll_attempts * self.poll_interval} seconds"
        )

    async def _download_result(
        self, session: aiohttp.ClientSession, api_key: str, task_id: str
    ) -> bytes:
        """Download the result from the completed task."""
        url = f"{KIE_API_BASE_URL}{self._get_download_endpoint(task_id)}"
        headers = self._get_headers(api_key)

        log.debug(f"Downloading result from {url}")
        async with session.get(url, headers=headers) as response:
            if response.status != 200:
                response_text = await response.text()
                raise ValueError(
                    f"Failed to download result: {response.status} - {response_text}"
                )

            # Check if response is JSON with a URL
            content_type = response.headers.get("Content-Type", "")
            if "application/json" in content_type:
                data = await response.json()
                # Extract image URL from response
                image_url = (
                    data.get("url")
                    or data.get("image_url")
                    or data.get("data", {}).get("url")
                    or data.get("data", {}).get("image_url")
                )
                if image_url:
                    async with session.get(image_url) as img_response:
                        if img_response.status != 200:
                            raise ValueError(
                                f"Failed to download image from URL: {image_url}"
                            )
                        return await img_response.read()
                raise ValueError(f"Could not extract image URL from response: {data}")

            return await response.read()

    async def _execute_task(self, context: ProcessingContext) -> bytes:
        """Execute the full task workflow: submit, poll, download."""
        api_key = await self._get_api_key(context)

        async with aiohttp.ClientSession() as session:
            # Submit the task
            submit_response = await self._submit_task(session, api_key)
            task_id = self._extract_task_id(submit_response)
            log.info(f"Task submitted with ID: {task_id}")

            # Poll for completion
            await self._poll_status(session, api_key, task_id)

            # Download the result
            return await self._download_result(session, api_key, task_id)


class Generate4OImage(KieBaseNode):
    """Generate images using Kie.ai's 4O Image API (GPT-4o powered).

    kie, 4o, gpt4o, image generation, ai, text-to-image

    Use cases:
    - Generate high-quality images from text descriptions
    - Create illustrations and artwork
    - Generate product mockups
    - Create visual content for marketing
    """

    _expose_as_tool: ClassVar[bool] = True

    prompt: str = Field(
        default="",
        description="The text prompt describing the image to generate.",
    )

    class AspectRatio(str, Enum):
        SQUARE = "1:1"
        LANDSCAPE = "16:9"
        PORTRAIT = "9:16"
        WIDE = "21:9"
        TALL = "9:21"

    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.SQUARE,
        description="The aspect ratio of the generated image.",
    )

    def _get_base_endpoint(self) -> str:
        return "/v1/4o-images"

    def _get_submit_payload(self) -> dict[str, Any]:
        if not self.prompt:
            raise ValueError("Prompt cannot be empty")
        return {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
        }

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_bytes = await self._execute_task(context)
        return await context.image_from_bytes(image_bytes)


class SeedreamGenerate(KieBaseNode):
    """Generate images using ByteDance's Seedream 4.5 model via Kie.ai.

    kie, seedream, bytedance, image generation, ai, text-to-image, 4k

    Seedream 4.5 generates high-quality visuals up to 4K resolution with
    improved detail fidelity, multi-image blending, and sharp text/face rendering.

    Use cases:
    - Generate creative and artistic images
    - Create diverse visual content up to 4K
    - Generate illustrations with unique styles
    - Multi-image reference for consistency
    """

    _expose_as_tool: ClassVar[bool] = True

    prompt: str = Field(
        default="",
        description="The text prompt describing the image to generate.",
    )

    class AspectRatio(str, Enum):
        SQUARE = "1:1"
        LANDSCAPE = "16:9"
        PORTRAIT = "9:16"
        WIDE = "4:3"
        TALL = "3:4"

    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.SQUARE,
        description="The aspect ratio of the generated image.",
    )

    def _get_base_endpoint(self) -> str:
        return "/v1/market/seedream-4-5"

    def _get_submit_payload(self) -> dict[str, Any]:
        if not self.prompt:
            raise ValueError("Prompt cannot be empty")
        return {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
        }

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_bytes = await self._execute_task(context)
        return await context.image_from_bytes(image_bytes)


class ZImageGenerate(KieBaseNode):
    """Generate images using Alibaba's Z-Image Turbo model via Kie.ai.

    kie, z-image, zimage, alibaba, image generation, ai, text-to-image, photorealistic

    Z-Image Turbo produces realistic, detail-rich images with very low latency.
    It supports bilingual text (English/Chinese) in images with sharp text rendering.

    Use cases:
    - Generate high-quality photorealistic images quickly
    - Create images with embedded text (English/Chinese)
    - Generate detailed illustrations with low latency
    - Product visualizations
    """

    _expose_as_tool: ClassVar[bool] = True

    prompt: str = Field(
        default="",
        description="The text prompt describing the image to generate.",
    )

    class AspectRatio(str, Enum):
        SQUARE = "1:1"
        LANDSCAPE = "16:9"
        PORTRAIT = "9:16"
        WIDE = "4:3"
        TALL = "3:4"

    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.SQUARE,
        description="The aspect ratio of the generated image.",
    )

    def _get_base_endpoint(self) -> str:
        return "/v1/market/z-image"

    def _get_submit_payload(self) -> dict[str, Any]:
        if not self.prompt:
            raise ValueError("Prompt cannot be empty")
        return {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
        }

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_bytes = await self._execute_task(context)
        return await context.image_from_bytes(image_bytes)


class NanoBananaGenerate(KieBaseNode):
    """Generate images using Google's Nano Banana model (Gemini 2.5) via Kie.ai.

    kie, nano-banana, google, gemini, image generation, ai, text-to-image, fast

    Nano Banana is powered by Gemini 2.5 Flash Image and offers fast
    language-driven image generation/editing, focusing on speed and iteration efficiency.

    Use cases:
    - Generate images with efficient processing
    - Create visual content quickly
    - Generate images for rapid prototyping
    - Fast iteration on image concepts
    """

    _expose_as_tool: ClassVar[bool] = True

    prompt: str = Field(
        default="",
        description="The text prompt describing the image to generate.",
    )

    class AspectRatio(str, Enum):
        SQUARE = "1:1"
        LANDSCAPE = "16:9"
        PORTRAIT = "9:16"
        WIDE = "4:3"
        TALL = "3:4"

    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.SQUARE,
        description="The aspect ratio of the generated image.",
    )

    def _get_base_endpoint(self) -> str:
        return "/v1/market/google/nano-banana"

    def _get_submit_payload(self) -> dict[str, Any]:
        if not self.prompt:
            raise ValueError("Prompt cannot be empty")
        return {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
        }

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_bytes = await self._execute_task(context)
        return await context.image_from_bytes(image_bytes)


class FluxProTextToImage(KieBaseNode):
    """Generate images using Black Forest Labs' Flux Pro model via Kie.ai.

    kie, flux, flux-pro, black-forest-labs, image generation, ai, text-to-image

    Use cases:
    - Generate high-quality artistic images
    - Create professional visual content
    - Generate images with fine detail and artistic style
    """

    _expose_as_tool: ClassVar[bool] = True

    prompt: str = Field(
        default="",
        description="The text prompt describing the image to generate.",
    )

    class AspectRatio(str, Enum):
        SQUARE = "1:1"
        LANDSCAPE = "16:9"
        PORTRAIT = "9:16"
        WIDE = "4:3"
        TALL = "3:4"

    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.SQUARE,
        description="The aspect ratio of the generated image.",
    )

    steps: int = Field(
        default=25,
        description="Number of inference steps. Higher values may produce better quality but take longer.",
        ge=1,
        le=50,
    )

    guidance_scale: float = Field(
        default=7.5,
        description="Guidance scale for the generation. Higher values adhere more closely to the prompt.",
        ge=1.0,
        le=20.0,
    )

    def _get_base_endpoint(self) -> str:
        return "/v1/market/flux2/pro"

    def _get_submit_endpoint(self) -> str:
        """Override to use text-to-image instead of generate."""
        return f"{self._get_base_endpoint()}/text-to-image"

    def _get_submit_payload(self) -> dict[str, Any]:
        if not self.prompt:
            raise ValueError("Prompt cannot be empty")
        return {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "steps": self.steps,
            "guidance_scale": self.guidance_scale,
        }

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_bytes = await self._execute_task(context)
        return await context.image_from_bytes(image_bytes)


class NanoBananaProGenerate(KieBaseNode):
    """Generate images using Google's Nano Banana Pro model (Gemini 3.0) via Kie.ai.

    kie, nano-banana-pro, google, gemini, image generation, ai, text-to-image, 4k, high-fidelity

    Nano Banana Pro is based on Gemini 3.0 Pro Image and provides higher fidelity
    with sharper structure, 4K output, and better text rendering for photorealistic results.

    Use cases:
    - Generate high-fidelity photorealistic images
    - Create 4K resolution content
    - Generate images with sharp text rendering
    - Professional-grade visual content
    """

    _expose_as_tool: ClassVar[bool] = True

    prompt: str = Field(
        default="",
        description="The text prompt describing the image to generate.",
    )

    class AspectRatio(str, Enum):
        SQUARE = "1:1"
        LANDSCAPE = "16:9"
        PORTRAIT = "9:16"
        WIDE = "4:3"
        TALL = "3:4"

    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.SQUARE,
        description="The aspect ratio of the generated image.",
    )

    def _get_base_endpoint(self) -> str:
        return "/v1/market/google/nano-banana-pro"

    def _get_submit_payload(self) -> dict[str, Any]:
        if not self.prompt:
            raise ValueError("Prompt cannot be empty")
        return {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
        }

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_bytes = await self._execute_task(context)
        return await context.image_from_bytes(image_bytes)


class FluxKontextGenerate(KieBaseNode):
    """Generate images using Black Forest Labs' Flux Kontext model via Kie.ai.

    kie, flux, flux-kontext, black-forest-labs, image generation, ai, text-to-image, editing

    Flux Kontext supports Pro (speed-optimized) and Max (quality-focused) variants
    with features like multiple aspect ratios, safety controls, and async processing.

    Use cases:
    - Generate high-quality artistic images
    - Advanced image editing and generation
    - Create professional visual content
    - Generate images with fine detail and artistic style
    """

    _expose_as_tool: ClassVar[bool] = True

    prompt: str = Field(
        default="",
        description="The text prompt describing the image to generate.",
    )

    class AspectRatio(str, Enum):
        SQUARE = "1:1"
        LANDSCAPE = "16:9"
        PORTRAIT = "9:16"
        WIDE = "4:3"
        TALL = "3:4"

    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.SQUARE,
        description="The aspect ratio of the generated image.",
    )

    class Mode(str, Enum):
        PRO = "pro"
        MAX = "max"

    mode: Mode = Field(
        default=Mode.PRO,
        description="Generation mode: 'pro' for speed, 'max' for quality.",
    )

    def _get_base_endpoint(self) -> str:
        return "/v1/market/flux-kontext"

    def _get_submit_payload(self) -> dict[str, Any]:
        if not self.prompt:
            raise ValueError("Prompt cannot be empty")
        return {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "mode": self.mode.value,
        }

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_bytes = await self._execute_task(context)
        return await context.image_from_bytes(image_bytes)


class GrokImagineGenerate(KieBaseNode):
    """Generate images using xAI's Grok Imagine model via Kie.ai.

    kie, grok, xai, image generation, ai, text-to-image, multimodal

    Grok Imagine is a multimodal generative model that can generate images
    from text prompts with coherent motion and synchronized background audio for videos.

    Use cases:
    - Generate images from text descriptions
    - Create visual content with AI
    - Multimodal content generation
    """

    _expose_as_tool: ClassVar[bool] = True

    prompt: str = Field(
        default="",
        description="The text prompt describing the image to generate.",
    )

    class AspectRatio(str, Enum):
        SQUARE = "1:1"
        LANDSCAPE = "16:9"
        PORTRAIT = "9:16"
        WIDE = "4:3"
        TALL = "3:4"

    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.SQUARE,
        description="The aspect ratio of the generated image.",
    )

    def _get_base_endpoint(self) -> str:
        return "/v1/market/grok-imagine"

    def _get_submit_payload(self) -> dict[str, Any]:
        if not self.prompt:
            raise ValueError("Prompt cannot be empty")
        return {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
        }

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_bytes = await self._execute_task(context)
        return await context.image_from_bytes(image_bytes)


class TopazImageUpscale(KieBaseNode):
    """Upscale and enhance images using Topaz Labs AI via Kie.ai.

    kie, topaz, upscale, enhance, image, ai, super-resolution

    Leverages Topaz Labs' image super-resolution models to upscale and enhance images.
    Can enlarge images by 2x, 4x, etc., while unblurring and sharpening details.

    Use cases:
    - Upscale low-resolution images to high resolution
    - Enhance old or degraded photos
    - Improve AI-generated art quality
    - Prepare images for large format printing
    """

    _expose_as_tool: ClassVar[bool] = True

    image: ImageRef = Field(
        default=ImageRef(),
        description="The image to upscale.",
    )

    class ScaleFactor(str, Enum):
        X2 = "2"
        X4 = "4"

    scale_factor: ScaleFactor = Field(
        default=ScaleFactor.X2,
        description="The upscaling factor (2x or 4x).",
    )

    def _get_base_endpoint(self) -> str:
        return "/v1/market/topaz-image-upscale"

    def _get_submit_payload(self) -> dict[str, Any]:
        if not self.image.is_set():
            raise ValueError("Image is required")
        return {
            "image": self.image.to_dict(),
            "scale_factor": self.scale_factor.value,
        }

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_bytes = await self._execute_task(context)
        return await context.image_from_bytes(image_bytes)
