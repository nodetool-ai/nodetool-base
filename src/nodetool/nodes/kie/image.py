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
import os
import uuid
from urllib.parse import urlparse
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
KIE_FILE_UPLOAD_URL = "https://kieai.redpandaai.co/api/file-stream-upload"


class KieBaseNode(BaseNode):
    """Base class for Kie.ai API nodes with polling logic for task completion.

    kie, ai, image generation, api

    This base class encapsulates the common pattern for Kie.ai APIs:
    1. Submit a task (POST request)
    2. Poll for task completion (GET request)
    3. Download the result
    """

    # Polling configuration - to be implemented by subclasses
    _poll_interval: float
    _max_poll_attempts: int

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

    def _resolve_upload_filename(
        self,
        asset: Any,
        default_extension: str,
        file_obj: Any | None = None,
    ) -> str:
        filename = None
        file_name_attr = getattr(file_obj, "name", None) if file_obj else None
        if file_name_attr and not str(file_name_attr).startswith("<"):
            filename = os.path.basename(str(file_name_attr))

        if not filename:
            uri = getattr(asset, "uri", "") or ""
            if uri:
                parsed = urlparse(uri)
                base = os.path.basename(parsed.path)
                if base:
                    filename = base

        if not filename:
            filename = f"nodetool-{uuid.uuid4().hex}{default_extension}"

        root, ext = os.path.splitext(filename)
        if not ext and default_extension:
            filename = f"{root}{default_extension}"

        return filename

    async def _upload_asset(
        self,
        context: ProcessingContext,
        asset: Any,
        upload_path: str,
        default_extension: str,
    ) -> str:
        """Upload an asset to Kie.ai and return the download URL."""
        api_key = await self._get_api_key(context)
        file_obj = await context.asset_to_io(asset)
        filename = self._resolve_upload_filename(
            asset, default_extension=default_extension, file_obj=file_obj
        )
        form = aiohttp.FormData()
        form.add_field("file", file_obj, filename=filename)
        form.add_field("uploadPath", upload_path)
        form.add_field("fileName", filename)

        headers = {"Authorization": f"Bearer {api_key}"}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    KIE_FILE_UPLOAD_URL, data=form, headers=headers
                ) as response:
                    response_data = await response.json()
                    if response.status != 200 or not response_data.get("success"):
                        raise ValueError(
                            f"Failed to upload file: {response.status} - {response_data}"
                        )
        finally:
            close_fn = getattr(file_obj, "close", None)
            if callable(close_fn):
                close_fn()

        download_url = response_data.get("data", {}).get("downloadUrl")
        if not download_url:
            raise ValueError(f"No downloadUrl in upload response: {response_data}")
        return download_url

    async def _upload_image(
        self, context: ProcessingContext, image: ImageRef
    ) -> str:
        return await self._upload_asset(
            context, asset=image, upload_path="images/user-uploads", default_extension=".png"
        )

    async def _upload_audio(
        self, context: ProcessingContext, audio: Any
    ) -> str:
        return await self._upload_asset(
            context, asset=audio, upload_path="audio/user-uploads", default_extension=".mp3"
        )

    async def _upload_video(
        self, context: ProcessingContext, video: Any
    ) -> str:
        return await self._upload_asset(
            context, asset=video, upload_path="videos/user-uploads", default_extension=".mp4"
        )

    @abstractmethod
    def _get_model(self) -> str:
        """Get the model name for this node.

        Subclasses only need to implement this single method.
        The model name will be used in API requests to Kie.ai.
        """
        ...

    @abstractmethod
    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        """Get the input parameters for the API request.

        Subclasses should return a dict with model-specific input parameters.
        This will be nested under the 'input' field in the API payload.

        Args:
            context: Optional ProcessingContext for async operations like file upload.
        """
        ...

    async def _get_submit_payload(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        """Get the payload for the task submission request.

        Returns payload in the format:
        {
          "model": "model-name",
          "callBackUrl": "string (optional)",
          "input": {
            // model-specific parameters
          }
        }
        """
        return {
            "model": self._get_model(),
            "input": await self._get_input_params(context),
        }

    def _extract_task_id(self, response: dict[str, Any]) -> str:
        """Extract the task ID from the submission response.

        According to Kie.ai API:
        {
          "code": 200,
          "message": "success",
          "data": {
            "taskId": "task_12345678"
          }
        }
        """
        if "data" in response and isinstance(response["data"], dict):
            if "taskId" in response["data"]:
                return response["data"]["taskId"]
        raise ValueError(f"Could not extract taskId from response: {response}")

    def _is_task_complete(self, status_response: dict[str, Any]) -> bool:
        """Check if the task is complete based on the status response.

        According to Kie.ai API, state can be: "processing", "success", "failed", etc.
        """
        state = status_response.get("data", {}).get("state", "")
        return state == "success"

    def _is_task_failed(self, status_response: dict[str, Any]) -> bool:
        """Check if the task has failed based on the status response."""
        state = status_response.get("data", {}).get("state", "")
        return state == "failed"

    def _get_error_message(self, status_response: dict[str, Any]) -> str:
        """Extract error message from a failed task response."""
        data = status_response.get("data", {})
        return data.get("failMsg") or "Unknown error occurred"

    async def _submit_task(
        self,
        session: aiohttp.ClientSession,
        api_key: str,
        context: ProcessingContext | None = None,
    ) -> dict[str, Any]:
        """Submit a task to the Kie.ai API.

        Uses the unified /api/v1/jobs/createTask endpoint with format:
        {
          "model": "model-name",
          "callBackUrl": "string (optional)",
          "input": {
            // model-specific parameters
          }
        }
        """
        url = f"{KIE_API_BASE_URL}/api/v1/jobs/createTask"
        payload = await self._get_submit_payload(context)
        headers = self._get_headers(api_key)
        log.debug(f"Submitting task to {url}")
        log.debug(f"Payload: {payload}")
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
        """Poll for task completion status.

        Uses the /api/v1/jobs/recordInfo endpoint:
        GET /api/v1/jobs/recordInfo?taskId=task_12345678
        """
        url = f"{KIE_API_BASE_URL}/api/v1/jobs/recordInfo?taskId={task_id}"
        headers = self._get_headers(api_key)

        for attempt in range(self._max_poll_attempts):
            log.debug(
                f"Polling task status (attempt {attempt + 1}/{self._max_poll_attempts})"
            )
            async with session.get(url, headers=headers) as response:
                status_data = await response.json()

                if self._is_task_complete(status_data):
                    log.debug("Task completed successfully")
                    return status_data

                if self._is_task_failed(status_data):
                    error_msg = self._get_error_message(status_data)
                    raise ValueError(f"Task failed: {error_msg}")

            await asyncio.sleep(self._poll_interval)

        raise TimeoutError(
            f"Task did not complete within {self._max_poll_attempts * self._poll_interval} seconds"
        )

    async def _download_result(
        self, session: aiohttp.ClientSession, api_key: str, task_id: str
    ) -> bytes:
        """Download the result from the completed task.

        Extracts result URL from the status response:
        {
          "data": {
            "resultJson": "{\"resultUrls\":[\"https://example.com/image.jpg\"]}"
          }
        }
        """
        # First, get the final status with result
        url = f"{KIE_API_BASE_URL}/api/v1/jobs/recordInfo?taskId={task_id}"
        headers = self._get_headers(api_key)

        async with session.get(url, headers=headers) as response:
            if response.status != 200:
                response_text = await response.text()
                raise ValueError(
                    f"Failed to get result: {response.status} - {response_text}"
                )

            status_data = await response.json()
            result_json_str = status_data.get("data", {}).get("resultJson", "")

            if not result_json_str:
                raise ValueError("No resultJson in response")

            # Parse the result JSON string
            import json

            result_data = json.loads(result_json_str)
            result_urls = result_data.get("resultUrls", [])

            if not result_urls:
                raise ValueError("No resultUrls in resultJson")

            # Download from the first URL
            result_url = result_urls[0]
            log.debug(f"Downloading result from {result_url}")

            async with session.get(result_url) as img_response:
                if img_response.status != 200:
                    raise ValueError(
                        f"Failed to download result from URL: {result_url}"
                    )
                return await img_response.read()

    async def _execute_task(self, context: ProcessingContext) -> bytes:
        """Execute the full task workflow: submit, poll, download."""
        api_key = await self._get_api_key(context)

        async with aiohttp.ClientSession() as session:
            submit_response = await self._submit_task(session, api_key, context)
            task_id = self._extract_task_id(submit_response)
            log.info(f"Task submitted with ID: {task_id}")

            await self._poll_status(session, api_key, task_id)

            return await self._download_result(session, api_key, task_id)


class Flux2ProTextToImage(KieBaseNode):
    """Generate images using Black Forest Labs' Flux 2 Pro Text-to-Image model via Kie.ai.

    kie, flux, flux-2, flux-pro, black-forest-labs, image generation, ai, text-to-image

    Use cases:
    - Generate high-quality artistic images from text
    - Create professional visual content
    - Generate images with fine detail and artistic style
    """

    _expose_as_tool: ClassVar[bool] = True
    _poll_interval: float = 1.5
    _max_poll_attempts: int = 60

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

    class Resolution(str, Enum):
        RES_1K = "1K"
        RES_2K = "2K"

    resolution: Resolution = Field(
        default=Resolution.RES_1K,
        description="Output image resolution.",
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

    def _get_model(self) -> str:
        return "flux-2/pro-text-to-image"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if not self.prompt:
            raise ValueError("Prompt cannot be empty")
        return {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "resolution": self.resolution.value,
            "steps": self.steps,
            "guidance_scale": self.guidance_scale,
        }

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_bytes = await self._execute_task(context)
        return await context.image_from_bytes(image_bytes)


class Flux2ProImageToImage(KieBaseNode):
    """Generate images using Black Forest Labs' Flux 2 Pro Image-to-Image model via Kie.ai.

    kie, flux, flux-2, flux-pro, black-forest-labs, image generation, ai, image-to-image

    Use cases:
    - Transform existing images with text prompts
    - Apply artistic styles to photos
    - Create variations of existing images
    - Enhance and modify images
    """

    _expose_as_tool: ClassVar[bool] = True
    _poll_interval: float = 1.5
    _max_poll_attempts: int = 60

    prompt: str = Field(
        default="",
        description="The text prompt describing how to transform the image.",
    )

    image: ImageRef = Field(
        default=ImageRef(),
        description="The source image to transform.",
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

    class Resolution(str, Enum):
        RES_1K = "1K"
        RES_2K = "2K"

    resolution: Resolution = Field(
        default=Resolution.RES_1K,
        description="Output image resolution.",
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

    def _get_model(self) -> str:
        return "flux-2/pro-image-to-image"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if context is None:
            raise ValueError("Context is required for image upload")
        if not self.prompt:
            raise ValueError("Prompt cannot be empty")
        if not self.image.is_set():
            raise ValueError("Image is required")
        input_url = await self._upload_image(context, self.image)
        return {
            "prompt": self.prompt,
            "input_urls": [
                input_url,
            ],
            "aspect_ratio": self.aspect_ratio.value,
            "resolution": self.resolution.value,
            "steps": self.steps,
            "guidance_scale": self.guidance_scale,
        }

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_bytes = await self._execute_task(context)
        return await context.image_from_bytes(image_bytes)


class Flux2FlexTextToImage(KieBaseNode):
    """Generate images using Black Forest Labs' Flux 2 Flex Text-to-Image model via Kie.ai.

    kie, flux, flux-2, flux-flex, black-forest-labs, image generation, ai, text-to-image

    Use cases:
    - Generate high-quality images from text with flexible parameters
    - Create professional visual content
    - Generate images with fine detail and artistic style
    """

    _expose_as_tool: ClassVar[bool] = True
    _poll_interval: float = 1.5
    _max_poll_attempts: int = 60

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

    class Resolution(str, Enum):
        RES_1K = "1K"
        RES_2K = "2K"

    resolution: Resolution = Field(
        default=Resolution.RES_1K,
        description="Output image resolution.",
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

    def _get_model(self) -> str:
        return "flux-2/flex-text-to-image"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if not self.prompt:
            raise ValueError("Prompt cannot be empty")
        return {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "resolution": self.resolution.value,
            "steps": self.steps,
            "guidance_scale": self.guidance_scale,
        }

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_bytes = await self._execute_task(context)
        return await context.image_from_bytes(image_bytes)


class Flux2FlexImageToImage(KieBaseNode):
    """Generate images using Black Forest Labs' Flux 2 Flex Image-to-Image model via Kie.ai.

    kie, flux, flux-2, flux-flex, black-forest-labs, image generation, ai, image-to-image

    Use cases:
    - Transform existing images with text prompts
    - Apply artistic styles to photos
    - Create variations of existing images
    - Enhance and modify images
    """

    _expose_as_tool: ClassVar[bool] = True
    _poll_interval: float = 1.5
    _max_poll_attempts: int = 60

    prompt: str = Field(
        default="",
        description="The text prompt describing how to transform the image.",
    )

    image: ImageRef = Field(
        default=ImageRef(),
        description="The source image to transform.",
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

    class Resolution(str, Enum):
        RES_1K = "1K"
        RES_2K = "2K"

    resolution: Resolution = Field(
        default=Resolution.RES_1K,
        description="Output image resolution.",
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

    def _get_model(self) -> str:
        return "flux-2/flex-image-to-image"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if context is None:
            raise ValueError("Context is required for image upload")
        if not self.prompt:
            raise ValueError("Prompt cannot be empty")
        if not self.image.is_set():
            raise ValueError("Image is required")
        input_url = await self._upload_image(context, self.image)
        return {
            "prompt": self.prompt,
            "input_urls": [
                input_url,
            ],
            "aspect_ratio": self.aspect_ratio.value,
            "resolution": self.resolution.value,
            "steps": self.steps,
            "guidance_scale": self.guidance_scale,
        }

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_bytes = await self._execute_task(context)
        return await context.image_from_bytes(image_bytes)


class Seedream45TextToImage(KieBaseNode):
    """Generate images using ByteDance's Seedream 4.5 Text-to-Image model via Kie.ai.

    kie, seedream, bytedance, image generation, ai, text-to-image, 4k

    Seedream 4.5 generates high-quality visuals up to 4K resolution with
    improved detail fidelity, multi-image blending, and sharp text/face rendering.

    Use cases:
    - Generate creative and artistic images from text
    - Create diverse visual content up to 4K
    - Generate illustrations with unique styles
    """

    _expose_as_tool: ClassVar[bool] = True
    _poll_interval: float = 1.5
    _max_poll_attempts: int = 60

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

    def _get_model(self) -> str:
        return "seedream/4.5-text-to-image"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if not self.prompt:
            raise ValueError("Prompt cannot be empty")
        return {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
        }

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_bytes = await self._execute_task(context)
        return await context.image_from_bytes(image_bytes)


class Seedream45Edit(KieBaseNode):
    """Edit images using ByteDance's Seedream 4.5 Edit model via Kie.ai.

    kie, seedream, bytedance, image editing, ai, image-to-image, 4k

    Seedream 4.5 Edit allows you to modify existing images while maintaining
    high quality and detail fidelity up to 4K resolution.

    Use cases:
    - Edit and enhance existing images
    - Apply style changes to photos
    - Modify specific regions of images
    - Improve image quality and resolution
    """

    _expose_as_tool: ClassVar[bool] = True
    _poll_interval: float = 1.5
    _max_poll_attempts: int = 60

    prompt: str = Field(
        default="",
        description="The text prompt describing how to edit the image.",
    )

    image: ImageRef = Field(
        default=ImageRef(),
        description="The source image to edit.",
    )

    class AspectRatio(str, Enum):
        SQUARE = "1:1"
        LANDSCAPE = "16:9"
        PORTRAIT = "9:16"
        WIDE = "4:3"
        TALL = "3:4"

    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.SQUARE,
        description="The aspect ratio of the output image.",
    )

    def _get_model(self) -> str:
        return "seedream/4.5-edit"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if context is None:
            raise ValueError("Context is required for image upload")
        if not self.prompt:
            raise ValueError("Prompt cannot be empty")
        if not self.image.is_set():
            raise ValueError("Image is required")
        input_url = await self._upload_image(context, self.image)
        return {
            "prompt": self.prompt,
            "input_urls": [
                input_url,
            ],
            "aspect_ratio": self.aspect_ratio.value,
        }

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_bytes = await self._execute_task(context)
        return await context.image_from_bytes(image_bytes)


class ZImage(KieBaseNode):
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
    _poll_interval: float = 1.5
    _max_poll_attempts: int = 60

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

    def _get_model(self) -> str:
        return "z-image"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if not self.prompt:
            raise ValueError("Prompt cannot be empty")
        return {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
        }

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_bytes = await self._execute_task(context)
        return await context.image_from_bytes(image_bytes)


class NanoBanana(KieBaseNode):
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
    _poll_interval: float = 1.5
    _max_poll_attempts: int = 60

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

    def _get_model(self) -> str:
        return "google/nano-banana"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if not self.prompt:
            raise ValueError("Prompt cannot be empty")
        return {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
        }

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_bytes = await self._execute_task(context)
        return await context.image_from_bytes(image_bytes)


class NanoBananaPro(KieBaseNode):
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
    _poll_interval: float = 1.5
    _max_poll_attempts: int = 60

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

    def _get_model(self) -> str:
        return "google/nano-banana-pro"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if not self.prompt:
            raise ValueError("Prompt cannot be empty")
        return {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
        }

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_bytes = await self._execute_task(context)
        return await context.image_from_bytes(image_bytes)


class FluxKontext(KieBaseNode):
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
    _poll_interval: float = 1.5
    _max_poll_attempts: int = 60

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

    def _get_model(self) -> str:
        return "flux-kontext"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
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


class GrokImagineTextToImage(KieBaseNode):
    """Generate images using xAI's Grok Imagine Text-to-Image model via Kie.ai.

    kie, grok, xai, image generation, ai, text-to-image, multimodal

    Grok Imagine is a multimodal generative model that can generate images
    from text prompts.

    Use cases:
    - Generate images from text descriptions
    - Create visual content with AI
    """

    _expose_as_tool: ClassVar[bool] = True
    _poll_interval: float = 1.5
    _max_poll_attempts: int = 60

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

    def _get_model(self) -> str:
        return "grok-imagine/text-to-image"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if not self.prompt:
            raise ValueError("Prompt cannot be empty")
        return {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
        }

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_bytes = await self._execute_task(context)
        return await context.image_from_bytes(image_bytes)


class GrokImagineUpscale(KieBaseNode):
    """Upscale images using xAI's Grok Imagine Upscale model via Kie.ai.

    kie, grok, xai, upscale, enhance, image, ai, super-resolution

    Grok Imagine Upscale enhances and upscales images to higher resolutions
    while maintaining quality and detail.

    Use cases:
    - Upscale low-resolution images
    - Enhance image quality and detail
    - Improve resolution for printing or display
    """

    _expose_as_tool: ClassVar[bool] = True
    _poll_interval: float = 1.5
    _max_poll_attempts: int = 60

    image: ImageRef = Field(
        default=ImageRef(),
        description="The image to upscale.",
    )

    class ScaleFactor(str, Enum):
        X2 = "2"
        X4 = "4"
        X8 = "8"

    scale_factor: ScaleFactor = Field(
        default=ScaleFactor.X2,
        description="The upscaling factor (2x, 4x, or 8x).",
    )

    def _get_model(self) -> str:
        return "grok-imagine/upscale"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if context is None:
            raise ValueError("Context is required for image upload")
        if not self.image.is_set():
            raise ValueError("Image is required")
        input_url = await self._upload_image(context, self.image)
        return {
            "input_urls": [
                input_url,
            ],
            "scale_factor": self.scale_factor.value,
        }

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_bytes = await self._execute_task(context)
        return await context.image_from_bytes(image_bytes)


class QwenTextToImage(KieBaseNode):
    """Generate images using Qwen's Text-to-Image model via Kie.ai.

    kie, qwen, alibaba, image generation, ai, text-to-image

    Qwen's text-to-image model generates high-quality images from text descriptions.

    Use cases:
    - Generate images from text descriptions
    - Create artistic and realistic images
    - Generate illustrations and artwork
    """

    _expose_as_tool: ClassVar[bool] = True
    _poll_interval: float = 1.5
    _max_poll_attempts: int = 60

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

    def _get_model(self) -> str:
        return "qwen/text-to-image"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if not self.prompt:
            raise ValueError("Prompt cannot be empty")
        return {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
        }

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_bytes = await self._execute_task(context)
        return await context.image_from_bytes(image_bytes)


class QwenImageToImage(KieBaseNode):
    """Transform images using Qwen's Image-to-Image model via Kie.ai.

    kie, qwen, alibaba, image transformation, ai, image-to-image

    Qwen's image-to-image model transforms existing images based on text prompts.

    Use cases:
    - Transform and edit existing images
    - Apply styles and effects to photos
    - Create variations of images
    - Enhance or modify image content
    """

    _expose_as_tool: ClassVar[bool] = True
    _poll_interval: float = 1.5
    _max_poll_attempts: int = 60

    prompt: str = Field(
        default="",
        description="The text prompt describing how to transform the image.",
    )

    image: ImageRef = Field(
        default=ImageRef(),
        description="The source image to transform.",
    )

    class AspectRatio(str, Enum):
        SQUARE = "1:1"
        LANDSCAPE = "16:9"
        PORTRAIT = "9:16"
        WIDE = "4:3"
        TALL = "3:4"

    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.SQUARE,
        description="The aspect ratio of the output image.",
    )

    def _get_model(self) -> str:
        return "qwen/image-to-image"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if context is None:
            raise ValueError("Context is required for image upload")
        if not self.prompt:
            raise ValueError("Prompt cannot be empty")
        if not self.image.is_set():
            raise ValueError("Image is required")
        input_url = await self._upload_image(context, self.image)
        return {
            "prompt": self.prompt,
            "input_urls": [
                input_url,
            ],
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
    _poll_interval: float = 1.5
    _max_poll_attempts: int = 60

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

    def _get_model(self) -> str:
        return "topaz-image-upscale"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if context is None:
            raise ValueError("Context is required for image upload")
        if not self.image.is_set():
            raise ValueError("Image is required")
        input_url = await self._upload_image(context, self.image)
        return {
            "input_urls": [
                input_url,
            ],
            "scale_factor": self.scale_factor.value,
        }

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_bytes = await self._execute_task(context)
        return await context.image_from_bytes(image_bytes)
