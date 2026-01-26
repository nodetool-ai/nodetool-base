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
import tempfile
import uuid
from urllib.parse import urlparse
from abc import abstractmethod
from enum import Enum
from typing import Any, ClassVar

import aiohttp
from pydantic import Field

from nodetool.config.logging_config import get_logger
from nodetool.media.common.media_utils import FFMPEG_PATH
from nodetool.metadata.types import ImageRef, VideoRef
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

    # User-configurable timeout (0 = use class default)
    timeout_seconds: int = Field(
        default=0,
        ge=0,
        le=3600,
        description="Timeout in seconds for API calls (0 = use default)",
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

    def _check_response_status(self, response_data: dict) -> None:
        """Check response status code and raise nicer error."""
        try:
            status = int(response_data.get("code"))  # type: ignore
        except (ValueError, TypeError):
            pass

        error_map = {
            401: "Unauthorized - Authentication credentials are missing or invalid",
            402: "Insufficient Credits - Account does not have enough credits to perform the operation",
            404: "Not Found - The requested resource or endpoint does not exist",
            422: "Validation Error - The request parameters failed validation checks",
            429: "Rate Limited - Request limit has been exceeded for this resource",
            455: "Service Unavailable - System is currently undergoing maintenance",
            500: "Server Error - An unexpected error occurred while processing the request",
            501: "Generation Failed - Content generation task failed",
            505: "Feature Disabled - The requested feature is currently disabled",
        }
        if status in error_map:
            raise ValueError(error_map[status] + str(response_data))

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
                    if "code" in response_data:
                        self._check_response_status(response_data)

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

    async def _upload_image(self, context: ProcessingContext, image: ImageRef) -> str:
        if image.uri and image.uri.startswith(("http://", "https://")):
            if "localhost" not in image.uri and "127.0.0.1" not in image.uri:
                return image.uri

        return await self._upload_asset(
            context,
            asset=image,
            upload_path="images/user-uploads",
            default_extension=".png",
        )

    async def _upload_audio(self, context: ProcessingContext, audio: Any) -> str:
        """Upload audio to Kie.ai after converting to MP3 format."""
        from io import BytesIO
        from nodetool.metadata.types import AudioRef

        # Convert audio to AudioSegment (handles any audio format)
        audio_segment = await context.audio_to_audio_segment(audio)

        # Export to MP3 bytes
        mp3_buffer = BytesIO()
        audio_segment.export(mp3_buffer, format="mp3")
        mp3_buffer.seek(0)

        # Create a temporary AudioRef with MP3 data for upload
        mp3_audio = AudioRef(data=mp3_buffer.read())

        return await self._upload_asset(
            context,
            asset=mp3_audio,
            upload_path="audio/user-uploads",
            default_extension=".mp3",
        )

    async def _convert_video_to_mp4(self, source_bytes: bytes) -> bytes:
        """Convert input video bytes to MP4 using ffmpeg."""
        input_path = None
        output_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".input") as temp_in:
                temp_in.write(source_bytes)
                temp_in.flush()
                input_path = temp_in.name

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_out:
                output_path = temp_out.name

            cmd = [
                FFMPEG_PATH,
                "-y",
                "-i",
                input_path,
                "-c:v",
                "libx264",
                "-c:a",
                "aac",
                "-movflags",
                "+faststart",
                output_path,
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _stdout, stderr = await process.communicate()
            if process.returncode != 0:
                error_msg = stderr.decode(errors="ignore").strip()
                raise ValueError(
                    f"ffmpeg error (using {FFMPEG_PATH}): {error_msg or 'unknown error'}"
                )

            with open(output_path, "rb") as f:
                return f.read()
        finally:
            if input_path and os.path.exists(input_path):
                os.remove(input_path)
            if output_path and os.path.exists(output_path):
                os.remove(output_path)

    async def _upload_video(self, context: ProcessingContext, video: Any) -> str:
        file_obj = await context.asset_to_io(video)
        try:
            filename = self._resolve_upload_filename(
                video, default_extension=".mp4", file_obj=file_obj
            )
            ext = os.path.splitext(filename)[1].lower()
            format_hint = getattr(video, "format", None)
            needs_conversion = ext != ".mp4" or (
                isinstance(format_hint, str) and format_hint.lower() != "mp4"
            )

            if needs_conversion:
                log.info(
                    "Converting video to MP4 before upload for Kie.ai compatibility."
                )
                source_bytes = file_obj.read()
                mp4_bytes = await self._convert_video_to_mp4(source_bytes)
                video = VideoRef(data=mp4_bytes)
        finally:
            close_fn = getattr(file_obj, "close", None)
            if callable(close_fn):
                close_fn()

        return await self._upload_asset(
            context,
            asset=video,
            upload_path="videos/user-uploads",
            default_extension=".mp4",
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
        log.info(f"Submitting task to {url} with payload: {payload}")
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
        """Poll for task completion status.

        Uses the /api/v1/jobs/recordInfo endpoint:
        GET /api/v1/jobs/recordInfo?taskId=task_12345678
        """
        url = f"{KIE_API_BASE_URL}/api/v1/jobs/recordInfo?taskId={task_id}"
        headers = self._get_headers(api_key)

        # Calculate max attempts based on timeout_seconds if set, otherwise use class default
        max_attempts = self._max_poll_attempts
        if self.timeout_seconds > 0:
            max_attempts = max(1, int(self.timeout_seconds / self._poll_interval))

        for attempt in range(max_attempts):
            log.debug(
                f"Polling task status (attempt {attempt + 1}/{self._max_poll_attempts})"
            )
            async with session.get(url, headers=headers) as response:
                status_data = await response.json()
                if "code" in status_data:
                    self._check_response_status(status_data)

                if self._is_task_complete(status_data):
                    log.debug("Task completed successfully")
                    return status_data

                if self._is_task_failed(status_data):
                    error_msg = self._get_error_message(status_data)
                    raise ValueError(f"Task failed: {error_msg}")

            await asyncio.sleep(self._poll_interval)

        raise TimeoutError(
            f"Task did not complete within {max_attempts * self._poll_interval} seconds"
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
            if "code" in status_data:
                self._check_response_status(status_data)
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

    async def _execute_task(self, context: ProcessingContext) -> tuple[bytes, str]:
        """Execute the full task workflow: submit, poll, download."""
        api_key = await self._get_api_key(context)

        async with aiohttp.ClientSession() as session:
            submit_response = await self._submit_task(session, api_key, context)
            task_id = self._extract_task_id(submit_response)
            log.info(f"Task submitted with ID: {task_id}")

            await self._poll_status(session, api_key, task_id)

            return await self._download_result(session, api_key, task_id), task_id


class Flux2ProTextToImage(KieBaseNode):
    """Generate images using Black Forest Labs' Flux 2 Pro Text-to-Image model via Kie.ai.

    kie, flux, flux-2, flux-pro, black-forest-labs, image generation, ai, text-to-image

    Use cases:
    - Generate high-quality artistic images from text
    - Create professional visual content
    - Generate images with fine detail and artistic style
    """
    _auto_save_asset: ClassVar[bool] = True

    _expose_as_tool: ClassVar[bool] = True
    _poll_interval: float = 1.5
    _max_poll_attempts: int = 200  # 300 seconds default

    @classmethod
    def get_title(cls) -> str:
        return "Flux 2 Pro Text To Image"

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
        image_bytes, task_id = await self._execute_task(context)
        return await context.image_from_bytes(
            image_bytes, metadata={"task_id": task_id}
        )


class Flux2ProImageToImage(KieBaseNode):
    """Generate images using Black Forest Labs' Flux 2 Pro Image-to-Image model via Kie.ai.

    kie, flux, flux-2, flux-pro, black-forest-labs, image generation, ai, image-to-image

    Use cases:
    - Transform existing images with text prompts
    - Apply artistic styles to photos
    - Create variations of existing images
    - Enhance and modify images
    """
    _auto_save_asset: ClassVar[bool] = True

    _expose_as_tool: ClassVar[bool] = True
    _poll_interval: float = 1.5
    _max_poll_attempts: int = 200  # 300 seconds default

    @classmethod
    def get_title(cls) -> str:
        return "Flux 2 Pro Image To Image"

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
        image_bytes, task_id = await self._execute_task(context)
        return await context.image_from_bytes(
            image_bytes, metadata={"task_id": task_id}
        )


class Flux2FlexTextToImage(KieBaseNode):
    """Generate images using Black Forest Labs' Flux 2 Flex Text-to-Image model via Kie.ai.

    kie, flux, flux-2, flux-flex, black-forest-labs, image generation, ai, text-to-image

    Use cases:
    - Generate high-quality images from text with flexible parameters
    - Create professional visual content
    - Generate images with fine detail and artistic style
    """
    _auto_save_asset: ClassVar[bool] = True

    _expose_as_tool: ClassVar[bool] = True
    _poll_interval: float = 1.5
    _max_poll_attempts: int = 200  # 300 seconds default

    @classmethod
    def get_title(cls) -> str:
        return "Flux 2 Flex Text To Image"

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
        image_bytes, task_id = await self._execute_task(context)
        return await context.image_from_bytes(
            image_bytes, metadata={"task_id": task_id}
        )


class Flux2FlexImageToImage(KieBaseNode):
    """Generate images using Black Forest Labs' Flux 2 Flex Image-to-Image model via Kie.ai.

    kie, flux, flux-2, flux-flex, black-forest-labs, image generation, ai, image-to-image

    Use cases:
    - Transform existing images with text prompts
    - Apply artistic styles to photos
    - Create variations of existing images
    - Enhance and modify images
    """
    _auto_save_asset: ClassVar[bool] = True

    _expose_as_tool: ClassVar[bool] = True
    _poll_interval: float = 1.5
    _max_poll_attempts: int = 200  # 300 seconds default

    @classmethod
    def get_title(cls) -> str:
        return "Flux 2 Flex Image To Image"

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
        image_bytes, task_id = await self._execute_task(context)
        return await context.image_from_bytes(
            image_bytes, metadata={"task_id": task_id}
        )


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
    _auto_save_asset: ClassVar[bool] = True

    _expose_as_tool: ClassVar[bool] = True
    _poll_interval: float = 1.5
    _max_poll_attempts: int = 200  # 300 seconds default

    @classmethod
    def get_title(cls) -> str:
        return "Seedream 4.5 Text To Image"

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

    class Quality(str, Enum):
        BASIC = "basic"
        HIGH = "high"

    quality: Quality = Field(
        default=Quality.BASIC,
        description="Basic outputs 2K images, while High outputs 4K images.",
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
            "quality": self.quality.value,
        }

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_bytes, task_id = await self._execute_task(context)
        return await context.image_from_bytes(
            image_bytes, metadata={"task_id": task_id}
        )


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
    _max_poll_attempts: int = 200  # 300 seconds default

    @classmethod
    def get_title(cls) -> str:
        return "Seedream 4.5 Edit"

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

    class Quality(str, Enum):
        BASIC = "basic"
        HIGH = "high"

    quality: Quality = Field(
        default=Quality.BASIC,
        description="Basic outputs 2K images, while High outputs 4K images.",
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
            "quality": self.quality.value,
        }

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_bytes, task_id = await self._execute_task(context)
        return await context.image_from_bytes(
            image_bytes, metadata={"task_id": task_id}
        )


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
    _auto_save_asset: ClassVar[bool] = True

    _expose_as_tool: ClassVar[bool] = True
    _poll_interval: float = 1.5
    _max_poll_attempts: int = 200  # 300 seconds default

    @classmethod
    def get_title(cls) -> str:
        return "Z-Image Turbo"

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
        image_bytes, task_id = await self._execute_task(context)
        return await context.image_from_bytes(
            image_bytes, metadata={"task_id": task_id}
        )


class NanoBanana(KieBaseNode):
    """Generate images using Google's Nano Banana model (Gemini 2.5) via Kie.ai.

    kie, nano-banana, google, gemini, image generation, ai, text-to-image, fast
    """
    _auto_save_asset: ClassVar[bool] = True

    _expose_as_tool: ClassVar[bool] = True
    _poll_interval: float = 1.5
    _max_poll_attempts: int = 200  # 300 seconds default

    @classmethod
    def get_title(cls) -> str:
        return "Nano Banana"

    prompt: str = Field(
        default="",
        description="The text prompt describing the image to generate.",
    )

    class ImageSize(str, Enum):
        SQUARE = "1:1"
        PORTRAIT_9_16 = "9:16"
        LANDSCAPE_16_9 = "16:9"
        PORTRAIT_3_4 = "3:4"
        LANDSCAPE_4_3 = "4:3"
        LANDSCAPE_3_2 = "3:2"
        PORTRAIT_2_3 = "2:3"
        LANDSCAPE_5_4 = "5:4"
        PORTRAIT_4_5 = "4:5"
        WIDE_21_9 = "21:9"
        AUTO = "auto"

    image_size: ImageSize = Field(
        default=ImageSize.SQUARE,
        description="The size of the output image.",
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
            "output_format": "png",
            "image_size": self.image_size.value,
        }

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_bytes, task_id = await self._execute_task(context)
        return await context.image_from_bytes(
            image_bytes, metadata={"task_id": task_id}
        )


class NanoBananaPro(KieBaseNode):
    """Generate images using Google's Nano Banana Pro model (Gemini 3.0) via Kie.ai.

    kie, nano-banana-pro, google, gemini, image generation, ai, text-to-image, 4k, high-fidelity
    """
    _auto_save_asset: ClassVar[bool] = True

    _expose_as_tool: ClassVar[bool] = True
    _poll_interval: float = 1.5
    _max_poll_attempts: int = 200  # 300 seconds default

    @classmethod
    def get_title(cls) -> str:
        return "Nano Banana Pro"

    prompt: str = Field(
        default="",
        description="The text prompt describing the image to generate.",
    )

    image_input: list[ImageRef] = Field(
        default=[],
        description="Optional image inputs for multimodal generation.",
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
        RES_4K = "4K"

    resolution: Resolution = Field(
        default=Resolution.RES_2K,
        description="Output image resolution.",
    )

    def _get_model(self) -> str:
        return "nano-banana-pro"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if not self.prompt:
            raise ValueError("Prompt cannot be empty")

        image_urls = []
        if context:
            for img in self.image_input:
                if img.is_set():
                    url = await self._upload_image(context, img)
                    image_urls.append(url)

        return {
            "prompt": self.prompt,
            "image_input": image_urls,
            "aspect_ratio": self.aspect_ratio.value,
            "resolution": self.resolution.value,
            "output_format": "png",
        }

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_bytes, task_id = await self._execute_task(context)
        return await context.image_from_bytes(
            image_bytes, metadata={"task_id": task_id}
        )


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
    _auto_save_asset: ClassVar[bool] = True

    _expose_as_tool: ClassVar[bool] = True
    _poll_interval: float = 1.5
    _max_poll_attempts: int = 200  # 300 seconds default

    @classmethod
    def get_title(cls) -> str:
        return "Flux Kontext"

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
        image_bytes, task_id = await self._execute_task(context)
        return await context.image_from_bytes(
            image_bytes, metadata={"task_id": task_id}
        )


class GrokImagineTextToImage(KieBaseNode):
    """Generate images using xAI's Grok Imagine Text-to-Image model via Kie.ai.

    kie, grok, xai, image generation, ai, text-to-image, multimodal

    Grok Imagine is a multimodal generative model that can generate images
    from text prompts.

    Use cases:
    - Generate images from text descriptions
    - Create visual content with AI
    """
    _auto_save_asset: ClassVar[bool] = True

    _expose_as_tool: ClassVar[bool] = True
    _poll_interval: float = 1.5
    _max_poll_attempts: int = 200  # 300 seconds default

    @classmethod
    def get_title(cls) -> str:
        return "Grok Imagine Text To Image"

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
        image_bytes, task_id = await self._execute_task(context)
        return await context.image_from_bytes(
            image_bytes, metadata={"task_id": task_id}
        )


class GrokImagineUpscale(KieBaseNode):
    """Upscale images using xAI's Grok Imagine Upscale model via Kie.ai.

    kie, grok, xai, upscale, enhance, image, ai, super-resolution

    Grok Imagine Upscale enhances and upscales images to higher resolutions
    while maintaining quality and detail.

    Constraints:
    - Only images generated by Kie AI models (via Grok Imagine) are supported for upscaling.
    """
    _expose_as_tool: ClassVar[bool] = True
    _poll_interval: float = 1.5
    _max_poll_attempts: int = 200  # 300 seconds default

    @classmethod
    def get_title(cls) -> str:
        return "Grok Imagine Upscale"

    image: ImageRef = Field(
        default=ImageRef(),
        description="The image to upscale. Must be an image previously generated by a Kie.ai node.",
    )

    def _get_model(self) -> str:
        return "grok-imagine/upscale"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if not self.image.is_set():
            raise ValueError("Image is required")

        task_id = self.image.metadata.get("task_id")
        if not task_id:
            raise ValueError(
                "Image metadata does not contain a 'task_id'. "
                "The image must be generated by a Kie.ai node to be upscaled."
            )

        return {
            "task_id": task_id,
        }

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_bytes, task_id = await self._execute_task(context)
        return await context.image_from_bytes(
            image_bytes, metadata={"task_id": task_id}
        )


class QwenTextToImage(KieBaseNode):
    """Generate images using Qwen's Text-to-Image model via Kie.ai.

    kie, qwen, alibaba, image generation, ai, text-to-image

    Qwen's text-to-image model generates high-quality images from text descriptions.

    Use cases:
    - Generate images from text descriptions
    - Create artistic and realistic images
    - Generate illustrations and artwork
    """
    _auto_save_asset: ClassVar[bool] = True

    _expose_as_tool: ClassVar[bool] = True
    _poll_interval: float = 1.5
    _max_poll_attempts: int = 200  # 300 seconds default

    @classmethod
    def get_title(cls) -> str:
        return "Qwen Text To Image"

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
        image_bytes, task_id = await self._execute_task(context)
        return await context.image_from_bytes(
            image_bytes, metadata={"task_id": task_id}
        )


class QwenImageToImage(KieBaseNode):
    """Transform images using Qwen's Image-to-Image model via Kie.ai.

    kie, qwen, alibaba, image transformation, ai, image-to-image

    Qwen's image-to-image model transforms images based on text prompts
    while preserving the overall structure and style.

    Use cases:
    - Transform images with text guidance
    - Apply artistic styles to photos
    - Create variations of existing images
    """
    _auto_save_asset: ClassVar[bool] = True

    _expose_as_tool: ClassVar[bool] = True
    _poll_interval: float = 1.5
    _max_poll_attempts: int = 200  # 300 seconds default

    @classmethod
    def get_title(cls) -> str:
        return "Qwen Image To Image"

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
            "image_url": input_url,
            "aspect_ratio": self.aspect_ratio.value,
        }

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_bytes, task_id = await self._execute_task(context)
        return await context.image_from_bytes(
            image_bytes, metadata={"task_id": task_id}
        )


class TopazImageUpscale(KieBaseNode):
    """Upscale and enhance images using Topaz Labs AI via Kie.ai.

    kie, topaz, upscale, enhance, image, ai, super-resolution

    Topaz Image Upscale uses advanced AI models to enlarge images
    while preserving and enhancing detail.

    Use cases:
    - Upscale low-resolution images
    - Enhance image quality and detail
    - Enlarge images for print or display
    """
    _expose_as_tool: ClassVar[bool] = True
    _poll_interval: float = 1.5
    _max_poll_attempts: int = 200  # 300 seconds default

    @classmethod
    def get_title(cls) -> str:
        return "Topaz Image Upscale"

    image: ImageRef = Field(
        default=ImageRef(),
        description="The image to upscale.",
    )

    class UpscaleFactor(str, Enum):
        X2 = "2"
        X4 = "4"

    upscale_factor: UpscaleFactor = Field(
        default=UpscaleFactor.X2,
        description="The upscaling factor (2x or 4x).",
    )

    def _get_model(self) -> str:
        return "topaz/image-upscale"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if context is None:
            raise ValueError("Context is required for image upload")
        if not self.image.is_set():
            raise ValueError("Image is required")
        image_url = await self._upload_image(context, self.image)
        return {
            "image_url": image_url,
            "upscale_factor": self.upscale_factor.value,
        }

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_bytes, task_id = await self._execute_task(context)
        return await context.image_from_bytes(
            image_bytes, metadata={"task_id": task_id}
        )


class RecraftRemoveBackground(KieBaseNode):
    """Remove background from images using Recraft's model via Kie.ai.

    kie, recraft, remove-background, image processing, ai

    Use cases:
    - Automatically remove backgrounds from photos
    - Create transparent PNGs for design work
    - Isolate subjects in images
    """
    _expose_as_tool: ClassVar[bool] = True
    _poll_interval: float = 1.0
    _max_poll_attempts: int = 30

    image: ImageRef = Field(
        default=ImageRef(),
        description="The image to remove the background from.",
    )

    def _get_model(self) -> str:
        return "recraft/remove-background"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if context is None:
            raise ValueError("Context is required for image upload")
        if not self.image.is_set():
            raise ValueError("Image is required")
        input_url = await self._upload_image(context, self.image)
        return {
            "image": input_url,
        }

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_bytes, task_id = await self._execute_task(context)
        return await context.image_from_bytes(
            image_bytes, metadata={"task_id": task_id}
        )


class IdeogramCharacterRemix(KieBaseNode):
    """Remix characters in images using Ideogram via Kie.ai.

    kie, ideogram, character-remix, image generation, ai, remix

    Ideogram Character Remix allows you to remix images while maintaining character consistency
    using reference images and text prompts.
    """
    _expose_as_tool: ClassVar[bool] = True
    _poll_interval: float = 2.0
    _max_poll_attempts: int = 200  # 300 seconds default

    prompt: str = Field(
        default="",
        description="Text description for remixing.",
    )

    image: ImageRef = Field(
        default=ImageRef(),
        description="Base image to remix.",
    )

    reference_images: list[ImageRef] = Field(
        default=[],
        description="Reference images for character guidance.",
    )

    class RenderingSpeed(str, Enum):
        TURBO = "TURBO"
        BALANCED = "BALANCED"
        QUALITY = "QUALITY"

    rendering_speed: RenderingSpeed = Field(
        default=RenderingSpeed.BALANCED,
        description="Rendering speed preference.",
    )

    class Style(str, Enum):
        AUTO = "AUTO"
        GENERAL = "GENERAL"
        REALISTIC = "REALISTIC"
        DESIGN = "DESIGN"

    style: Style = Field(
        default=Style.AUTO,
        description="Generation style.",
    )

    expand_prompt: bool = Field(
        default=True,
        description="Whether to expand/augment the prompt.",
    )

    class ImageSize(str, Enum):
        SQUARE = "square"
        SQUARE_HD = "square_hd"
        PORTRAIT_4_3 = "portrait_4_3"
        PORTRAIT_16_9 = "portrait_16_9"
        LANDSCAPE_4_3 = "landscape_4_3"
        LANDSCAPE_16_9 = "landscape_16_9"

    image_size: ImageSize = Field(
        default=ImageSize.SQUARE_HD,
        description="The size of the output image.",
    )

    strength: float = Field(
        default=0.8,
        description="How strongly to apply the remix (0.0 to 1.0).",
        ge=0.0,
        le=1.0,
    )

    negative_prompt: str = Field(
        default="",
        description="Undesired elements to exclude from the image.",
    )

    additional_images: list[ImageRef] = Field(
        default=[],
        description="Additional image inputs.",
    )

    reference_mask_urls: str = Field(
        default="",
        description="URL(s) to masks for references (comma-separated).",
    )

    def _get_model(self) -> str:
        return "ideogram/character-remix"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if context is None:
            raise ValueError("Context is required for image upload")

        if not self.image.is_set():
            raise ValueError("Base image is required")

        image_url = await self._upload_image(context, self.image)

        reference_image_urls = []
        for ref_img in self.reference_images:
            if ref_img.is_set():
                url = await self._upload_image(context, ref_img)
                reference_image_urls.append(url)

        additional_image_urls = []
        for add_img in self.additional_images:
            if add_img.is_set():
                url = await self._upload_image(context, add_img)
                additional_image_urls.append(url)

        return {
            "prompt": self.prompt,
            "image_url": image_url,
            "reference_image_urls": reference_image_urls,
            "rendering_speed": self.rendering_speed.value,
            "style": self.style.value,
            "expand_prompt": self.expand_prompt,
            "image_size": self.image_size.value,
            "num_images": "1",
            "strength": self.strength,
            "negative_prompt": self.negative_prompt,
            "image_urls": additional_image_urls,
            "reference_mask_urls": self.reference_mask_urls,
        }

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_bytes, task_id = await self._execute_task(context)
        return await context.image_from_bytes(
            image_bytes, metadata={"task_id": task_id}
        )


class IdeogramV3Reframe(KieBaseNode):
    """Reframe images using Ideogram v3 via Kie.ai.

    kie, ideogram, v3-reframe, image processing, ai, reframe

    Use cases:
    - Reframe and rescale existing images
    - Change aspect ratio of images while maintaining quality
    """
    _expose_as_tool: ClassVar[bool] = True
    _poll_interval: float = 2.0
    _max_poll_attempts: int = 200  # 300 seconds default

    image: ImageRef = Field(
        default=ImageRef(),
        description="URL of the image to reframe.",
    )

    class ImageSize(str, Enum):
        SQUARE = "square"
        SQUARE_HD = "square_hd"
        PORTRAIT_4_3 = "portrait_4_3"
        PORTRAIT_16_9 = "portrait_16_9"
        LANDSCAPE_4_3 = "landscape_4_3"
        LANDSCAPE_16_9 = "landscape_16_9"

    image_size: ImageSize = Field(
        default=ImageSize.SQUARE_HD,
        description="Output resolution preset.",
    )

    class RenderingSpeed(str, Enum):
        TURBO = "TURBO"
        BALANCED = "BALANCED"
        QUALITY = "QUALITY"

    rendering_speed: RenderingSpeed = Field(
        default=RenderingSpeed.BALANCED,
        description="Rendering speed preference.",
    )

    class Style(str, Enum):
        AUTO = "AUTO"
        GENERAL = "GENERAL"
        REALISTIC = "REALISTIC"
        DESIGN = "DESIGN"

    style: Style = Field(
        default=Style.AUTO,
        description="Generation style.",
    )

    seed: int = Field(
        default=0,
        description="RNG seed.",
    )

    def _get_model(self) -> str:
        return "ideogram/v3-reframe"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if context is None:
            raise ValueError("Context is required for image upload")
        if not self.image.is_set():
            raise ValueError("Image is required")
        image_url = await self._upload_image(context, self.image)
        return {
            "image_url": image_url,
            "image_size": self.image_size.value,
            "rendering_speed": self.rendering_speed.value,
            "style": self.style.value,
            "num_images": "1",
            "seed": self.seed,
        }

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_bytes, task_id = await self._execute_task(context)
        return await context.image_from_bytes(
            image_bytes, metadata={"task_id": task_id}
        )


class RecraftCrispUpscale(KieBaseNode):
    """Upscale images using Recraft's Crisp Upscale model via Kie.ai.

    kie, recraft, crisp-upscale, upscale, ai
    """
    _expose_as_tool: ClassVar[bool] = True
    _poll_interval: float = 1.0
    _max_poll_attempts: int = 30

    image: ImageRef = Field(
        default=ImageRef(),
        description="The image to upscale.",
    )

    def _get_model(self) -> str:
        return "recraft/crisp-upscale"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if context is None:
            raise ValueError("Context is required for image upload")
        if not self.image.is_set():
            raise ValueError("Image is required")
        image_url = await self._upload_image(context, self.image)
        return {
            "image": image_url,
        }

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_bytes, task_id = await self._execute_task(context)
        return await context.image_from_bytes(
            image_bytes, metadata={"task_id": task_id}
        )


class Imagen4Fast(KieBaseNode):
    """Generate images using Google's Imagen 4 Fast model via Kie.ai.

    kie, google, imagen, imagen4, fast, image generation, ai
    """
    _auto_save_asset: ClassVar[bool] = True

    _expose_as_tool: ClassVar[bool] = True
    _poll_interval: float = 1.5
    _max_poll_attempts: int = 200  # 300 seconds default

    prompt: str = Field(
        default="",
        description="The text prompt describing the image to generate.",
    )

    negative_prompt: str = Field(
        default="",
        description="Undesired elements to exclude.",
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
        return "google/imagen4-fast"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if not self.prompt:
            raise ValueError("Prompt cannot be empty")
        return {
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "num_images": "1",
        }

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_bytes, task_id = await self._execute_task(context)
        return await context.image_from_bytes(
            image_bytes, metadata={"task_id": task_id}
        )


class Imagen4Ultra(KieBaseNode):
    """Generate images using Google's Imagen 4 Ultra model via Kie.ai.

    kie, google, imagen, imagen4, ultra, image generation, ai
    """
    _auto_save_asset: ClassVar[bool] = True

    _expose_as_tool: ClassVar[bool] = True
    _poll_interval: float = 2.0
    _max_poll_attempts: int = 200  # 300 seconds default

    prompt: str = Field(
        default="",
        description="The text prompt describing the image to generate.",
    )

    negative_prompt: str = Field(
        default="",
        description="Undesired elements to exclude.",
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

    seed: int = Field(
        default=0,
        description="RNG seed.",
    )

    def _get_model(self) -> str:
        return "google/imagen4-ultra"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if not self.prompt:
            raise ValueError("Prompt cannot be empty")
        return {
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "seed": self.seed,
        }

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_bytes, task_id = await self._execute_task(context)
        return await context.image_from_bytes(
            image_bytes, metadata={"task_id": task_id}
        )


class Imagen4(KieBaseNode):
    """Generate images using Google's Imagen 4 model via Kie.ai.

    kie, google, imagen, imagen4, image generation, ai
    """
    _auto_save_asset: ClassVar[bool] = True

    _expose_as_tool: ClassVar[bool] = True
    _poll_interval: float = 2.0
    _max_poll_attempts: int = 200  # 300 seconds default

    prompt: str = Field(
        default="",
        description="The text prompt describing the image to generate.",
    )

    negative_prompt: str = Field(
        default="",
        description="Undesired elements to exclude.",
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

    seed: int = Field(
        default=0,
        description="RNG seed.",
    )

    def _get_model(self) -> str:
        return "google/imagen4"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if not self.prompt:
            raise ValueError("Prompt cannot be empty")
        return {
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "num_images": "1",
            "seed": self.seed,
        }

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_bytes, task_id = await self._execute_task(context)
        return await context.image_from_bytes(
            image_bytes, metadata={"task_id": task_id}
        )


class NanoBananaEdit(KieBaseNode):
    """Edit images using Google's Nano Banana model via Kie.ai.

    kie, google, nano-banana, nano-banana-edit, image editing, ai
    """
    _expose_as_tool: ClassVar[bool] = True
    _poll_interval: float = 1.5
    _max_poll_attempts: int = 200  # 300 seconds default

    prompt: str = Field(
        default="",
        description="Text description of the changes to make.",
    )

    image_input: list[ImageRef] = Field(
        default=[],
        description="Images to edit.",
    )

    class ImageSize(str, Enum):
        SQUARE = "1:1"
        PORTRAIT_9_16 = "9:16"
        LANDSCAPE_16_9 = "16:9"
        PORTRAIT_3_4 = "3:4"
        LANDSCAPE_4_3 = "4:3"
        LANDSCAPE_3_2 = "3:2"
        PORTRAIT_2_3 = "2:3"
        LANDSCAPE_5_4 = "5:4"
        PORTRAIT_4_5 = "4:5"
        WIDE_21_9 = "21:9"
        AUTO = "auto"

    image_size: ImageSize = Field(
        default=ImageSize.SQUARE,
        description="The size of the output image.",
    )

    def _get_model(self) -> str:
        return "google/nano-banana-edit"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if not self.prompt:
            raise ValueError("Prompt cannot be empty")

        image_urls = []
        if context:
            for img in self.image_input:
                if img.is_set():
                    url = await self._upload_image(context, img)
                    image_urls.append(url)

        return {
            "prompt": self.prompt,
            "image_urls": image_urls,
            "output_format": "png",
            "image_size": self.image_size.value,
        }

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_bytes, task_id = await self._execute_task(context)
        return await context.image_from_bytes(
            image_bytes, metadata={"task_id": task_id}
        )


class GPTImage15TextToImage(KieBaseNode):
    """Generate images using OpenAI's GPT Image 1.5 Text-to-Image model via Kie.ai.

    kie, openai, gpt-image, gpt-image-1.5, text-to-image, ai

    GPT Image 1.5 creates high-quality images from text descriptions with
    support for multiple aspect ratios and quality settings.

    Use cases:
    - Generate photorealistic images from text prompts
    - Create artistic and creative imagery
    - Generate product visuals and marketing content
    """

    _expose_as_tool: ClassVar[bool] = True
    _poll_interval: float = 2.0
    _max_poll_attempts: int = 60

    prompt: str = Field(
        default="",
        description="A text description of the image to generate.",
    )

    class AspectRatio(str, Enum):
        RATIO_1_1 = "1:1"
        RATIO_2_3 = "2:3"
        RATIO_3_2 = "3:2"

    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_1_1,
        description="The aspect ratio of the generated image.",
    )

    class Quality(str, Enum):
        MEDIUM = "medium"
        HIGH = "high"

    quality: Quality = Field(
        default=Quality.MEDIUM,
        description="Quality of the generated image. High is slower but more detailed.",
    )

    def _get_model(self) -> str:
        return "gpt-image/1.5-text-to-image"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if not self.prompt:
            raise ValueError("Prompt cannot be empty")
        return {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "quality": self.quality.value,
        }

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_bytes, task_id = await self._execute_task(context)
        return await context.image_from_bytes(
            image_bytes, metadata={"task_id": task_id}
        )
