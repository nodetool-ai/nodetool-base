import uuid
from pydantic import Field
from enum import Enum
from nodetool.metadata.types import ImageRef, VideoRef
from nodetool.workflows.base_node import ApiKeyMissingError, BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from google.genai.client import AsyncClient
from google.genai.types import GenerateVideosConfig
from nodetool.common.environment import Environment
from google.genai import Client


def get_genai_client() -> AsyncClient:
    env = Environment.get_environment()
    api_key = env.get("GEMINI_API_KEY")
    if not api_key:
        raise ApiKeyMissingError("GEMINI_API_KEY is not set")
    return Client(api_key=api_key).aio


class VeoModel(str, Enum):
    VEO_3_PREVIEW = "veo-3.0-generate-preview"
    VEO_3_FAST_PREVIEW = "veo-3.0-fast-generate-preview"
    VEO_2 = "veo-2.0-generate-001"


class VeoAspectRatio(str, Enum):
    RATIO_16_9 = "16:9"
    RATIO_9_16 = "9:16"


class TextToVideo(BaseNode):
    """
    Generate videos from text prompts using Google's Veo models.
    google, video, generation, text-to-video, veo, ai

    This node uses Google's Veo models to generate high-quality videos from text descriptions.
    Supports 720p resolution at 24fps with 8-second duration and native audio generation.

    Use cases:
    - Create cinematic clips from text descriptions
    - Generate social media video content
    - Produce marketing and promotional videos
    - Visualize creative concepts and storyboards
    - Create animated content with accompanying audio
    """

    _expose_as_tool: bool = True

    prompt: str = Field(
        default="", description="The text prompt describing the video to generate"
    )

    model: VeoModel = Field(
        default=VeoModel.VEO_3_PREVIEW,
        description="The Veo model to use for video generation",
    )

    aspect_ratio: VeoAspectRatio = Field(
        default=VeoAspectRatio.RATIO_16_9,
        description="The aspect ratio of the generated video",
    )

    negative_prompt: str = Field(
        default="", description="Negative prompt to guide what to avoid in the video"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        """
        Generate a video using the Veo model based on the provided prompt.

        Returns:
            VideoRef containing the generated video
        """
        if not self.prompt:
            raise ValueError("Video generation prompt is required")

        # client = get_genai_client()

        # Prepare the configuration
        config_args = {}
        if self.aspect_ratio:
            config_args["aspect_ratio"] = self.aspect_ratio.value
        if self.negative_prompt:
            config_args["negative_prompt"] = self.negative_prompt

        config = GenerateVideosConfig(**config_args) if config_args else None
        client = get_genai_client()

        res = await client.models.generate_videos(
            model=self.model.value,
            prompt=self.prompt,
            config=config,
        )
        response = res.response

        assert response, "No video generated"
        assert response.generated_videos, "No video generated"
        assert response.generated_videos[0].video, "No video bytes"
        assert response.generated_videos[0].video.video_bytes, "No video bytes"

        return await context.video_from_bytes(
            response.generated_videos[0].video.video_bytes
        )


class ImageToVideo(BaseNode):
    """
    Generate videos from images using Google's Veo models.
    google, video, generation, image-to-video, veo, ai, animation

    This node uses Google's Veo models to animate static images into dynamic videos.
    Supports 720p resolution at 24fps with 8-second duration and native audio generation.

    Use cases:
    - Animate still artwork and photographs
    - Create dynamic social media content from images
    - Generate product showcase videos from photos
    - Transform static graphics into engaging animations
    - Create video presentations from slide images
    """

    _expose_as_tool: bool = True

    image: ImageRef = Field(
        default=ImageRef(), description="The image to animate into a video"
    )

    prompt: str = Field(
        default="", description="Optional text prompt describing the desired animation"
    )

    model: VeoModel = Field(
        default=VeoModel.VEO_3_PREVIEW,
        description="The Veo model to use for video generation",
    )

    aspect_ratio: VeoAspectRatio = Field(
        default=VeoAspectRatio.RATIO_16_9,
        description="The aspect ratio of the generated video",
    )

    negative_prompt: str = Field(
        default="", description="Negative prompt to guide what to avoid in the video"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        """
        Generate a video from an image using the Veo model.

        Returns:
            VideoRef containing the generated video
        """
        if not self.image.uri:
            raise ValueError("Input image is required")

        client = get_genai_client()

        # Convert image to bytes for upload
        image_bytes = await context.asset_to_bytes(self.image)

        # Prepare the configuration
        config_args = {}
        if self.aspect_ratio:
            config_args["aspect_ratio"] = self.aspect_ratio.value
        if self.negative_prompt:
            config_args["negative_prompt"] = self.negative_prompt

        config = GenerateVideosConfig(**config_args) if config_args else None

        # Generate video from image
        res = await client.models.generate_videos(  # type: ignore
            model=self.model.value,
            prompt=self.prompt or "Animate this image",
            image={
                "image_bytes": image_bytes,
                "mime_type": "image/png",
            },
            config=config,
        )

        response = res.response
        assert response, "No video generated"
        assert response.generated_videos, "No video generated"
        assert response.generated_videos[0].video, "No video bytes"
        assert response.generated_videos[0].video.video_bytes, "No video bytes"

        return await context.video_from_bytes(
            response.generated_videos[0].video.video_bytes
        )
