"""Kie.ai API nodes for AI generation.

This package provides nodes for integrating with Kie.ai's unified API platform,
which offers access to state-of-the-art AI models for:
- Image generation and editing (4O, Flux, Seedream, Z-Image, Nano Banana, Grok, Topaz)
- Video generation (Veo, Wan, Sora, Seedance, Hailuo, Kling)
- Music generation (Suno)

All nodes require a KIE_API_KEY secret to be configured.
"""

from nodetool.nodes.kie.image import (
    Flux2ProTextToImage,
    Flux2ProImageToImage,
    Flux2FlexTextToImage,
    Flux2FlexImageToImage,
    Seedream45TextToImage,
    Seedream45Edit,
    ZImage,
    NanoBanana,
    NanoBananaPro,
    FluxKontext,
    GrokImagineTextToImage,
    GrokImagineUpscale,
    TopazImageUpscale,
)

from nodetool.nodes.kie.video import (
    Sora2TextToVideo,
    Sora2ProTextToVideo,
    Sora2ProImageToVideo,
    Sora2ProStoryboard,
    Sora2ImageToVideo,
    SeedanceV1LiteTextToVideo,
    SeedanceV1ProTextToVideo,
    SeedanceV1LiteImageToVideo,
    SeedanceV1ProImageToVideo,
    SeedanceV1ProFastImageToVideo,
    HailuoImageToVideoPro,
    HailuoImageToVideoStandard,
    KlingTextToVideo,
    KlingImageToVideo,
    KlingAIAvatar,
    TopazVideoUpscale,
    GrokImagineImageToVideo,
    GrokImagineTextToVideo,
)

from nodetool.nodes.kie.audio import Suno

__all__ = [
    # Image generation nodes
    "Flux2ProTextToImage",
    "Flux2ProImageToImage",
    "Flux2FlexTextToImage",
    "Flux2FlexImageToImage",
    "Seedream45TextToImage",
    "Seedream45Edit",
    "ZImage",
    "NanoBanana",
    "NanoBananaPro",
    "FluxKontext",
    "GrokImagineTextToImage",
    "GrokImagineUpscale",
    "TopazImageUpscale",
    # Video generation nodes
    "Sora2TextToVideo",
    "Sora2ProTextToVideo",
    "Sora2ProImageToVideo",
    "Sora2ProStoryboard",
    "Sora2ImageToVideo",
    "SeedanceV1LiteTextToVideo",
    "SeedanceV1ProTextToVideo",
    "SeedanceV1LiteImageToVideo",
    "SeedanceV1ProImageToVideo",
    "SeedanceV1ProFastImageToVideo",
    "HailuoImageToVideoPro",
    "HailuoImageToVideoStandard",
    "KlingTextToVideo",
    "KlingImageToVideo",
    "KlingAIAvatar",
    "TopazVideoUpscale",
    "GrokImagineImageToVideo",
    "GrokImagineTextToVideo",
    # Audio generation nodes
    "Suno",
]
