from __future__ import annotations

from typing import Any

from pydantic import Field

import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode

import nodetool.nodes.mlx.mflux as mlx_mflux


class ImageGeneration(GraphNode):
    """
    Generate images locally using the MFLUX MLX implementation of FLUX.1.
    mlx, flux, image generation, apple-silicon

    Use cases:
    - Create high quality images on Apple Silicon without external APIs
    - Prototype prompts locally before running on cloud inference providers
    - Experiment with quantized FLUX models (schnell/dev/krea-dev variants)
    """

    prompt: str | GraphNode | tuple[GraphNode, str] = Field(
        default="A vivid concept art piece of a futuristic city at sunset",
        description="The text prompt describing the image to generate.",
    )
    model: mlx_mflux.MFluxModel | GraphNode | tuple[GraphNode, str] = Field(
        default=mlx_mflux.MFluxModel.SCHNELL,
        description="MFLUX model variant to load.",
    )
    quantize: mlx_mflux.QuantizationLevel | None | GraphNode | tuple[GraphNode, str] = Field(
        default=mlx_mflux.QuantizationLevel.BITS_8,
        description="Optional quantization level for model weights.",
    )
    steps: int | GraphNode | tuple[GraphNode, str] = Field(
        default=4,
        description="Number of denoising steps for the generation run.",
    )
    guidance: float | None | GraphNode | tuple[GraphNode, str] = Field(
        default=3.5,
        description="Classifier-free guidance scale (used by dev/krea-dev models).",
    )
    height: int | GraphNode | tuple[GraphNode, str] = Field(
        default=1024,
        description="Height of the generated image in pixels.",
    )
    width: int | GraphNode | tuple[GraphNode, str] = Field(
        default=1024,
        description="Width of the generated image in pixels.",
    )
    seed: int | None | GraphNode | tuple[GraphNode, str] = Field(
        default=None,
        description="Seed for deterministic generation. Leave empty for random.",
    )

    @classmethod
    def get_node_type(cls) -> str:
        return "mlx.mflux.ImageGeneration"

    @classmethod
    def get_output_type(cls) -> dict[str, Any]:
        return {"output": types.ImageRef(type="image", uri="", asset_id=None, data=None)}
