from nodetool.metadata.types import ColorRef, FontRef
import pytest
from nodetool.dsl.graph import graph_result
from nodetool.dsl.lib.pillow.enhance import (
    AutoContrast,
    Brightness,
    Contrast,
    EdgeEnhance,
    Sharpen,
    UnsharpMask,
)
from nodetool.dsl.lib.pillow.draw import (
    Background,
    GaussianNoise,
    RenderText,
)
from nodetool.dsl.nodetool.output import ImageOutput

# Create a background image
bg = Background(
    width=512,
    height=512,
    color=ColorRef(type="color", value="#E0E0E0"),
)
background = ImageOutput(
    name="background",
    value=bg.output,
)

# Add text to an image
bg_text = Background(
    width=512, height=512, color=ColorRef(type="color", value="#FFFFFF")
)
render_text = RenderText(
    image=bg_text.output,
    text="Hello, World!",
    x=256,
    y=256,
    size=48,
    color=ColorRef(type="color", value="#000000"),
    align=RenderText.TextAlignment("center"),
    font=FontRef(name="Verdana.ttf"),
)
text_on_image = ImageOutput(
    name="text_on_image",
    value=render_text.output,
)

# Image enhancement chain
bg_enhance = Background(
    width=512, height=512, color=ColorRef(type="color", value="#FFFFFF")
)
brightness_node = Brightness(image=bg_enhance.output, factor=1.2)
contrast_node = Contrast(image=brightness_node.output, factor=1.3)
sharpen_node = Sharpen(image=contrast_node.output)
enhanced_image = ImageOutput(
    name="enhanced_image",
    value=sharpen_node.output,
)

# Noise and edge enhancement
noise = GaussianNoise(width=512, height=512, mean=0.5, stddev=0.1)
edge_enhance_node = EdgeEnhance(image=noise.output)
noise_with_edges = ImageOutput(
    name="noise_with_edges",
    value=edge_enhance_node.output,
)

# Advanced image processing
bg_advanced = Background(
    width=512, height=512, color=ColorRef(type="color", value="#CCCCCC")
)
autocontrast_node = AutoContrast(image=bg_advanced.output, cutoff=5)
unsharp_node = UnsharpMask(
    image=autocontrast_node.output,
    radius=2,
    percent=150,
    threshold=3,
)
advanced_image = ImageOutput(
    name="advanced_image",
    value=unsharp_node.output,
)


@pytest.mark.asyncio
async def test_background():
    result = await graph_result(background)
    assert result["background"] is not None


# TODO: fails on CI
# @pytest.mark.asyncio
# async def test_text_on_image():
#     result = await graph_result(text_on_image)
#     assert result["text_on_image"] is not None


@pytest.mark.asyncio
async def test_enhanced_image():
    result = await graph_result(enhanced_image)
    assert result["enhanced_image"] is not None


@pytest.mark.asyncio
async def test_noise_with_edges():
    result = await graph_result(noise_with_edges)
    assert result["noise_with_edges"] is not None


@pytest.mark.asyncio
async def test_advanced_image():
    result = await graph_result(advanced_image)
    assert result["advanced_image"] is not None
