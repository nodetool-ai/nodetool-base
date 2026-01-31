import pytest
from io import BytesIO
from PIL import Image
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.metadata.types import ImageRef
from nodetool.nodes.lib.pillow.color_grading import (
    LiftGammaGain,
    CDL,
    ColorBalance,
    Exposure,
    SaturationVibrance,
)


# Create a dummy ImageRef for testing
buffer = BytesIO()
Image.new("RGB", (100, 100), color=(128, 128, 128)).save(buffer, format="PNG")
dummy_image = ImageRef(data=buffer.getvalue())


@pytest.fixture
def context():
    return ProcessingContext(user_id="test", auth_token="test")


@pytest.mark.asyncio
async def test_lift_gamma_gain_defaults(context: ProcessingContext):
    """Test LiftGammaGain with default settings (no change)."""
    node = LiftGammaGain(image=dummy_image)
    result = await node.process(context)
    assert isinstance(result, ImageRef)
    img = await context.image_to_pil(result)
    assert img.size == (100, 100)


@pytest.mark.asyncio
async def test_lift_gamma_gain_adjust_lift(context: ProcessingContext):
    """Test LiftGammaGain with adjusted lift (shadows)."""
    node = LiftGammaGain(
        image=dummy_image,
        lift_r=0.1,
        lift_g=0.05,
        lift_b=0.0,
        lift_master=0.05,
    )
    result = await node.process(context)
    assert isinstance(result, ImageRef)


@pytest.mark.asyncio
async def test_lift_gamma_gain_adjust_gamma(context: ProcessingContext):
    """Test LiftGammaGain with adjusted gamma (midtones)."""
    node = LiftGammaGain(
        image=dummy_image,
        gamma_r=1.2,
        gamma_g=1.0,
        gamma_b=0.9,
        gamma_master=1.1,
    )
    result = await node.process(context)
    assert isinstance(result, ImageRef)


@pytest.mark.asyncio
async def test_lift_gamma_gain_adjust_gain(context: ProcessingContext):
    """Test LiftGammaGain with adjusted gain (highlights)."""
    node = LiftGammaGain(
        image=dummy_image,
        gain_r=1.2,
        gain_g=1.1,
        gain_b=0.9,
        gain_master=1.05,
    )
    result = await node.process(context)
    assert isinstance(result, ImageRef)


@pytest.mark.asyncio
async def test_lift_gamma_gain_empty_image(context: ProcessingContext):
    """Test LiftGammaGain with empty image raises error."""
    node = LiftGammaGain(image=ImageRef())
    with pytest.raises(ValueError, match="Image is not connected"):
        await node.process(context)


@pytest.mark.asyncio
async def test_cdl_defaults(context: ProcessingContext):
    """Test CDL with default settings (no change)."""
    node = CDL(image=dummy_image)
    result = await node.process(context)
    assert isinstance(result, ImageRef)
    img = await context.image_to_pil(result)
    assert img.size == (100, 100)


@pytest.mark.asyncio
async def test_cdl_adjust_slope(context: ProcessingContext):
    """Test CDL with adjusted slope (multiplier)."""
    node = CDL(
        image=dummy_image,
        slope_r=1.2,
        slope_g=1.0,
        slope_b=0.9,
    )
    result = await node.process(context)
    assert isinstance(result, ImageRef)


@pytest.mark.asyncio
async def test_cdl_adjust_offset(context: ProcessingContext):
    """Test CDL with adjusted offset (addition)."""
    node = CDL(
        image=dummy_image,
        offset_r=0.1,
        offset_g=0.0,
        offset_b=-0.05,
    )
    result = await node.process(context)
    assert isinstance(result, ImageRef)


@pytest.mark.asyncio
async def test_cdl_adjust_power(context: ProcessingContext):
    """Test CDL with adjusted power (gamma)."""
    node = CDL(
        image=dummy_image,
        power_r=1.2,
        power_g=1.0,
        power_b=0.9,
    )
    result = await node.process(context)
    assert isinstance(result, ImageRef)


@pytest.mark.asyncio
async def test_cdl_adjust_saturation(context: ProcessingContext):
    """Test CDL with saturation adjustment."""
    node = CDL(
        image=dummy_image,
        saturation=1.5,
    )
    result = await node.process(context)
    assert isinstance(result, ImageRef)


@pytest.mark.asyncio
async def test_cdl_empty_image(context: ProcessingContext):
    """Test CDL with empty image raises error."""
    node = CDL(image=ImageRef())
    with pytest.raises(ValueError, match="Image is not connected"):
        await node.process(context)


@pytest.mark.asyncio
async def test_color_balance_defaults(context: ProcessingContext):
    """Test ColorBalance with default settings (no change)."""
    node = ColorBalance(image=dummy_image)
    result = await node.process(context)
    assert isinstance(result, ImageRef)
    img = await context.image_to_pil(result)
    assert img.size == (100, 100)


@pytest.mark.asyncio
async def test_color_balance_warm(context: ProcessingContext):
    """Test ColorBalance with warmer temperature."""
    node = ColorBalance(
        image=dummy_image,
        temperature=0.5,  # Warmer
    )
    result = await node.process(context)
    assert isinstance(result, ImageRef)


@pytest.mark.asyncio
async def test_color_balance_cool(context: ProcessingContext):
    """Test ColorBalance with cooler temperature."""
    node = ColorBalance(
        image=dummy_image,
        temperature=-0.5,  # Cooler
    )
    result = await node.process(context)
    assert isinstance(result, ImageRef)


@pytest.mark.asyncio
async def test_color_balance_tint(context: ProcessingContext):
    """Test ColorBalance with tint adjustment."""
    node = ColorBalance(
        image=dummy_image,
        tint=0.3,  # Magenta
    )
    result = await node.process(context)
    assert isinstance(result, ImageRef)


@pytest.mark.asyncio
async def test_color_balance_empty_image(context: ProcessingContext):
    """Test ColorBalance with empty image raises error."""
    node = ColorBalance(image=ImageRef())
    with pytest.raises(ValueError, match="Image is not connected"):
        await node.process(context)


@pytest.mark.asyncio
async def test_exposure_defaults(context: ProcessingContext):
    """Test Exposure with default settings (no change)."""
    node = Exposure(image=dummy_image)
    result = await node.process(context)
    assert isinstance(result, ImageRef)
    img = await context.image_to_pil(result)
    assert img.size == (100, 100)


@pytest.mark.asyncio
async def test_exposure_increase(context: ProcessingContext):
    """Test Exposure with increased exposure."""
    node = Exposure(
        image=dummy_image,
        exposure=1.0,  # +1 stop
    )
    result = await node.process(context)
    assert isinstance(result, ImageRef)


@pytest.mark.asyncio
async def test_exposure_decrease(context: ProcessingContext):
    """Test Exposure with decreased exposure."""
    node = Exposure(
        image=dummy_image,
        exposure=-1.0,  # -1 stop
    )
    result = await node.process(context)
    assert isinstance(result, ImageRef)


@pytest.mark.asyncio
async def test_exposure_contrast(context: ProcessingContext):
    """Test Exposure with contrast adjustment."""
    node = Exposure(
        image=dummy_image,
        contrast=0.5,
    )
    result = await node.process(context)
    assert isinstance(result, ImageRef)


@pytest.mark.asyncio
async def test_exposure_highlights_shadows(context: ProcessingContext):
    """Test Exposure with highlights and shadows adjustment."""
    node = Exposure(
        image=dummy_image,
        highlights=-0.3,
        shadows=0.3,
    )
    result = await node.process(context)
    assert isinstance(result, ImageRef)


@pytest.mark.asyncio
async def test_exposure_whites_blacks(context: ProcessingContext):
    """Test Exposure with whites and blacks adjustment."""
    node = Exposure(
        image=dummy_image,
        whites=0.2,
        blacks=-0.2,
    )
    result = await node.process(context)
    assert isinstance(result, ImageRef)


@pytest.mark.asyncio
async def test_exposure_empty_image(context: ProcessingContext):
    """Test Exposure with empty image raises error."""
    node = Exposure(image=ImageRef())
    with pytest.raises(ValueError, match="Image is not connected"):
        await node.process(context)


@pytest.mark.asyncio
async def test_saturation_vibrance_defaults(context: ProcessingContext):
    """Test SaturationVibrance with default settings (no change)."""
    node = SaturationVibrance(image=dummy_image)
    result = await node.process(context)
    assert isinstance(result, ImageRef)
    img = await context.image_to_pil(result)
    assert img.size == (100, 100)


@pytest.mark.asyncio
async def test_saturation_vibrance_increase_saturation(context: ProcessingContext):
    """Test SaturationVibrance with increased saturation."""
    node = SaturationVibrance(
        image=dummy_image,
        saturation=0.5,  # Range is -1 to 1, where 0 = no change
    )
    result = await node.process(context)
    assert isinstance(result, ImageRef)


@pytest.mark.asyncio
async def test_saturation_vibrance_decrease_saturation(context: ProcessingContext):
    """Test SaturationVibrance with decreased saturation (desaturated)."""
    node = SaturationVibrance(
        image=dummy_image,
        saturation=-1.0,  # Grayscale (range is -1 to 1)
    )
    result = await node.process(context)
    assert isinstance(result, ImageRef)


@pytest.mark.asyncio
async def test_saturation_vibrance_vibrance(context: ProcessingContext):
    """Test SaturationVibrance with vibrance adjustment."""
    node = SaturationVibrance(
        image=dummy_image,
        vibrance=0.5,
    )
    result = await node.process(context)
    assert isinstance(result, ImageRef)


@pytest.mark.asyncio
async def test_saturation_vibrance_empty_image(context: ProcessingContext):
    """Test SaturationVibrance with empty image raises error."""
    node = SaturationVibrance(image=ImageRef())
    with pytest.raises(ValueError, match="Image is not connected"):
        await node.process(context)
