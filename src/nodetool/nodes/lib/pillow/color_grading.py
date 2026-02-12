"""
Professional color grading nodes modeled after DaVinci Resolve, Nuke, and Fusion.

These nodes provide modular, industry-standard color correction tools that can be
combined in a node graph to achieve complex cinematic looks.
"""

from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
from pydantic import Field

from nodetool.metadata.types import ImageRef
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext

if TYPE_CHECKING:
    import PIL.Image


# =============================================================================
# Utility Functions
# =============================================================================


def _to_float_rgb(image: "PIL.Image.Image") -> np.ndarray:
    """Convert PIL image to float32 RGB array normalized to 0-1."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    return np.array(image, dtype=np.float32) / 255.0


def _from_float_rgb(arr: np.ndarray) -> "PIL.Image.Image":
    """Convert float32 array back to PIL Image."""
    import PIL.Image

    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    return PIL.Image.fromarray(arr)


def _get_luminance(arr: np.ndarray) -> np.ndarray:
    """Calculate Rec. 709 luminance from RGB array."""
    return 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]


# =============================================================================
# Lift/Gamma/Gain (Three-Way Color Corrector)
# =============================================================================


class LiftGammaGain(BaseNode):
    """
    Three-way color corrector for shadows, midtones, and highlights.
    lift, gamma, gain, color wheels, primary correction, shadows, midtones, highlights

    Use cases:
    - Apply the industry-standard three-way color correction
    - Balance colors across different tonal ranges
    - Create color contrast between shadows and highlights
    - Match footage from different sources

    Lift affects shadows, Gamma affects midtones, Gain affects highlights.
    Each control adjusts both luminance and color for its tonal range.
    """

    image: ImageRef = Field(
        default=ImageRef(), description="The image to color correct."
    )

    # Lift (shadows) - adds to dark values
    lift_r: float = Field(
        default=0.0, ge=-1.0, le=1.0, description="Red lift (shadow color shift)."
    )
    lift_g: float = Field(
        default=0.0, ge=-1.0, le=1.0, description="Green lift (shadow color shift)."
    )
    lift_b: float = Field(
        default=0.0, ge=-1.0, le=1.0, description="Blue lift (shadow color shift)."
    )
    lift_master: float = Field(
        default=0.0, ge=-1.0, le=1.0, description="Master lift (shadow brightness)."
    )

    # Gamma (midtones) - power function
    gamma_r: float = Field(
        default=1.0, ge=0.1, le=4.0, description="Red gamma (midtone adjustment)."
    )
    gamma_g: float = Field(
        default=1.0, ge=0.1, le=4.0, description="Green gamma (midtone adjustment)."
    )
    gamma_b: float = Field(
        default=1.0, ge=0.1, le=4.0, description="Blue gamma (midtone adjustment)."
    )
    gamma_master: float = Field(
        default=1.0, ge=0.1, le=4.0, description="Master gamma (overall midtones)."
    )

    # Gain (highlights) - multiplier
    gain_r: float = Field(
        default=1.0, ge=0.0, le=4.0, description="Red gain (highlight multiplier)."
    )
    gain_g: float = Field(
        default=1.0, ge=0.0, le=4.0, description="Green gain (highlight multiplier)."
    )
    gain_b: float = Field(
        default=1.0, ge=0.0, le=4.0, description="Blue gain (highlight multiplier)."
    )
    gain_master: float = Field(
        default=1.0, ge=0.0, le=4.0, description="Master gain (overall brightness)."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        if self.image.is_empty():
            raise ValueError("Image is not connected.")

        image = await context.image_to_pil(self.image)
        arr = _to_float_rgb(image)

        # Apply per-channel: output = (input * gain + lift) ^ (1/gamma)
        # Lift
        arr[:, :, 0] = arr[:, :, 0] + self.lift_r + self.lift_master
        arr[:, :, 1] = arr[:, :, 1] + self.lift_g + self.lift_master
        arr[:, :, 2] = arr[:, :, 2] + self.lift_b + self.lift_master

        # Gain
        arr[:, :, 0] = arr[:, :, 0] * self.gain_r * self.gain_master
        arr[:, :, 1] = arr[:, :, 1] * self.gain_g * self.gain_master
        arr[:, :, 2] = arr[:, :, 2] * self.gain_b * self.gain_master

        # Gamma (inverse power)
        arr = np.clip(arr, 0.0001, None)  # Avoid negative values for power
        arr[:, :, 0] = np.power(arr[:, :, 0], 1.0 / (self.gamma_r * self.gamma_master))
        arr[:, :, 1] = np.power(arr[:, :, 1], 1.0 / (self.gamma_g * self.gamma_master))
        arr[:, :, 2] = np.power(arr[:, :, 2], 1.0 / (self.gamma_b * self.gamma_master))

        return await context.image_from_pil(_from_float_rgb(arr))


# =============================================================================
# ASC CDL (Color Decision List)
# =============================================================================


class CDL(BaseNode):
    """
    ASC CDL (Color Decision List) color correction.
    cdl, slope, offset, power, saturation, asc, color decision list

    Use cases:
    - Apply industry-standard CDL color correction
    - Exchange color grades between different software
    - Apply precise mathematical color transformations
    - Create consistent looks across multiple shots

    Formula: output = (input * slope + offset) ^ power
    Followed by saturation adjustment.
    """

    image: ImageRef = Field(
        default=ImageRef(), description="The image to color correct."
    )

    # Slope (multiplier, similar to gain)
    slope_r: float = Field(
        default=1.0, ge=0.0, le=4.0, description="Red slope (multiplier)."
    )
    slope_g: float = Field(
        default=1.0, ge=0.0, le=4.0, description="Green slope (multiplier)."
    )
    slope_b: float = Field(
        default=1.0, ge=0.0, le=4.0, description="Blue slope (multiplier)."
    )

    # Offset (addition)
    offset_r: float = Field(
        default=0.0, ge=-1.0, le=1.0, description="Red offset (addition)."
    )
    offset_g: float = Field(
        default=0.0, ge=-1.0, le=1.0, description="Green offset (addition)."
    )
    offset_b: float = Field(
        default=0.0, ge=-1.0, le=1.0, description="Blue offset (addition)."
    )

    # Power (gamma)
    power_r: float = Field(
        default=1.0, ge=0.1, le=4.0, description="Red power (gamma)."
    )
    power_g: float = Field(
        default=1.0, ge=0.1, le=4.0, description="Green power (gamma)."
    )
    power_b: float = Field(
        default=1.0, ge=0.1, le=4.0, description="Blue power (gamma)."
    )

    # Saturation
    saturation: float = Field(
        default=1.0, ge=0.0, le=4.0, description="Saturation adjustment."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        if self.image.is_empty():
            raise ValueError("Image is not connected.")

        image = await context.image_to_pil(self.image)
        arr = _to_float_rgb(image)

        # Apply CDL formula: out = (in * slope + offset) ^ power
        arr[:, :, 0] = arr[:, :, 0] * self.slope_r + self.offset_r
        arr[:, :, 1] = arr[:, :, 1] * self.slope_g + self.offset_g
        arr[:, :, 2] = arr[:, :, 2] * self.slope_b + self.offset_b

        arr = np.clip(arr, 0.0001, None)
        arr[:, :, 0] = np.power(arr[:, :, 0], self.power_r)
        arr[:, :, 1] = np.power(arr[:, :, 1], self.power_g)
        arr[:, :, 2] = np.power(arr[:, :, 2], self.power_b)

        # Apply saturation
        if self.saturation != 1.0:
            lum = _get_luminance(arr)[:, :, np.newaxis]
            arr = lum + (arr - lum) * self.saturation

        return await context.image_from_pil(_from_float_rgb(arr))


# =============================================================================
# Color Balance (Temperature/Tint)
# =============================================================================


class ColorBalance(BaseNode):
    """
    Adjust color temperature and tint for white balance correction.
    white balance, temperature, tint, color balance, warm, cool

    Use cases:
    - Correct white balance in photos and video
    - Warm up or cool down the overall image
    - Fix color casts from mixed lighting
    - Create mood through color temperature shifts
    """

    image: ImageRef = Field(default=ImageRef(), description="The image to adjust.")

    temperature: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Color temperature. Positive = warmer (orange), negative = cooler (blue).",
    )

    tint: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Color tint. Positive = magenta, negative = green.",
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        if self.image.is_empty():
            raise ValueError("Image is not connected.")

        image = await context.image_to_pil(self.image)
        arr = _to_float_rgb(image)

        # Temperature: shift red-blue balance
        temp_shift = self.temperature * 0.3
        arr[:, :, 0] = arr[:, :, 0] + temp_shift  # Red
        arr[:, :, 2] = arr[:, :, 2] - temp_shift  # Blue

        # Tint: shift green-magenta balance
        tint_shift = self.tint * 0.3
        arr[:, :, 1] = arr[:, :, 1] - tint_shift  # Green
        arr[:, :, 0] = arr[:, :, 0] + tint_shift * 0.5  # Add some red for magenta
        arr[:, :, 2] = arr[:, :, 2] + tint_shift * 0.5  # Add some blue for magenta

        return await context.image_from_pil(_from_float_rgb(arr))


# =============================================================================
# Exposure (Exposure, Contrast, Highlights, Shadows, Whites, Blacks)
# =============================================================================


class Exposure(BaseNode):
    """
    Comprehensive tonal exposure controls similar to Lightroom/Camera Raw.
    exposure, contrast, highlights, shadows, whites, blacks, tonal

    Use cases:
    - Correct over/underexposed images
    - Recover highlight and shadow detail
    - Adjust overall contrast and tonal range
    - Fine-tune the brightness of specific tonal regions
    """

    image: ImageRef = Field(default=ImageRef(), description="The image to adjust.")

    exposure: float = Field(
        default=0.0,
        ge=-5.0,
        le=5.0,
        description="Exposure adjustment in stops. Affects entire image.",
    )

    contrast: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Contrast adjustment. Affects midtone separation.",
    )

    highlights: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Highlight recovery/boost. Affects brightest areas.",
    )

    shadows: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Shadow recovery/darken. Affects darkest areas.",
    )

    whites: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="White point adjustment. Sets the brightest white.",
    )

    blacks: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Black point adjustment. Sets the darkest black.",
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        if self.image.is_empty():
            raise ValueError("Image is not connected.")

        image = await context.image_to_pil(self.image)
        arr = _to_float_rgb(image)

        # Exposure (power of 2 for stops)
        if self.exposure != 0.0:
            arr = arr * (2.0**self.exposure)

        # Calculate luminance for tonal masks
        lum = _get_luminance(arr)

        # Highlights (affect values > 0.5)
        if self.highlights != 0.0:
            highlight_mask = np.clip((lum - 0.5) * 2, 0, 1)[:, :, np.newaxis]
            arr = arr - highlight_mask * self.highlights * 0.5

        # Shadows (affect values < 0.5)
        if self.shadows != 0.0:
            shadow_mask = np.clip((0.5 - lum) * 2, 0, 1)[:, :, np.newaxis]
            arr = arr + shadow_mask * self.shadows * 0.5

        # Whites (top end expansion/compression)
        if self.whites != 0.0:
            arr = arr + self.whites * 0.2 * arr  # Proportional adjustment

        # Blacks (bottom end expansion/compression)
        if self.blacks != 0.0:
            arr = arr + self.blacks * 0.2 * (1.0 - arr)  # Inverse proportional

        # Contrast (S-curve around midpoint)
        if self.contrast != 0.0:
            # Soft contrast using sigmoid-like curve
            midpoint = 0.5
            contrast_factor = 1.0 + self.contrast
            arr = midpoint + (arr - midpoint) * contrast_factor

        return await context.image_from_pil(_from_float_rgb(arr))


# =============================================================================
# Saturation and Vibrance
# =============================================================================


class SaturationVibrance(BaseNode):
    """
    Adjust color saturation with vibrance protection for skin tones.
    saturation, vibrance, color intensity, skin tones

    Use cases:
    - Boost color intensity without clipping
    - Protect skin tones while increasing saturation
    - Create desaturated or oversaturated looks
    - Fine-tune color intensity independently
    """

    image: ImageRef = Field(default=ImageRef(), description="The image to adjust.")

    saturation: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Global saturation. 0 = no change, -1 = grayscale, 1 = 2x saturation.",
    )

    vibrance: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Smart saturation that protects already-saturated colors and skin tones.",
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        if self.image.is_empty():
            raise ValueError("Image is not connected.")

        image = await context.image_to_pil(self.image)
        arr = _to_float_rgb(image)

        lum = _get_luminance(arr)[:, :, np.newaxis]

        # Apply saturation
        if self.saturation != 0.0:
            sat_factor = 1.0 + self.saturation
            arr = lum + (arr - lum) * sat_factor

        # Apply vibrance (smart saturation)
        if self.vibrance != 0.0:
            # Calculate current saturation per pixel
            max_rgb = np.max(arr, axis=2, keepdims=True)
            min_rgb = np.min(arr, axis=2, keepdims=True)
            current_sat = np.where(
                max_rgb > 0, (max_rgb - min_rgb) / (max_rgb + 0.001), 0
            )

            # Vibrance affects less-saturated colors more
            vibrance_mask = 1.0 - current_sat
            vibrance_factor = 1.0 + self.vibrance * vibrance_mask
            arr = lum + (arr - lum) * vibrance_factor

        return await context.image_from_pil(_from_float_rgb(arr))


# =============================================================================
# HSL Adjust (Per-Color Adjustments)
# =============================================================================


class HSLAdjust(BaseNode):
    """
    Adjust hue, saturation, and luminance for specific color ranges.
    hsl, hue, saturation, luminance, selective color, color range

    Use cases:
    - Shift specific colors (e.g., make blues more cyan)
    - Desaturate or boost individual color ranges
    - Brighten or darken specific colors
    - Create color-specific looks (teal skies, orange skin)
    """

    class ColorRange(str, Enum):
        ALL = "all"
        REDS = "reds"
        ORANGES = "oranges"
        YELLOWS = "yellows"
        GREENS = "greens"
        CYANS = "cyans"
        BLUES = "blues"
        PURPLES = "purples"
        MAGENTAS = "magentas"

    image: ImageRef = Field(default=ImageRef(), description="The image to adjust.")

    color_range: ColorRange = Field(
        default=ColorRange.ALL,
        description="The color range to adjust.",
    )

    hue_shift: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Hue shift for the selected color range. -1 to 1 = -180 to +180 degrees.",
    )

    saturation: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Saturation adjustment for the selected color range.",
    )

    luminance: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Luminance adjustment for the selected color range.",
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        if self.image.is_empty():
            raise ValueError("Image is not connected.")

        from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

        image = await context.image_to_pil(self.image)
        arr = _to_float_rgb(image)
        arr = np.clip(arr, 0, 1)

        # Vectorized RGB -> HSV conversion (operates on entire array)
        hsv = rgb_to_hsv(arr)
        h_ch = hsv[:, :, 0]
        s_ch = hsv[:, :, 1]
        v_ch = hsv[:, :, 2]

        is_all = self.color_range == self.ColorRange.ALL

        if is_all:
            blend = np.ones_like(h_ch)
        else:
            # Define hue ranges (0-1 scale, where 0 and 1 are red)
            hue_ranges = {
                self.ColorRange.REDS: (0.95, 0.05),  # Wraps around
                self.ColorRange.ORANGES: (0.02, 0.10),
                self.ColorRange.YELLOWS: (0.10, 0.18),
                self.ColorRange.GREENS: (0.18, 0.45),
                self.ColorRange.CYANS: (0.45, 0.55),
                self.ColorRange.BLUES: (0.55, 0.72),
                self.ColorRange.PURPLES: (0.72, 0.83),
                self.ColorRange.MAGENTAS: (0.83, 0.95),
            }

            hue_start, hue_end = hue_ranges[self.color_range]

            # Build in-range mask
            if hue_start > hue_end:  # Wraps around (e.g., reds)
                in_range = (h_ch >= hue_start) | (h_ch <= hue_end)
                center = (hue_start + hue_end + 1) / 2 % 1
                range_width = 1 - hue_start + hue_end
            else:
                in_range = (h_ch >= hue_start) & (h_ch <= hue_end)
                center = (hue_start + hue_end) / 2
                range_width = hue_end - hue_start

            # Only adjust sufficiently saturated pixels
            in_range = in_range & (s_ch > 0.1)

            # Calculate blend factor based on distance from range center
            dist = np.minimum(np.abs(h_ch - center), 1 - np.abs(h_ch - center))
            blend = np.maximum(0, 1 - dist / (range_width / 2 + 0.01))
            blend = np.where(in_range, blend, 0.0)

        # Apply adjustments
        new_h = (h_ch + self.hue_shift * 0.5 * blend) % 1.0
        new_s = np.clip(s_ch * (1 + self.saturation * blend), 0, 1)
        new_v = np.clip(v_ch * (1 + self.luminance * blend), 0, 1)

        new_hsv = np.stack([new_h, new_s, new_v], axis=2)
        result = hsv_to_rgb(new_hsv)

        return await context.image_from_pil(_from_float_rgb(result))


# =============================================================================
# Curves
# =============================================================================


class Curves(BaseNode):
    """
    RGB curves adjustment with control points for precise tonal control.
    curves, rgb, tonal, contrast, levels

    Use cases:
    - Create custom contrast curves
    - Adjust specific tonal ranges precisely
    - Create cross-processed or stylized looks
    - Match the tonal characteristics of film stocks
    """

    image: ImageRef = Field(default=ImageRef(), description="The image to adjust.")

    # Black and white points
    black_point: float = Field(
        default=0.0,
        ge=0.0,
        le=0.5,
        description="Input black point (lifts shadows).",
    )

    white_point: float = Field(
        default=1.0,
        ge=0.5,
        le=1.0,
        description="Input white point (compresses highlights).",
    )

    # Shadows, midtones, highlights control
    shadows: float = Field(
        default=0.0,
        ge=-0.5,
        le=0.5,
        description="Shadow curve adjustment.",
    )

    midtones: float = Field(
        default=0.0,
        ge=-0.5,
        le=0.5,
        description="Midtone curve adjustment (gamma).",
    )

    highlights: float = Field(
        default=0.0,
        ge=-0.5,
        le=0.5,
        description="Highlight curve adjustment.",
    )

    # Per-channel adjustments
    red_midtones: float = Field(
        default=0.0,
        ge=-0.5,
        le=0.5,
        description="Red channel midtone adjustment.",
    )

    green_midtones: float = Field(
        default=0.0,
        ge=-0.5,
        le=0.5,
        description="Green channel midtone adjustment.",
    )

    blue_midtones: float = Field(
        default=0.0,
        ge=-0.5,
        le=0.5,
        description="Blue channel midtone adjustment.",
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        if self.image.is_empty():
            raise ValueError("Image is not connected.")

        image = await context.image_to_pil(self.image)
        arr = _to_float_rgb(image)

        # Apply input levels (black/white point)
        arr = (arr - self.black_point) / (self.white_point - self.black_point)
        arr = np.clip(arr, 0, 1)

        # Apply tonal adjustments using soft curves
        # Shadows (affect lower values more)
        if self.shadows != 0.0:
            shadow_curve = 1.0 - arr  # Inverted for shadow weighting
            arr = arr + self.shadows * shadow_curve * arr

        # Midtones (gamma-like, affect middle values most)
        if self.midtones != 0.0:
            gamma = 1.0 / (1.0 + self.midtones)
            arr = np.power(arr, gamma)

        # Highlights (affect upper values more)
        if self.highlights != 0.0:
            highlight_curve = arr  # Direct weighting for highlights
            arr = arr + self.highlights * highlight_curve * (1.0 - arr)

        # Per-channel midtone adjustments
        if self.red_midtones != 0.0:
            gamma_r = 1.0 / (1.0 + self.red_midtones)
            arr[:, :, 0] = np.power(np.clip(arr[:, :, 0], 0.0001, 1), gamma_r)

        if self.green_midtones != 0.0:
            gamma_g = 1.0 / (1.0 + self.green_midtones)
            arr[:, :, 1] = np.power(np.clip(arr[:, :, 1], 0.0001, 1), gamma_g)

        if self.blue_midtones != 0.0:
            gamma_b = 1.0 / (1.0 + self.blue_midtones)
            arr[:, :, 2] = np.power(np.clip(arr[:, :, 2], 0.0001, 1), gamma_b)

        return await context.image_from_pil(_from_float_rgb(arr))


# =============================================================================
# Vignette
# =============================================================================


class Vignette(BaseNode):
    """
    Apply cinematic vignette effect to darken or lighten image edges.
    vignette, edge, darken, focus, cinematic

    Use cases:
    - Draw attention to the center of the image
    - Create a classic cinematic look
    - Simulate lens light falloff
    - Add subtle framing to photos
    """

    image: ImageRef = Field(
        default=ImageRef(), description="The image to apply vignette to."
    )

    amount: float = Field(
        default=0.5,
        ge=-1.0,
        le=1.0,
        description="Vignette amount. Positive darkens edges, negative lightens.",
    )

    midpoint: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Distance from center where vignette begins (0=center, 1=edges).",
    )

    roundness: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Shape of vignette. 0=oval matching image aspect, 1=circular, -1=rectangular.",
    )

    feather: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Softness of the vignette edge.",
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        if self.image.is_empty():
            raise ValueError("Image is not connected.")

        image = await context.image_to_pil(self.image)
        arr = _to_float_rgb(image)
        h, w = arr.shape[:2]

        # Create coordinate grids
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h / 2, w / 2

        # Normalize coordinates to -1 to 1
        norm_x = (x - center_x) / (w / 2)
        norm_y = (y - center_y) / (h / 2)

        # Adjust for roundness (aspect ratio compensation)
        aspect = w / h
        if self.roundness > 0:
            # Move toward circular
            norm_x = norm_x * (1 + (1 / aspect - 1) * self.roundness)
        elif self.roundness < 0:
            # Move toward rectangular (use max instead of euclidean)
            pass  # Handled below

        # Calculate distance from center
        if self.roundness < 0:
            # Rectangular falloff
            rect_blend = -self.roundness
            euclidean = np.sqrt(norm_x**2 + norm_y**2)
            rectangular = np.maximum(np.abs(norm_x), np.abs(norm_y))
            dist = euclidean * (1 - rect_blend) + rectangular * rect_blend
        else:
            dist = np.sqrt(norm_x**2 + norm_y**2)

        # Apply midpoint and feather
        feather_amount = max(self.feather, 0.01)  # Avoid division by zero
        vignette = (dist - self.midpoint) / feather_amount
        vignette = np.clip(vignette, 0, 1)

        # Apply amount (positive darkens, negative lightens)
        if self.amount > 0:
            vignette_factor = 1.0 - vignette * self.amount
        else:
            vignette_factor = 1.0 + vignette * (-self.amount)

        arr = arr * vignette_factor[:, :, np.newaxis]

        return await context.image_from_pil(_from_float_rgb(arr))


# =============================================================================
# Film Look (Preset-based cinematic grades)
# =============================================================================


class FilmLookPreset(str, Enum):
    """Predefined cinematic film looks."""

    TEAL_ORANGE = "teal_orange"
    BLOCKBUSTER = "blockbuster"
    NOIR = "noir"
    VINTAGE = "vintage"
    COLD_BLUE = "cold_blue"
    WARM_SUNSET = "warm_sunset"
    MATRIX = "matrix"
    BLEACH_BYPASS = "bleach_bypass"
    CROSS_PROCESS = "cross_process"
    FADED_FILM = "faded_film"


class FilmLook(BaseNode):
    """
    Apply preset cinematic film looks with adjustable intensity.
    film look, cinematic, preset, movie, lut, color grade

    Use cases:
    - Quickly apply popular cinematic color grades
    - Create consistent looks across multiple images
    - Emulate classic film stock characteristics
    - Starting point for custom color grading
    """

    image: ImageRef = Field(
        default=ImageRef(), description="The image to apply the film look to."
    )

    preset: FilmLookPreset = Field(
        default=FilmLookPreset.TEAL_ORANGE,
        description="The cinematic look to apply.",
    )

    intensity: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Intensity of the effect. 0=none, 1=full, 2=exaggerated.",
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        if self.image.is_empty():
            raise ValueError("Image is not connected.")

        image = await context.image_to_pil(self.image)
        arr = _to_float_rgb(image)
        original = arr.copy()

        # Define preset parameters
        # Format: (shadow_color, highlight_color, contrast, saturation, fade)
        presets = {
            FilmLookPreset.TEAL_ORANGE: (
                np.array([0.0, 0.5, 0.55]),  # Teal shadows
                np.array([1.0, 0.78, 0.6]),  # Orange highlights
                1.1,
                1.1,
                0.0,
            ),
            FilmLookPreset.BLOCKBUSTER: (
                np.array([0.12, 0.24, 0.35]),  # Blue shadows
                np.array([1.0, 0.9, 0.78]),  # Warm highlights
                1.2,
                1.0,
                0.0,
            ),
            FilmLookPreset.NOIR: (
                np.array([0.2, 0.2, 0.24]),  # Cool dark shadows
                np.array([0.86, 0.86, 0.9]),  # Cool highlights
                1.4,
                0.3,
                0.0,
            ),
            FilmLookPreset.VINTAGE: (
                np.array([0.4, 0.32, 0.24]),  # Warm brown shadows
                np.array([1.0, 0.94, 0.78]),  # Cream highlights
                0.9,
                0.8,
                0.15,
            ),
            FilmLookPreset.COLD_BLUE: (
                np.array([0.16, 0.24, 0.4]),  # Deep blue shadows
                np.array([0.78, 0.86, 1.0]),  # Icy highlights
                1.1,
                0.9,
                0.0,
            ),
            FilmLookPreset.WARM_SUNSET: (
                np.array([0.47, 0.24, 0.16]),  # Red-brown shadows
                np.array([1.0, 0.86, 0.7]),  # Warm highlights
                1.0,
                1.3,
                0.0,
            ),
            FilmLookPreset.MATRIX: (
                np.array([0.0, 0.16, 0.0]),  # Green shadows
                np.array([0.6, 1.0, 0.6]),  # Green highlights
                1.3,
                0.7,
                0.0,
            ),
            FilmLookPreset.BLEACH_BYPASS: (
                np.array([0.24, 0.24, 0.27]),  # Cool shadows
                np.array([0.94, 0.94, 0.98]),  # Cool highlights
                1.5,
                0.5,
                0.05,
            ),
            FilmLookPreset.CROSS_PROCESS: (
                np.array([0.0, 0.2, 0.3]),  # Cyan shadows
                np.array([1.0, 0.9, 0.5]),  # Yellow highlights
                1.2,
                1.2,
                0.0,
            ),
            FilmLookPreset.FADED_FILM: (
                np.array([0.3, 0.28, 0.26]),  # Warm gray shadows
                np.array([0.95, 0.92, 0.88]),  # Cream highlights
                0.85,
                0.7,
                0.2,
            ),
        }

        shadow_color, highlight_color, contrast, saturation, fade = presets[self.preset]

        # Calculate luminance for shadow/highlight masks
        lum = _get_luminance(arr)
        shadow_mask = np.clip(1.0 - lum * 2, 0, 1)[:, :, np.newaxis]
        highlight_mask = np.clip(lum * 2 - 1, 0, 1)[:, :, np.newaxis]

        # Apply shadow and highlight tinting
        arr = arr + shadow_mask * (shadow_color - 0.5) * 0.3
        arr = arr + highlight_mask * (highlight_color - 0.5) * 0.3

        # Apply contrast
        arr = (arr - 0.5) * contrast + 0.5

        # Apply saturation
        lum_3d = _get_luminance(arr)[:, :, np.newaxis]
        arr = lum_3d + (arr - lum_3d) * saturation

        # Apply fade (lift blacks)
        if fade > 0:
            arr = arr * (1.0 - fade) + fade * 0.15

        # Blend with original based on intensity
        arr = original + (arr - original) * self.intensity

        return await context.image_from_pil(_from_float_rgb(arr))


# =============================================================================
# Split Toning
# =============================================================================


class SplitToning(BaseNode):
    """
    Apply different color tints to shadows and highlights.
    split toning, shadows, highlights, tint, duotone

    Use cases:
    - Create classic teal and orange looks
    - Add color contrast between shadows and highlights
    - Emulate film processing techniques
    - Create stylized color-graded images
    """

    image: ImageRef = Field(
        default=ImageRef(), description="The image to apply split toning to."
    )

    # Shadow tint
    shadow_hue: float = Field(
        default=200.0,
        ge=0.0,
        le=360.0,
        description="Hue of shadow tint in degrees (0=red, 120=green, 240=blue).",
    )

    shadow_saturation: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Saturation of shadow tint.",
    )

    # Highlight tint
    highlight_hue: float = Field(
        default=40.0,
        ge=0.0,
        le=360.0,
        description="Hue of highlight tint in degrees.",
    )

    highlight_saturation: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Saturation of highlight tint.",
    )

    # Balance
    balance: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Balance between shadows (-1) and highlights (+1).",
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        if self.image.is_empty():
            raise ValueError("Image is not connected.")

        import colorsys

        image = await context.image_to_pil(self.image)
        arr = _to_float_rgb(image)

        # Convert hues to RGB colors
        shadow_r, shadow_g, shadow_b = colorsys.hsv_to_rgb(
            self.shadow_hue / 360.0, 1.0, 1.0
        )
        shadow_color = np.array([shadow_r, shadow_g, shadow_b])

        highlight_r, highlight_g, highlight_b = colorsys.hsv_to_rgb(
            self.highlight_hue / 360.0, 1.0, 1.0
        )
        highlight_color = np.array([highlight_r, highlight_g, highlight_b])

        # Calculate luminance for masks
        lum = _get_luminance(arr)

        # Adjust balance
        balance_offset = self.balance * 0.25
        shadow_mask = np.clip((0.5 + balance_offset - lum) * 2, 0, 1)[:, :, np.newaxis]
        highlight_mask = np.clip((lum - 0.5 + balance_offset) * 2, 0, 1)[
            :, :, np.newaxis
        ]

        # Apply tints
        arr = arr + shadow_mask * (shadow_color - 0.5) * self.shadow_saturation
        arr = arr + highlight_mask * (highlight_color - 0.5) * self.highlight_saturation

        return await context.image_from_pil(_from_float_rgb(arr))
