"""
Design nodes for visual layout composition.
"""

import io
import logging
from typing import Any, ClassVar, Dict, Optional
from pydantic import Field

from nodetool.metadata.types import (
    ImageRef,
    LayoutCanvasData,
    LayoutElement,
)
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext

log = logging.getLogger(__name__)

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = None
    ImageDraw = None
    ImageFont = None


def hex_to_rgba(hex_color: str, opacity: float = 1.0) -> tuple:
    """Convert hex color string to RGBA tuple."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 6:
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        return (r, g, b, int(opacity * 255))
    elif len(hex_color) == 8:
        r, g, b, a = (
            int(hex_color[0:2], 16),
            int(hex_color[2:4], 16),
            int(hex_color[4:6], 16),
            int(hex_color[6:8], 16),
        )
        return (r, g, b, int(a * opacity))
    return (0, 0, 0, int(opacity * 255))


class LayoutCanvas(BaseNode):
    """
    Visual layout editor node for composing designs with text, images, and shapes.
    design, layout, canvas, image, compose, figma

    Use cases:
    - Create social media graphics with dynamic text
    - Design banners and thumbnails
    - Compose multi-element layouts
    - Create image templates with replaceable content
    """

    _is_dynamic: ClassVar[bool] = True

    canvas: LayoutCanvasData = Field(
        default_factory=LayoutCanvasData,
        description="Canvas layout data with elements and settings.",
    )

    @classmethod
    def get_title(cls):
        return "Layout Canvas"

    @classmethod
    def is_dynamic(cls) -> bool:
        return True

    def _apply_dynamic_properties(self) -> list:
        """Apply dynamic property values to elements based on exposed inputs."""
        import copy

        # Create a deep copy of elements to avoid mutating the original
        elements = copy.deepcopy(self.canvas.elements)

        # Build a mapping of elementId -> element for quick lookup
        element_map: Dict[str, Any] = {}
        for el in elements:
            element_map[el.id] = el

        # Apply dynamic properties based on exposed inputs
        for exposed in self.canvas.exposedInputs:
            input_name = exposed.inputName
            element_id = exposed.elementId
            prop_name = exposed.property

            # Get the dynamic property value
            value = self._dynamic_properties.get(input_name)
            if value is None:
                continue

            # Find the element
            element = element_map.get(element_id)
            if element is None:
                continue

            # Handle ImageRef values for image inputs
            if exposed.inputType == "image" and isinstance(value, ImageRef):
                value = value.uri or ""

            # Apply the value to the element's properties
            if hasattr(element, "properties") and isinstance(element.properties, dict):
                element.properties[prop_name] = value

        return elements

    async def process(self, context: ProcessingContext) -> ImageRef:
        """Render the canvas to an image."""
        log.info(f"LayoutCanvas.process started - canvas size: {self.canvas.width}x{self.canvas.height}")
        log.info(f"Elements count: {len(self.canvas.elements)}")
        log.info(f"Exposed inputs: {self.canvas.exposedInputs}")
        log.info(f"Dynamic properties: {self._dynamic_properties}")

        if Image is None:
            raise ImportError("PIL is required for LayoutCanvas node")

        # Create the base image
        img = Image.new(
            "RGBA",
            (self.canvas.width, self.canvas.height),
            hex_to_rgba(self.canvas.backgroundColor),
        )
        log.info(f"Created base image with background: {self.canvas.backgroundColor}")

        # Handle background image if present
        if self.canvas.backgroundImage:
            try:
                bg_ref = ImageRef(uri=self.canvas.backgroundImage)
                bg_bytes = await context.asset_to_bytes(bg_ref)
                bg_img = Image.open(io.BytesIO(bg_bytes)).convert("RGBA")
                bg_img = bg_img.resize((self.canvas.width, self.canvas.height))
                img.paste(bg_img, (0, 0))
            except Exception as e:
                log.warning(f"Failed to load background image: {e}")

        # Get elements with dynamic properties applied
        elements = self._apply_dynamic_properties()
        log.info(f"Elements after applying dynamic properties: {len(elements)}")

        # Sort elements by zIndex
        sorted_elements = sorted(elements, key=lambda e: e.zIndex)

        # Render each element
        for element in sorted_elements:
            if not element.visible:
                log.debug(f"Skipping hidden element: {element.name}")
                continue
            log.info(f"Rendering element: {element.name} (type={element.type})")
            await self._render_element(context, img, element)

        # Convert to bytes and create ImageRef
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)

        result = await context.image_from_bytes(buffer.read())
        log.info(f"LayoutCanvas.process completed - result: {result}")
        return result

    async def _render_element(
        self, context: ProcessingContext, img: Image.Image, element: LayoutElement
    ):
        """Render a single element onto the canvas."""
        props = element.properties
        element_type = element.type

        if element_type == "rectangle":
            await self._render_rectangle(img, element, props)
        elif element_type == "text":
            await self._render_text(img, element, props)
        elif element_type == "image":
            await self._render_image(context, img, element, props)
        elif element_type == "group":
            # Render children
            for child in element.children:
                if child.visible:
                    await self._render_element(context, img, child)

    async def _render_rectangle(
        self, img: Image.Image, element: LayoutElement, props: Dict[str, Any]
    ):
        """Render a rectangle element."""
        draw = ImageDraw.Draw(img, "RGBA")
        x, y = int(element.x), int(element.y)
        w, h = int(element.width), int(element.height)
        opacity = props.get("opacity", 1.0)
        fill_color = hex_to_rgba(props.get("fillColor", "#cccccc"), opacity)
        border_color = hex_to_rgba(props.get("borderColor", "#000000"), opacity)
        border_width = props.get("borderWidth", 0)

        # Draw filled rectangle
        draw.rectangle([x, y, x + w, y + h], fill=fill_color)

        # Draw border if present
        if border_width > 0:
            draw.rectangle(
                [x, y, x + w, y + h], outline=border_color, width=int(border_width)
            )

    async def _render_text(
        self, img: Image.Image, element: LayoutElement, props: Dict[str, Any]
    ):
        """Render a text element."""
        draw = ImageDraw.Draw(img, "RGBA")
        x, y = int(element.x), int(element.y)
        content = props.get("content", "")
        font_size = props.get("fontSize", 16)
        color = hex_to_rgba(props.get("color", "#000000"))

        # Try to load font, fall back to default
        try:
            font_family = props.get("fontFamily", "Arial")
            font = ImageFont.truetype(font_family, font_size)
        except (OSError, IOError):
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", font_size)
            except (OSError, IOError):
                font = ImageFont.load_default()

        draw.text((x, y), content, fill=color, font=font)

    async def _render_image(
        self,
        context: ProcessingContext,
        img: Image.Image,
        element: LayoutElement,
        props: Dict[str, Any],
    ):
        """Render an image element."""
        source = props.get("source", "")
        if not source:
            return

        try:
            # Load the image
            img_ref = ImageRef(uri=source)
            img_bytes = await context.asset_to_bytes(img_ref)
            element_img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")

            # Apply fit mode
            fit = props.get("fit", "contain")
            target_w, target_h = int(element.width), int(element.height)

            if fit == "fill":
                element_img = element_img.resize((target_w, target_h))
            elif fit == "cover":
                # Scale to cover, then crop
                scale = max(target_w / element_img.width, target_h / element_img.height)
                new_w = int(element_img.width * scale)
                new_h = int(element_img.height * scale)
                element_img = element_img.resize((new_w, new_h))
                # Crop center
                left = (new_w - target_w) // 2
                top = (new_h - target_h) // 2
                element_img = element_img.crop((left, top, left + target_w, top + target_h))
            else:  # contain
                # Scale to fit within bounds
                scale = min(target_w / element_img.width, target_h / element_img.height)
                new_w = int(element_img.width * scale)
                new_h = int(element_img.height * scale)
                element_img = element_img.resize((new_w, new_h))

            # Apply opacity
            opacity = props.get("opacity", 1.0)
            if opacity < 1.0:
                alpha = element_img.getchannel("A")
                alpha = alpha.point(lambda p: int(p * opacity))
                element_img.putalpha(alpha)

            # Paste onto canvas
            x, y = int(element.x), int(element.y)
            img.paste(element_img, (x, y), element_img)

        except Exception:
            # Draw placeholder on error
            draw = ImageDraw.Draw(img, "RGBA")
            x, y = int(element.x), int(element.y)
            w, h = int(element.width), int(element.height)
            draw.rectangle([x, y, x + w, y + h], fill=(200, 200, 200, 128))
            draw.line([x, y, x + w, y + h], fill=(100, 100, 100, 128), width=2)
            draw.line([x + w, y, x, y + h], fill=(100, 100, 100, 128), width=2)

