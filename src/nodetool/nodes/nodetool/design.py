"""
Design nodes for visual layout composition.
"""

import io
import json
import logging
import sys
from typing import Any, ClassVar, Dict
from pydantic import Field

from nodetool.metadata.types import (
    ImageRef,
    LayoutCanvasData,
    LayoutElement,
)
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext

log = logging.getLogger(__name__)


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


def generate_konva_html(canvas_data: dict, elements: list) -> str:
    """
    Generate HTML page with Konva.js that renders the canvas.
    
    This creates a standalone HTML page that:
    1. Loads Konva.js from CDN
    2. Creates a Stage with the canvas dimensions
    3. Renders all elements using Konva
    4. The page can be screenshotted by Playwright for pixel-perfect output
    """
    width = canvas_data.get("width", 800)
    height = canvas_data.get("height", 600)
    bg_color = canvas_data.get("backgroundColor", "#ffffff")
    
    # Convert elements to JSON for JavaScript
    elements_json = json.dumps(elements)
    
    # Collect unique font families from elements
    font_families = set()
    for el in elements:
        props = el.get("properties", {})
        if "fontFamily" in props:
            font_families.add(props["fontFamily"])
    
    # Map common font names to Google Fonts
    google_fonts_map = {
        "Arial": "Arial",  # System font, no need to load
        "Helvetica": "Arial",  # System font fallback
        "Times New Roman": "Times New Roman",  # System font
        "Georgia": "Georgia",  # System font
        "Courier New": "Courier New",  # System font
        "JetBrains Mono": "JetBrains+Mono",
        "Roboto": "Roboto",
        "Open Sans": "Open+Sans",
        "Lato": "Lato",
        "Montserrat": "Montserrat",
        "Poppins": "Poppins",
        "Inter": "Inter",
        "Playfair Display": "Playfair+Display",
        "Source Code Pro": "Source+Code+Pro",
        "Fira Code": "Fira+Code",
        "Ubuntu": "Ubuntu",
        "Nunito": "Nunito",
        "Raleway": "Raleway",
        "Oswald": "Oswald",
        "Merriweather": "Merriweather",
    }
    
    # Build Google Fonts URL for non-system fonts
    google_font_families = []
    for font in font_families:
        if font in google_fonts_map and font not in ["Arial", "Helvetica", "Times New Roman", "Georgia", "Courier New"]:
            google_font_families.append(google_fonts_map[font] + ":wght@400;700")
        elif font not in google_fonts_map:
            # Try to load it from Google Fonts anyway (replace spaces with +)
            google_font_families.append(font.replace(" ", "+") + ":wght@400;700")
    
    fonts_link = ""
    if google_font_families:
        # Google Fonts CSS2 API uses &family= for each font
        fonts_param = "&family=".join(google_font_families)
        fonts_link = f'<link href="https://fonts.googleapis.com/css2?family={fonts_param}&display=swap" rel="stylesheet">'
        log.debug(f"Loading fonts: {fonts_link}")
    
    html = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    {fonts_link}
    <style>
        * {{ margin: 0; padding: 0; }}
        body {{ overflow: hidden; }}
        #container {{ width: {width}px; height: {height}px; }}
    </style>
    <script src="https://unpkg.com/konva@9/konva.min.js"></script>
</head>
<body>
    <div id="container"></div>
    <script>
        const canvasData = {{
            width: {width},
            height: {height},
            backgroundColor: "{bg_color}"
        }};
        const elements = {elements_json};
        
        // Create stage
        const stage = new Konva.Stage({{
            container: 'container',
            width: canvasData.width,
            height: canvasData.height
        }});
        
        // Background layer
        const bgLayer = new Konva.Layer();
        stage.add(bgLayer);
        
        const bgRect = new Konva.Rect({{
            x: 0,
            y: 0,
            width: canvasData.width,
            height: canvasData.height,
            fill: canvasData.backgroundColor
        }});
        bgLayer.add(bgRect);
        
        // Elements layer
        const layer = new Konva.Layer();
        stage.add(layer);
        
        // Image loading promises
        const imagePromises = [];
        
        // Sort elements by zIndex
        const sortedElements = [...elements].sort((a, b) => a.zIndex - b.zIndex);
        
        // Render each element
        function renderElement(element) {{
            if (!element.visible) return null;
            
            const props = element.properties || {{}};
            const type = element.type;
            
            if (type === 'rectangle') {{
                const rect = new Konva.Rect({{
                    x: element.x,
                    y: element.y,
                    width: element.width,
                    height: element.height,
                    rotation: element.rotation || 0,
                    fill: props.fillColor || '#cccccc',
                    stroke: props.borderWidth > 0 ? (props.borderColor || '#000000') : undefined,
                    strokeWidth: props.borderWidth || 0,
                    cornerRadius: props.borderRadius || 0,
                    opacity: props.opacity !== undefined ? props.opacity : 1
                }});
                layer.add(rect);
                return rect;
            }}
            
            if (type === 'text') {{
                const text = new Konva.Text({{
                    x: element.x,
                    y: element.y,
                    width: element.width,
                    height: element.height,
                    rotation: element.rotation || 0,
                    text: props.content || '',
                    fontSize: props.fontSize || 16,
                    fontFamily: props.fontFamily || 'Arial',
                    fontStyle: props.fontWeight === 'bold' ? 'bold' : 'normal',
                    fill: props.color || '#000000',
                    align: props.alignment || 'left',
                    lineHeight: props.lineHeight || 1.2,
                    opacity: props.opacity !== undefined ? props.opacity : 1
                }});
                layer.add(text);
                return text;
            }}
            
            if (type === 'image') {{
                const source = props.source || '';
                if (!source) return null;
                
                const imagePromise = new Promise((resolve, reject) => {{
                    const imageObj = new Image();
                    imageObj.crossOrigin = 'anonymous';
                    imageObj.onload = function() {{
                        // Calculate fit
                        const fit = props.fit || 'contain';
                        let sx = 0, sy = 0, sw = imageObj.width, sh = imageObj.height;
                        let dx = element.x, dy = element.y, dw = element.width, dh = element.height;
                        let cropX = 0, cropY = 0, cropW = imageObj.width, cropH = imageObj.height;
                        
                        if (fit === 'cover') {{
                            const scale = Math.max(element.width / imageObj.width, element.height / imageObj.height);
                            const scaledW = imageObj.width * scale;
                            const scaledH = imageObj.height * scale;
                            cropX = (scaledW - element.width) / 2 / scale;
                            cropY = (scaledH - element.height) / 2 / scale;
                            cropW = element.width / scale;
                            cropH = element.height / scale;
                        }} else if (fit === 'contain') {{
                            const scale = Math.min(element.width / imageObj.width, element.height / imageObj.height);
                            dw = imageObj.width * scale;
                            dh = imageObj.height * scale;
                            dx = element.x + (element.width - dw) / 2;
                            dy = element.y + (element.height - dh) / 2;
                        }}
                        // fill: just use element dimensions directly
                        
                        const konvaImage = new Konva.Image({{
                            x: dx,
                            y: dy,
                            width: dw,
                            height: dh,
                            image: imageObj,
                            rotation: element.rotation || 0,
                            opacity: props.opacity !== undefined ? props.opacity : 1,
                            crop: fit === 'cover' ? {{
                                x: cropX,
                                y: cropY,
                                width: cropW,
                                height: cropH
                            }} : undefined
                        }});
                        layer.add(konvaImage);
                        resolve(konvaImage);
                    }};
                    imageObj.onerror = function(e) {{
                        console.error('Failed to load image:', source, e);
                        // Draw placeholder
                        const rect = new Konva.Rect({{
                            x: element.x,
                            y: element.y,
                            width: element.width,
                            height: element.height,
                            fill: '#cccccc',
                            opacity: 0.5
                        }});
                        layer.add(rect);
                        resolve(rect);
                    }};
                    imageObj.src = source;
                }});
                imagePromises.push(imagePromise);
                return null;
            }}
            
            if (type === 'group') {{
                // Groups are containers - render children
                const group = new Konva.Group({{
                    x: element.x,
                    y: element.y,
                    rotation: element.rotation || 0,
                    opacity: props.opacity !== undefined ? props.opacity : 1
                }});
                layer.add(group);
                
                if (element.children) {{
                    element.children.forEach(child => {{
                        const childNode = renderElement(child);
                        if (childNode) {{
                            group.add(childNode);
                        }}
                    }});
                }}
                return group;
            }}
            
            return null;
        }}
        
        // Render all elements
        sortedElements.forEach(el => renderElement(el));
        
        // Wait for fonts and images to load, then signal ready
        async function waitForReady() {{
            // Wait for fonts
            if (document.fonts && document.fonts.ready) {{
                await document.fonts.ready;
            }}
            
            // Wait for images
            await Promise.all(imagePromises);
            
            // Redraw layer to apply loaded fonts
            layer.draw();
            
            // Signal that rendering is complete
            window.renderComplete = true;
            document.body.setAttribute('data-render-complete', 'true');
        }}
        
        waitForReady().catch(err => {{
            console.error('Error during rendering:', err);
            window.renderComplete = true;
            document.body.setAttribute('data-render-complete', 'true');
        }});
        
        // Fallback timeout
        setTimeout(() => {{
            if (!window.renderComplete) {{
                window.renderComplete = true;
                document.body.setAttribute('data-render-complete', 'true');
            }}
        }}, 5000);
    </script>
</body>
</html>'''
    
    return html


class LayoutCanvas(BaseNode):
    """
    Visual layout editor node for composing designs with text, images, and shapes.
    Uses Playwright + Konva.js for pixel-perfect rendering that matches the editor.
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

    pixel_ratio: float = Field(
        default=2.0,
        ge=0.5,
        le=4.0,
        description="Pixel ratio for output quality. 2.0 = retina quality.",
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

    def _elements_to_dict(self, elements: list) -> list:
        """Convert LayoutElement objects to dictionaries for JSON serialization."""
        result = []
        for el in elements:
            el_dict = {
                "id": el.id,
                "type": el.type,
                "x": el.x,
                "y": el.y,
                "width": el.width,
                "height": el.height,
                "rotation": el.rotation,
                "zIndex": el.zIndex,
                "visible": el.visible,
                "locked": el.locked,
                "name": el.name,
                "properties": el.properties,
            }
            if hasattr(el, "children") and el.children:
                el_dict["children"] = self._elements_to_dict(el.children)
            result.append(el_dict)
        return result

    async def process(self, context: ProcessingContext) -> ImageRef:
        """Render the canvas to an image using Playwright + Konva.js."""
        import asyncio
        import subprocess
        import tempfile
        import os

        log.info(f"LayoutCanvas.process started - canvas size: {self.canvas.width}x{self.canvas.height}")
        log.info(f"Elements count: {len(self.canvas.elements)}")
        log.info(f"Exposed inputs: {self.canvas.exposedInputs}")
        log.info(f"Dynamic properties: {self._dynamic_properties}")

        # Get elements with dynamic properties applied
        elements = self._apply_dynamic_properties()
        elements_dict = self._elements_to_dict(elements)
        log.info(f"Elements after applying dynamic properties: {len(elements)}")

        # Generate HTML with Konva
        canvas_dict = {
            "width": self.canvas.width,
            "height": self.canvas.height,
            "backgroundColor": self.canvas.backgroundColor,
            "backgroundImage": self.canvas.backgroundImage,
        }
        html_content = generate_konva_html(canvas_dict, elements_dict)

        # Create temp files for HTML input and PNG output
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as html_file:
            html_file.write(html_content)
            html_path = html_file.name

        output_path = html_path.replace('.html', '.png')

        try:
            # Create a Python script to run Playwright in a separate process
            render_script = f'''
import sys
from playwright.sync_api import sync_playwright

html_path = sys.argv[1]
output_path = sys.argv[2]
width = int(sys.argv[3])
height = int(sys.argv[4])
pixel_ratio = float(sys.argv[5])

with sync_playwright() as pw:
    browser = pw.chromium.launch()
    ctx = browser.new_context(
        viewport={{"width": width, "height": height}},
        device_scale_factor=pixel_ratio
    )
    page = ctx.new_page()
    
    # Load the HTML file
    page.goto(f"file://{{html_path}}", wait_until="networkidle")
    
    # Wait for fonts to load
    try:
        page.wait_for_function("document.fonts.ready.then(() => true)", timeout=5000)
    except Exception as e:
        print(f"Font wait error: {{e}}", file=sys.stderr)
    
    # Extra wait for font rendering
    page.wait_for_timeout(500)
    
    # Wait for Konva to finish rendering
    try:
        page.wait_for_function("window.renderComplete === true", timeout=10000)
    except Exception as e:
        print(f"Render wait error: {{e}}", file=sys.stderr)
    
    # Extra wait after render complete
    page.wait_for_timeout(200)
    
    # Screenshot
    container = page.query_selector("#container")
    if container:
        container.screenshot(path=output_path)
    else:
        page.screenshot(path=output_path)
    
    browser.close()
'''
            
            # Run the script in a subprocess
            loop = asyncio.get_event_loop()
            
            def run_playwright_subprocess():
                result = subprocess.run(
                    [
                        sys.executable, '-c', render_script,
                        html_path, output_path,
                        str(self.canvas.width), str(self.canvas.height),
                        str(self.pixel_ratio)
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                if result.returncode != 0:
                    raise RuntimeError(f"Playwright subprocess failed: {result.stderr}")
                
                with open(output_path, 'rb') as f:
                    return f.read()
            
            # Run in executor to not block
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=1) as executor:
                screenshot_bytes = await loop.run_in_executor(executor, run_playwright_subprocess)

            # Convert to ImageRef
            result = await context.image_from_bytes(screenshot_bytes)
            log.info(f"LayoutCanvas.process completed - result: {result}")
            return result

        except subprocess.TimeoutExpired:
            log.error("Playwright rendering timed out")
            raise RuntimeError("Playwright rendering timed out after 30 seconds")
        except FileNotFoundError:
            log.error("Playwright is not installed or chromium browser not available")
            raise ImportError("Playwright is required. Install with: pip install playwright && playwright install chromium")
        except Exception as e:
            log.error(f"Playwright rendering failed: {e}")
            raise
        finally:
            # Cleanup temp files
            import os
            try:
                os.unlink(html_path)
            except:
                pass
            try:
                os.unlink(output_path)
            except:
                pass
