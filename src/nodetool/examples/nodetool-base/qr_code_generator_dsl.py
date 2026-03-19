"""
Batch QR Code Generator DSL Example

Create branded QR codes from URLs or short text snippets with automatic sizing,
color control, and workspace exports.

Workflow overview:
1. **Input capture** – Provide one or more target URLs/text snippets and style controls.
2. **URL encoding** – Sanitize and encode each payload for safe QR generation requests.
3. **Remote QR generation** – Build QuickChart API URLs and download PNG assets.
4. **Brand customization** – Add padding and captions with Pillow-based nodes.
5. **Workspace export** – Persist the finished images and return structured metadata.
"""

from nodetool.dsl.graph import create_graph, run_graph
from nodetool.workflows.processing_context import AssetOutputMode
from nodetool.dsl.nodetool.input import (
    StringListInput,
    IntegerInput,
    StringInput,
    ColorInput,
)
from nodetool.dsl.nodetool.control import ForEach, Collect
from nodetool.dsl.nodetool.text import FormatText
from nodetool.dsl.lib.urllib import QuoteURL
from nodetool.dsl.nodetool.dictionary import MakeDictionary
from nodetool.dsl.nodetool.list import MapField, GetElement
from nodetool.dsl.lib.math import Add, Multiply
from nodetool.dsl.lib.http import ImageDownloader
from nodetool.dsl.lib.pillow.draw import Background, RenderText
from nodetool.dsl.nodetool.image import Paste
from nodetool.dsl.nodetool.workspace import SaveImageFile
from nodetool.dsl.nodetool.output import DictionaryOutput
from nodetool.metadata.types import ColorRef


# --- Workflow Inputs ---------------------------------------------------------
qr_payloads = StringListInput(
    name="targets",
    description="URLs or text snippets to encode into QR codes.",
    value=[
        "https://example.com/welcome",
        "https://docs.nodetool.ai",
        "Contact us at support@example.com",
    ],
)

qr_size = IntegerInput(
    name="qr_size",
    description="QR code size in pixels (square output).",
    value=512,
)

qr_margin = IntegerInput(
    name="qr_margin",
    description="Quiet-zone margin supplied to the QuickChart QR endpoint.",
    value=2,
)

dark_modules = StringInput(
    name="dark_color",
    description="Hex color (without #) for QR modules.",
    value="111827",
)

light_modules = StringInput(
    name="light_color",
    description="Hex color (without #) for QR background.",
    value="F8FAFC",
)

canvas_color = ColorInput(
    name="canvas_color",
    description="Background color applied around the QR code.",
    value=ColorRef(type="color", value="#F8FAFC"),
)

text_color = ColorInput(
    name="caption_color",
    description="Text color used for QR captions.",
    value=ColorRef(type="color", value="#0F172A"),
)

padding = IntegerInput(
    name="padding",
    description="Padding (pixels) around each QR code inside the canvas.",
    value=32,
)

label_space = IntegerInput(
    name="label_space",
    description="Reserved vertical pixels for captions under each QR code.",
    value=96,
)

label_offset = IntegerInput(
    name="label_offset",
    description="Additional offset from the QR block to the caption baseline.",
    value=24,
)

label_font_size = IntegerInput(
    name="label_font_size",
    description="Font size for rendered captions.",
    value=28,
)

output_folder = StringInput(
    name="output_folder",
    description="Workspace folder for generated QR code images.",
    value="qr-codes",
)


# --- Derived layout math -----------------------------------------------------
double_padding = Multiply(a=padding.output, b=2)
canvas_width = Add(a=qr_size.output, b=double_padding.output)
qr_block_height = Add(a=qr_size.output, b=double_padding.output)
canvas_height = Add(a=qr_block_height.output, b=label_space.output)
text_baseline = Add(a=qr_size.output, b=padding.output)
text_y = Add(a=text_baseline.output, b=label_offset.output)


# --- QR request preparation --------------------------------------------------
request_iterator = ForEach(input_list=qr_payloads.output)

normalized_text = FormatText(
    template="{{ value|trim }}",
    value=request_iterator.out.output,
)

encoded_text = QuoteURL(text=normalized_text.output)

slug = FormatText(
    template="{{ value|trim|lower|replace(' ', '-')|replace('/', '-')|replace(':', '-')|replace('?', '') }}",
    value=normalized_text.output,
)

api_url = FormatText(
    template=(
        "https://quickchart.io/qr?text={{ data }}&size={{ size }}&margin={{ margin }}"
        "&light={{ light }}&dark={{ dark }}"
    ),
    data=encoded_text.output,
    size=qr_size.output,
    margin=qr_margin.output,
    light=light_modules.output,
    dark=dark_modules.output,
)

request_packet = MakeDictionary(
    original=normalized_text.output,
    slug=slug.output,
    url=api_url.output,
)

request_collector = Collect(input_item=request_packet.output)

requests = request_collector.out.output
request_urls = MapField(values=requests, field="url")
request_slugs = MapField(values=requests, field="slug")
request_labels = MapField(values=requests, field="original")


# --- QR download -------------------------------------------------------------
downloaded_images = ImageDownloader(images=request_urls.output)


# --- Image customization & export -------------------------------------------
image_iterator = ForEach(input_list=downloaded_images.out.images)

current_slug = GetElement(values=request_slugs.output, index=image_iterator.out.index)
original_label = GetElement(values=request_labels.output, index=image_iterator.out.index)

caption_text = FormatText(
    template="{{ text|upper }}",
    text=original_label.output,
)

filename = FormatText(template="{{ slug }}.png", slug=current_slug.output)

canvas = Background(
    width=canvas_width.output,
    height=canvas_height.output,
    color=canvas_color.output,
)

with_padding = Paste(
    image=canvas.output,
    paste=image_iterator.out.output,
    left=padding.output,
    top=padding.output,
)

captioned = RenderText(
    image=with_padding.output,
    text=caption_text.output,
    x=padding.output,
    y=text_y.output,
    size=label_font_size.output,
    color=text_color.output,
)

saved_image = SaveImageFile(
    image=captioned.output,
    folder=output_folder.output,
    filename=filename.output,
    overwrite=True,
)

saved_images = Collect(input_item=saved_image.output)


# --- Final output ------------------------------------------------------------
qr_summary = MakeDictionary(
    requests=requests,
    saved_images=saved_images.out.output,
    failed_downloads=downloaded_images.out.failed_urls,
)

output = DictionaryOutput(
    name="qr_code_batch",
    description="Metadata and workspace references for generated QR codes.",
    value=qr_summary.output,
)

graph = create_graph(output)


if __name__ == "__main__":
    result = run_graph(graph, asset_output_mode=AssetOutputMode.WORKSPACE)
    print("Generated QR codes:", result)
