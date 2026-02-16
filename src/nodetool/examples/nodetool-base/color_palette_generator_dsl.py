"""
Color Palette Generator DSL Example

Automate color harmony creation starting from a single brand seed color.

Workflow Overview:
1. Capture a base color input for the palette seed.
2. Generate harmonious variants (tint, shade, complement, accent, muted neutral).
3. Compose a palette preview strip using Pillow image compositing nodes.
4. Produce design-ready JSON metadata and CSS custom properties.
5. Persist the CSS/JSON artifacts in the workspace for downstream design tools.

Demonstrates:
- Pillow-based color manipulation nodes for automated swatch generation.
- Jinja-powered templating for CSS/JSON output via `nodetool.text`.
- Workspace file emission for sharing palettes with external tooling.
"""

from __future__ import annotations

from nodetool.dsl.graph import create_graph, run_graph
from nodetool.dsl.lib.pillow import Blend
from nodetool.dsl.lib.pillow.draw import Background
from nodetool.dsl.lib.pillow.filter import Invert
from nodetool.dsl.nodetool.data import JSONToDataframe
from nodetool.dsl.nodetool.dictionary import MakeDictionary
from nodetool.dsl.nodetool.image import Paste
from nodetool.dsl.nodetool.input import ColorInput
from nodetool.dsl.nodetool.output import (
    DataframeOutput,
    DictionaryOutput,
    ImageOutput,
)
from nodetool.dsl.nodetool.text import FormatText, ParseJSON
from nodetool.dsl.nodetool.workspace import SaveImageFile, WriteTextFile
from nodetool.metadata.types import ColorRef
from nodetool.workflows.processing_context import AssetOutputMode


# ---------------------------------------------------------------------------
# Configuration constants for swatch generation
SWATCH_WIDTH = 200
SWATCH_HEIGHT = 200
GUTTER = 20
PALETTE_CANVAS_WIDTH = 20 + 6 * (SWATCH_WIDTH + GUTTER)
PALETTE_CANVAS_HEIGHT = SWATCH_HEIGHT + (GUTTER * 2)

TINT_ALPHA = 0.35
SHADE_ALPHA = 0.35
ACCENT_ALPHA = 0.4
MUTED_ALPHA = 0.5
ACCENT_MIX_COLOR = "#FFB347"
MUTED_MIX_COLOR = "#ECEFF1"


# ---------------------------------------------------------------------------
# Step 1: Base color input
base_color = ColorInput(
    name="base_brand_color",
    description="Primary brand seed color for palette generation",
    value=ColorRef(value="#3498DB"),
)


# ---------------------------------------------------------------------------
# Step 2: Generate image swatches via Pillow nodes
base_swatch = Background(
    width=SWATCH_WIDTH,
    height=SWATCH_HEIGHT,
    color=base_color.output,
)

white_background = Background(
    width=SWATCH_WIDTH,
    height=SWATCH_HEIGHT,
    color=ColorRef(value="#FFFFFF"),
)

black_background = Background(
    width=SWATCH_WIDTH,
    height=SWATCH_HEIGHT,
    color=ColorRef(value="#000000"),
)

accent_target = Background(
    width=SWATCH_WIDTH,
    height=SWATCH_HEIGHT,
    color=ColorRef(value=ACCENT_MIX_COLOR),
)

muted_target = Background(
    width=SWATCH_WIDTH,
    height=SWATCH_HEIGHT,
    color=ColorRef(value=MUTED_MIX_COLOR),
)

tint_swatch = Blend(
    image1=base_swatch.output,
    image2=white_background.output,
    alpha=TINT_ALPHA,
)

shade_swatch = Blend(
    image1=base_swatch.output,
    image2=black_background.output,
    alpha=SHADE_ALPHA,
)

accent_swatch = Blend(
    image1=base_swatch.output,
    image2=accent_target.output,
    alpha=ACCENT_ALPHA,
)

muted_swatch = Blend(
    image1=base_swatch.output,
    image2=muted_target.output,
    alpha=MUTED_ALPHA,
)

complement_swatch = Invert(image=base_swatch.output)


# ---------------------------------------------------------------------------
# Step 3: Assemble palette preview strip
palette_canvas = Background(
    width=PALETTE_CANVAS_WIDTH,
    height=PALETTE_CANVAS_HEIGHT,
    color=ColorRef(value="#F7F9FC"),
)

swatch_sequence = [
    base_swatch.output,
    tint_swatch.output,
    accent_swatch.output,
    complement_swatch.output,
    muted_swatch.output,
    shade_swatch.output,
]

composed_preview = palette_canvas.output
for index, swatch in enumerate(swatch_sequence):
    composed_preview = Paste(
        image=composed_preview,
        paste=swatch,
        left=GUTTER + index * (SWATCH_WIDTH + GUTTER),
        top=GUTTER,
    ).output


# Persist a high-resolution palette strip in the workspace
palette_preview_file = SaveImageFile(
    image=composed_preview,
    folder="artifacts",
    filename="color_palette_preview.png",
    overwrite=True,
)


# ---------------------------------------------------------------------------
# Step 4: Compute derived hex codes + contrast helpers via FormatText
mix_template = """{%- set base_value = base.value or '#000000' -%}\n{%- set mix_value = mix_color or '#FFFFFF' -%}\n{%- set base_r = base_value[1:3]|int(base=16) -%}\n{%- set base_g = base_value[3:5]|int(base=16) -%}\n{%- set base_b = base_value[5:7]|int(base=16) -%}\n{%- set mix_r = mix_value[1:3]|int(base=16) -%}\n{%- set mix_g = mix_value[3:5]|int(base=16) -%}\n{%- set mix_b = mix_value[5:7]|int(base=16) -%}\n{%- set result_r = ((base_r * (1 - mix_alpha)) + (mix_r * mix_alpha))|round|int -%}\n{%- set result_g = ((base_g * (1 - mix_alpha)) + (mix_g * mix_alpha))|round|int -%}\n{%- set result_b = ((base_b * (1 - mix_alpha)) + (mix_b * mix_alpha))|round|int -%}\n{%- set output = '#{:02X}{:02X}{:02X}'.format(result_r, result_g, result_b) -%}\n{{ output }}"""

complement_template = """{%- set base_value = base.value or '#000000' -%}\n{%- set r = 255 - (base_value[1:3]|int(base=16)) -%}\n{%- set g = 255 - (base_value[3:5]|int(base=16)) -%}\n{%- set b = 255 - (base_value[5:7]|int(base=16)) -%}\n{{ '#{:02X}{:02X}{:02X}'.format(r, g, b) }}"""

contrast_template = """{%- set value = color_hex.strip() -%}\n{%- set r = value[1:3]|int(base=16) -%}\n{%- set g = value[3:5]|int(base=16) -%}\n{%- set b = value[5:7]|int(base=16) -%}\n{%- set r_lin = (r / 255)**2.2 -%}\n{%- set g_lin = (g / 255)**2.2 -%}\n{%- set b_lin = (b / 255)**2.2 -%}\n{%- set luminance = 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin -%}\n{%- if luminance > 0.45 -%}#111111{%- else -%}#FFFFFF{%- endif -%}"""

base_hex = FormatText(
    template="{{ (base.value or '#000000')|upper }}",
    base=base_color.output,
)

tint_hex = FormatText(
    template=mix_template,
    base=base_color.output,
    mix_color="#FFFFFF",
    mix_alpha=TINT_ALPHA,
)

accent_hex = FormatText(
    template=mix_template,
    base=base_color.output,
    mix_color=ACCENT_MIX_COLOR,
    mix_alpha=ACCENT_ALPHA,
)

muted_hex = FormatText(
    template=mix_template,
    base=base_color.output,
    mix_color=MUTED_MIX_COLOR,
    mix_alpha=MUTED_ALPHA,
)

shade_hex = FormatText(
    template=mix_template,
    base=base_color.output,
    mix_color="#000000",
    mix_alpha=SHADE_ALPHA,
)

complement_hex = FormatText(
    template=complement_template,
    base=base_color.output,
)

base_contrast = FormatText(
    template=contrast_template,
    color_hex=base_hex.output,
)

tint_contrast = FormatText(
    template=contrast_template,
    color_hex=tint_hex.output,
)

accent_contrast = FormatText(
    template=contrast_template,
    color_hex=accent_hex.output,
)

muted_contrast = FormatText(
    template=contrast_template,
    color_hex=muted_hex.output,
)

shade_contrast = FormatText(
    template=contrast_template,
    color_hex=shade_hex.output,
)

complement_contrast = FormatText(
    template=contrast_template,
    color_hex=complement_hex.output,
)


# ---------------------------------------------------------------------------
# Step 5: Structured JSON + CSS outputs
palette_json = FormatText(
    template="""[
  {
    "name": "Base",
    "hex": "{{ base_hex.strip() }}",
    "role": "Primary brand anchor",
    "contrast_text": "{{ base_contrast.strip() }}",
    "usage": "CTA buttons, navigation highlights, primary badges"
  },
  {
    "name": "Tint",
    "hex": "{{ tint_hex.strip() }}",
    "role": "Soft highlight",
    "contrast_text": "{{ tint_contrast.strip() }}",
    "usage": "Hover states, subtle backgrounds, cards"
  },
  {
    "name": "Accent",
    "hex": "{{ accent_hex.strip() }}",
    "role": "Warm accent",
    "contrast_text": "{{ accent_contrast.strip() }}",
    "usage": "Notifications, emphasis tags, charts"
  },
  {
    "name": "Complement",
    "hex": "{{ complement_hex.strip() }}",
    "role": "Contrast pairing",
    "contrast_text": "{{ complement_contrast.strip() }}",
    "usage": "Secondary CTAs, charts, accessibility-friendly highlights"
  },
  {
    "name": "Muted",
    "hex": "{{ muted_hex.strip() }}",
    "role": "Neutral canvas",
    "contrast_text": "{{ muted_contrast.strip() }}",
    "usage": "Page backgrounds, form surfaces, tables"
  },
  {
    "name": "Shade",
    "hex": "{{ shade_hex.strip() }}",
    "role": "Depth & focus",
    "contrast_text": "{{ shade_contrast.strip() }}",
    "usage": "Text, borders, elevated components"
  }
]""",
    base_hex=base_hex.output,
    tint_hex=tint_hex.output,
    accent_hex=accent_hex.output,
    complement_hex=complement_hex.output,
    muted_hex=muted_hex.output,
    shade_hex=shade_hex.output,
    base_contrast=base_contrast.output,
    tint_contrast=tint_contrast.output,
    accent_contrast=accent_contrast.output,
    complement_contrast=complement_contrast.output,
    muted_contrast=muted_contrast.output,
    shade_contrast=shade_contrast.output,
)

palette_css = FormatText(
    template=""":root {
  --color-base: {{ base_hex.strip() }};
  --color-base-contrast: {{ base_contrast.strip() }};
  --color-tint: {{ tint_hex.strip() }};
  --color-tint-contrast: {{ tint_contrast.strip() }};
  --color-accent: {{ accent_hex.strip() }};
  --color-accent-contrast: {{ accent_contrast.strip() }};
  --color-complement: {{ complement_hex.strip() }};
  --color-complement-contrast: {{ complement_contrast.strip() }};
  --color-muted: {{ muted_hex.strip() }};
  --color-muted-contrast: {{ muted_contrast.strip() }};
  --color-shade: {{ shade_hex.strip() }};
  --color-shade-contrast: {{ shade_contrast.strip() }};
}

.button-primary {
  background: var(--color-base);
  color: var(--color-base-contrast);
  border-color: var(--color-shade);
}

.card-surface {
  background: var(--color-muted);
  border: 1px solid var(--color-shade);
  color: var(--color-shade-contrast);
}

.badge-accent {
  background: var(--color-accent);
  color: var(--color-accent-contrast);
}
""",
    base_hex=base_hex.output,
    base_contrast=base_contrast.output,
    tint_hex=tint_hex.output,
    tint_contrast=tint_contrast.output,
    accent_hex=accent_hex.output,
    accent_contrast=accent_contrast.output,
    complement_hex=complement_hex.output,
    complement_contrast=complement_contrast.output,
    muted_hex=muted_hex.output,
    muted_contrast=muted_contrast.output,
    shade_hex=shade_hex.output,
    shade_contrast=shade_contrast.output,
)


# Persist design artifacts for downstream consumption
css_file = WriteTextFile(
    path="artifacts/palette.css",
    content=palette_css.output,
)

json_file = WriteTextFile(
    path="artifacts/palette.json",
    content=palette_json.output,
)

palette_data = ParseJSON(text=palette_json.output)
palette_table = JSONToDataframe(text=palette_json.output)


# ---------------------------------------------------------------------------
# Step 6: Collect outputs and expose to DSL graph consumers
palette_bundle = MakeDictionary(
    base=base_hex.output,
    tint=tint_hex.output,
    accent=accent_hex.output,
    complement=complement_hex.output,
    muted=muted_hex.output,
    shade=shade_hex.output,
    base_contrast=base_contrast.output,
    tint_contrast=tint_contrast.output,
    accent_contrast=accent_contrast.output,
    complement_contrast=complement_contrast.output,
    muted_contrast=muted_contrast.output,
    shade_contrast=shade_contrast.output,
    css=palette_css.output,
    json=palette_json.output,
    css_file=css_file.output,
    json_file=json_file.output,
    preview_asset=palette_preview_file.output,
    palette=palette_data.output,
)

palette_package_output = DictionaryOutput(
    name="color_palette_package",
    description="Bundled palette metadata, CSS variables, and saved artifacts",
    value=palette_bundle.output,
)

palette_table_output = DataframeOutput(
    name="color_palette_table",
    description="Palette breakdown as a tabular dataset",
    value=palette_table.output,
)

palette_image_output = ImageOutput(
    name="color_palette_preview",
    description="Composite preview strip of generated colors",
    value=palette_preview_file.output,
)


graph = create_graph(
    palette_package_output,
    palette_table_output,
    palette_image_output,
)


if __name__ == "__main__":
    results = run_graph(graph, asset_output_mode=AssetOutputMode.LOCAL)

    bundle = results.get("color_palette_package", {})
    css_path = bundle.get("css_file")
    json_path = bundle.get("json_file")

    print("Generated palette swatches saved to:", bundle.get("preview_asset"))
    print("CSS variables file:", css_path)
    print("JSON metadata file:", json_path)
