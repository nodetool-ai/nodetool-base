"""
Interior Design Visualizer DSL Example

Generate cohesive interior design shots that explore layout, styling, and material palettes.

Workflow:
1. **Design Brief Inputs** – Capture room type, style, color palette, and focal feature.
2. **Prompt Generation** – Derive prompts for layout plan, styled vignette, and material board.
3. **Text-to-Image Rendering** – Produce high-resolution visualizations for each angle.
4. **Design Packet Summary** – Output prompts for reuse alongside design narrative.
"""

from nodetool.dsl.graph import create_graph, run_graph
from nodetool.dsl.nodetool.input import StringInput
from nodetool.dsl.nodetool.text import FormatText
from nodetool.dsl.nodetool.image import TextToImage
from nodetool.dsl.nodetool.output import ImageOutput, StringOutput
from nodetool.workflows.processing_context import AssetOutputMode


# --- Design Brief Inputs -----------------------------------------------------
room_type = StringInput(
    name="room_type",
    description="Room being designed",
    value="Open concept living room with reading nook",
)

style_direction = StringInput(
    name="style_direction",
    description="Interior design style",
    value="Japandi fusion of Japanese minimalism and Scandinavian warmth",
)

palette_direction = StringInput(
    name="palette_direction",
    description="Color palette guidance",
    value="Soft clay neutrals, charcoal contrast, muted forest green accents",
)

focal_feature = StringInput(
    name="focal_feature",
    description="Signature element to highlight",
    value="Custom built-in shelving with integrated lighting and curved edges",
)

# --- Prompt Generation -------------------------------------------------------
base_design_prompt = FormatText(
    template="""{{ room }} styled in {{ style }} with palette {{ palette }}. Natural light, realistic rendering, design-magazine photography.""",
    room=room_type.output,
    style=style_direction.output,
    palette=palette_direction.output,
)

layout_prompt = FormatText(
    template="""{{ base }} Shot type: wide layout view showing circulation flow, furniture arrangement, and focal element {{ focal }}. Render floor-to-ceiling perspective with architectural accuracy.""",
    base=base_design_prompt.output,
    focal=focal_feature.output,
)

vignette_prompt = FormatText(
    template="""{{ base }} Shot type: styled vignette focusing on seating area, layered textiles, curated accessories, and ambient lighting. Include subtle human touch like open book or tea set.""",
    base=base_design_prompt.output,
)

material_prompt = FormatText(
    template="""{{ base }} Shot type: flat-lay material board featuring textiles, woods, stone samples, and metal finishes. Arrange swatches neatly with shadows and labeling tape aesthetic.""",
    base=base_design_prompt.output,
)

negative_prompt = (
    "low resolution, cluttered composition, watermark, text overlay, exaggerated proportions, warped perspective, lens distortion"
)

# --- Text-to-Image Rendering -------------------------------------------------
layout_render = TextToImage(
    prompt=layout_prompt.output,
    negative_prompt=negative_prompt,
    width=1280,
    height=960,
    guidance_scale=6.8,
    num_inference_steps=28,
)

vignette_render = TextToImage(
    prompt=vignette_prompt.output,
    negative_prompt=negative_prompt,
    width=960,
    height=1280,
    guidance_scale=7.0,
    num_inference_steps=30,
)

material_render = TextToImage(
    prompt=material_prompt.output,
    negative_prompt=negative_prompt,
    width=1024,
    height=1024,
    guidance_scale=6.2,
    num_inference_steps=24,
)

# --- Outputs -----------------------------------------------------------------
layout_output = ImageOutput(
    name="layout_view",
    description="Overall room layout visualization",
    value=layout_render.output,
)

vignette_output = ImageOutput(
    name="styled_vignette",
    description="Intimate styled moment",
    value=vignette_render.output,
)

material_output = ImageOutput(
    name="material_board",
    description="Flat-lay material exploration",
    value=material_render.output,
)

design_packet = FormatText(
    template="""# Interior Design Prompt Packet

- Layout View:\n{{ layout }}
- Styled Vignette:\n{{ vignette }}
- Material Board:\n{{ material }}

Negative Prompt:\n{{ negative }}
""",
    layout=layout_prompt.output,
    vignette=vignette_prompt.output,
    material=material_prompt.output,
    negative=negative_prompt,
)

packet_output = StringOutput(
    name="design_packet",
    description="Prompts and directions for each visualization",
    value=design_packet.output,
)

# --- Graph -------------------------------------------------------------------
graph = create_graph(
    layout_output,
    vignette_output,
    material_output,
    packet_output,
)


if __name__ == "__main__":
    results = run_graph(graph, asset_output_mode=AssetOutputMode.WORKSPACE)
    print("Generated interior design visuals:")
    for key in ["layout_view", "styled_vignette", "material_board"]:
        print(f"- {key}: {results[key]}")
    print("\nPrompts:\n", results["design_packet"])
