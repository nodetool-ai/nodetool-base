"""
Concept Art Generator DSL Example

Transform a single story idea into three complementary concept art frames.

Workflow:
1. **Narrative Inputs** – Define the core theme, key character, and cinematic mood.
2. **Prompt Crafting** – Shape tailored prompts for establishing, action, and detail shots.
3. **Text-to-Image Generation** – Render scene variations with high-resolution settings.
4. **Prompt Manifest** – Output the exact prompts for reproducibility and iteration.
"""

from nodetool.dsl.graph import create_graph, run_graph
from nodetool.dsl.nodetool.input import StringInput
from nodetool.dsl.nodetool.text import FormatText
from nodetool.dsl.nodetool.image import TextToImage
from nodetool.dsl.nodetool.output import ImageOutput, StringOutput
from nodetool.workflows.processing_context import AssetOutputMode


# --- Narrative Inputs --------------------------------------------------------
concept_theme = StringInput(
    name="concept_theme",
    description="World-building hook or setting", 
    value="Solar punk city rooftop gardens reconnecting nature and technology",
)

hero_focus = StringInput(
    name="hero_focus",
    description="Protagonist description",
    value="A young botanist engineer with braided hair, modular exo-suit, and bioluminescent tools",
)

mood_direction = StringInput(
    name="mood_direction",
    description="Overall tone and lighting",
    value="Optimistic dusk lighting with warm rim light, misty atmosphere, hopeful energy",
)

camera_brief = StringInput(
    name="camera_brief",
    description="Desired composition or perspective",
    value="Dynamic wide-angle composition with layered depth cues",
)

# --- Prompt Crafting ---------------------------------------------------------
base_prompt = FormatText(
    template="""{{ theme }}. {{ mood }}. Cinematic illustration style, hyper-detailed, artstation trending.""",
    theme=concept_theme.output,
    mood=mood_direction.output,
)

establishing_prompt = FormatText(
    template="""{{ base }} Scene focus: panoramic establishing shot showcasing architectural silhouettes, elevated gardens, suspended walkways, atmospheric perspective, volumetric lighting. Camera direction: {{ camera }}.""",
    base=base_prompt.output,
    camera=camera_brief.output,
)

action_prompt = FormatText(
    template="""{{ base }} Character focus: {{ hero }} mid-action tending to kinetic hydroponic arrays, motion trails, storytelling props, expressive body language. Capture sense of scale with foreground foliage and distant city core.""",
    base=base_prompt.output,
    hero=hero_focus.output,
)

detail_prompt = FormatText(
    template="""{{ base }} Detail focus: close-up of engineered plants with translucent leaves, glowing sap lines, and micro-drones pollinating. Macro depth of field, crisp material rendering, intricate surface textures.""",
    base=base_prompt.output,
)

negative_prompt = (
    "low resolution, blurry, dull colors, watermark, text overlay, distorted anatomy, extra limbs, frame, border"
)

# --- Text-to-Image Generation ------------------------------------------------
establishing_render = TextToImage(
    prompt=establishing_prompt.output,
    negative_prompt=negative_prompt,
    width=1152,
    height=640,
    guidance_scale=6.0,
    num_inference_steps=28,
)

action_render = TextToImage(
    prompt=action_prompt.output,
    negative_prompt=negative_prompt,
    width=896,
    height=1152,
    guidance_scale=6.5,
    num_inference_steps=30,
)

detail_render = TextToImage(
    prompt=detail_prompt.output,
    negative_prompt=negative_prompt,
    width=1024,
    height=1024,
    guidance_scale=7.0,
    num_inference_steps=32,
)

# --- Outputs -----------------------------------------------------------------
establishing_output = ImageOutput(
    name="establishing_scene",
    description="Panoramic world-building frame",
    value=establishing_render.output,
)

action_output = ImageOutput(
    name="hero_moment",
    description="Character-focused action beat",
    value=action_render.output,
)

detail_output = ImageOutput(
    name="detail_macro",
    description="Macro detail exploration",
    value=detail_render.output,
)

prompt_manifest = FormatText(
    template="""# Concept Art Prompt Manifest

- Establishing Scene Prompt:\n{{ establish }}
- Hero Moment Prompt:\n{{ action }}
- Detail Macro Prompt:\n{{ detail }}

Negative Prompt:\n{{ negative }}
""",
    establish=establishing_prompt.output,
    action=action_prompt.output,
    detail=detail_prompt.output,
    negative=negative_prompt,
)

prompt_output = StringOutput(
    name="prompt_manifest",
    description="Prompts used for each render",
    value=prompt_manifest.output,
)

# --- Graph -------------------------------------------------------------------
graph = create_graph(
    establishing_output,
    action_output,
    detail_output,
    prompt_output,
)


if __name__ == "__main__":
    results = run_graph(graph, asset_output_mode=AssetOutputMode.WORKSPACE)
    print("Generated assets:")
    for key in ["establishing_scene", "hero_moment", "detail_macro"]:
        print(f"- {key}: {results[key]}")
    print("\nPrompts:\n", results["prompt_manifest"])
