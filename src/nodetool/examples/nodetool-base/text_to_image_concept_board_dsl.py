"""
Text-to-Image Concept Board DSL Example

Turn a creative brief into multiple concept art renders using OpenAI's text-to-image node.

Workflow:
1. **Creative Brief Inputs** – Capture product, mood, setting, and style direction
2. **Prompt Assembly** – Build reusable prompt fragments with templated text nodes
3. **Image Generation** – Produce hero and variant renders via `CreateImage`
4. **Asset Packaging** – Collect images and prompt metadata for downstream review

ASCII pipeline:

[StringInput] --
               \
[StringInput] ----> [FormatText] --> [CreateImage] --> [ImageOutput]
               /
[StringInput] --
               \
[StringInput] ----> [FormatText] --> [CreateImage] --> [ImageOutput]
                         \
                          --> [MakeDictionary] --> [DictionaryOutput]
                             \
                              --> [FormatText] --> [StringOutput]
"""

from nodetool.dsl.graph import create_graph, run_graph
from nodetool.dsl.nodetool.input import StringInput
from nodetool.dsl.nodetool.text import FormatText
from nodetool.dsl.openai.image import CreateImage
from nodetool.dsl.nodetool.dictionary import MakeDictionary
from nodetool.dsl.nodetool.output import (
    DictionaryOutput,
    ImageOutput,
    StringOutput,
)
from nodetool.workflows.processing_context import AssetOutputMode


# --- Creative Brief Inputs --------------------------------------------------
brand_name = StringInput(
    name="brand_name",
    description="Brand or campaign the artwork should represent",
    value="Aurora Labs sustainable tech launch",
)

hero_product = StringInput(
    name="hero_product",
    description="Flagship product or subject to visualize",
    value="lightweight foldable solar drone hovering above a misty alpine forest",
)

visual_style = StringInput(
    name="visual_style",
    description="High-level art direction keywords",
    value="cinematic concept art, volumetric lighting, ultra-detailed, sharp focus",
)

color_language = StringInput(
    name="color_language",
    description="Preferred palette and tones to emphasize",
    value="twilight blues, teal gradients, warm amber highlights",
)

atmosphere = StringInput(
    name="atmosphere",
    description="Desired mood or narrative feeling",
    value="optimistic, future-forward, quietly awe inspiring",
)


# --- Prompt Assembly -------------------------------------------------------
core_scene_prompt = FormatText(
    template=(
        "{{ product }} with {{ style }}."
        " Emphasize {{ colors }} and an atmosphere that feels {{ mood }}."
    ),
    product=hero_product.output,
    style=visual_style.output,
    colors=color_language.output,
    mood=atmosphere.output,
)

hero_prompt = FormatText(
    template=(
        "Design a flagship key art poster for {{ brand }} showing {{ core_scene }}"
        " Shot on an anamorphic lens with sweeping cinematic scale,"
        " dramatic volumetric god rays, and intricate environmental storytelling."
        " Include subtle UI holograms around the drone and distant mountain silhouettes."
    ),
    brand=brand_name.output,
    core_scene=core_scene_prompt.output,
)

variant_prompt = FormatText(
    template=(
        "Create an alternate storyboard frame for {{ brand }} featuring {{ core_scene }}"
        " Capture a lower-angle perspective framed by pine silhouettes"
        " with long-exposure light trails around the drone and a soft-focus background."
    ),
    brand=brand_name.output,
    core_scene=core_scene_prompt.output,
)

prompt_brief = FormatText(
    template="""# Concept Board Prompt Brief

## Campaign
{{ brand }}

## Hero Prompt
{{ hero_prompt }}

## Variant Prompt
{{ variant_prompt }}

## Art Direction Notes
- Style: {{ style }}
- Palette: {{ colors }}
- Mood: {{ mood }}
""",
    brand=brand_name.output,
    hero_prompt=hero_prompt.output,
    variant_prompt=variant_prompt.output,
    style=visual_style.output,
    colors=color_language.output,
    mood=atmosphere.output,
)


# --- Image Generation ------------------------------------------------------
hero_render = CreateImage(
    prompt=hero_prompt.output,
    size=CreateImage.Size._1024x1024,
    quality=CreateImage.Quality.high,
)

variant_render = CreateImage(
    prompt=variant_prompt.output,
    size=CreateImage.Size._1024x1536,
    quality=CreateImage.Quality.high,
)


# --- Asset Packaging -------------------------------------------------------
hero_output = ImageOutput(
    name="hero_concept",
    description="Hero key art render for campaign review",
    value=hero_render.output,
)

variant_output = ImageOutput(
    name="variant_frame",
    description="Storyboard-style alternate frame for the concept board",
    value=variant_render.output,
)

image_manifest = MakeDictionary(
    hero=hero_output.out.output,
    variant=variant_output.out.output,
)

manifest_output = DictionaryOutput(
    name="concept_gallery",
    description="Collection of generated image assets",
    value=image_manifest.out.output,
)

brief_output = StringOutput(
    name="prompt_brief",
    description="Human-readable prompts and art direction notes",
    value=prompt_brief.output,
)

# The graph exposes both individual image outputs and the packaged manifest.
graph = create_graph(hero_output, variant_output, manifest_output, brief_output)


if __name__ == "__main__":
    results = run_graph(
        graph,
        asset_output_mode=AssetOutputMode.WORKSPACE,
    )
    print("Generated concept assets:")
    for key, value in results.items():
        print(f"- {key}: {value}")
