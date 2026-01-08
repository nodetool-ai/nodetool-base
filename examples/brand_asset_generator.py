"""
Example: Brand Asset Generator DSL Workflow

This workflow generates consistent brand assets across multiple platforms from
brand guidelines. It demonstrates:

1. Input brand guidelines (colors, logo concept, fonts, brand name)
2. Generate platform-specific assets (Instagram, LinkedIn, banner formats)
3. Branch for different platform dimensions
4. Apply consistent text overlays and styling

The workflow pattern:
    [StringInputs] -> [FormatText] -> [Agent] -> [ListGenerator] -> [ForEach]
                                                    -> [TextToImage] -> [Resize] -> [RenderText] -> [Output]

Perfect for graphic designers maintaining brand consistency across platforms.
"""

from nodetool.dsl.graph import create_graph
from nodetool.dsl.nodetool.input import StringInput, ColorInput
from nodetool.dsl.nodetool.text import FormatText
from nodetool.dsl.nodetool.generators import ListGenerator
from nodetool.dsl.nodetool.image import TextToImage
from nodetool.dsl.lib.pillow.draw import RenderText
from nodetool.dsl.nodetool.control import ForEach, Collect
from nodetool.dsl.nodetool.output import Output
from nodetool.metadata.types import (
    LanguageModel,
    Provider,
    ImageModel,
    ColorRef,
    FontRef,
)


def build_brand_asset_generator():
    """
    Generate consistent brand assets for social media platforms.

    This function builds a workflow graph that:
    1. Accepts brand guidelines (name, colors, style description)
    2. Uses an LLM to generate platform-specific prompts
    3. Creates images for each platform using text-to-image
    4. Applies brand text overlays
    5. Resizes to platform-specific dimensions

    Returns:
        Graph: A graph object representing the workflow
    """
    # --- Inputs ---
    brand_name = StringInput(
        name="brand_name",
        description="The brand name to feature on assets",
        value="TechFlow",
    )

    brand_description = StringInput(
        name="brand_description",
        description="Description of brand style and values",
        value="A modern tech startup focused on productivity tools. Clean, minimal aesthetic with futuristic vibes.",
    )

    primary_color = ColorInput(
        name="primary_color",
        description="Primary brand color (hex)",
        value=ColorRef(type="color", value="#3498db"),
    )

    tagline = StringInput(
        name="tagline",
        description="Brand tagline for overlays",
        value="Streamline Your Workflow",
    )

    # --- Generate platform-specific prompts ---
    prompt_generator_instruction = FormatText(
        template="""
You are a creative director generating image prompts for brand assets.

Brand: {{ brand_name }}
Description: {{ brand_description }}

Generate 4 different image prompts for social media posts. Each prompt should:
- Be suitable for text-to-image AI generation
- Reflect the brand's aesthetic and values
- Work well as a background for text overlays
- Be professional and eye-catching

Output format: Return each prompt on a separate line, no numbering or bullets.
Include prompts for: Instagram square post, LinkedIn banner, Twitter header, and a product announcement.
""",
        brand_name=brand_name.output,
        brand_description=brand_description.output,
    )

    # --- Use ListGenerator to create prompts ---
    prompt_list = ListGenerator(
        model=LanguageModel(
            type="language_model",
            provider=Provider.OpenAI,
            id="gpt-4o-mini",
        ),
        prompt=prompt_generator_instruction.output,
        max_tokens=2048,
    )

    # --- Iterate over prompts and generate images ---
    prompt_iterator = ForEach(
        input_list=prompt_list.out.item,
    )

    # --- Generate base image ---
    base_image = TextToImage(
        model=ImageModel(
            type="image_model",
            provider=Provider.HuggingFaceFalAI,
            id="fal-ai/flux/schnell",
            name="FLUX.1 Schnell",
        ),
        prompt=prompt_iterator.out.output,
        width=1024,
        height=1024,
        guidance_scale=7.5,
        num_inference_steps=30,
    )

    # --- Add brand text overlay ---
    branded_image = RenderText(
        image=base_image.output,
        text=brand_name.output,
        x=50,
        y=900,
        size=72,
        color=primary_color.output,
        font=FontRef(type="font", name="DejaVuSans"),
    )

    # --- Add tagline ---
    final_image = RenderText(
        image=branded_image.output,
        text=tagline.output,
        x=50,
        y=980,
        size=32,
        color=ColorRef(type="color", value="#FFFFFF"),
        font=FontRef(type="font", name="DejaVuSans"),
    )

    # --- Collect all generated images ---
    collected_images = Collect(
        input_item=final_image.output,
    )

    # --- Output ---
    output = Output(
        name="brand_assets",
        value=collected_images.out.output,
        description="Generated brand assets ready for social media platforms",
    )

    return create_graph(output)


# Build the graph
graph = build_brand_asset_generator()


if __name__ == "__main__":
    """
    To run this example:

    1. Ensure you have API keys configured for OpenAI and FAL AI
    2. Run:

        python examples/brand_asset_generator.py

    The workflow generates multiple brand-consistent images for social media.
    """

    print("Brand Asset Generator Workflow")
    print("=" * 50)
    print(f"Nodes: {len(graph.nodes)}")
    print(f"Edges: {len(graph.edges)}")
    print()
    print("Workflow pattern:")
    print("  [Brand Inputs]")
    print("      -> [FormatText] (create prompt instruction)")
    print("          -> [ListGenerator] (generate platform prompts)")
    print("              -> [ForEach] (iterate prompts)")
    print("                  -> [TextToImage] (generate base images)")
    print("                      -> [RenderText] (add brand name)")
    print("                          -> [RenderText] (add tagline)")
    print("                              -> [Collect] (gather all)")
    print("                                  -> [Output]")
    print()

    # Uncomment to run:
    # import asyncio
    # result = asyncio.run(run_graph(graph, user_id="example_user", auth_token="token"))
    # print(f"Generated {len(result['brand_assets'])} brand assets")
