"""
Example: Product Photography Mockup Generator DSL Workflow

This workflow generates product mockups in diverse scenes without physical shoots. It demonstrates:

1. Upload plain product image
2. Place product in various lifestyle scenes
3. Add realistic lighting and shadows
4. Generate multiple scene variations (studio, outdoor, lifestyle)
5. Batch process for e-commerce listings

The workflow pattern:
    [ImageInput] -> [Agent] (scene concepts) -> [ListGenerator]
                        -> [ForEach] -> [ImageToImage] -> [Collect] -> [Output]

Empowers designers to create product photography without physical shoots.
"""

from nodetool.dsl.graph import create_graph, run_graph, AssetOutputMode
from nodetool.dsl.nodetool.input import ImageInput, StringInput, IntegerInput
from nodetool.dsl.nodetool.text import FormatText
from nodetool.dsl.nodetool.agents import Agent
from nodetool.dsl.nodetool.generators import ListGenerator
from nodetool.dsl.nodetool.image import ImageToImage, TextToImage
from nodetool.dsl.lib.pillow.enhance import Brightness, Contrast
from nodetool.dsl.nodetool.control import ForEach, Collect
from nodetool.dsl.nodetool.output import Output
from nodetool.metadata.types import (
    LanguageModel,
    Provider,
    ImageModel,
    ImageRef,
)


def build_product_mockup_generator():
    """
    Generate product mockups in diverse scenes.

    This function builds a workflow graph that:
    1. Accepts a product image and description
    2. Generates scene concepts for different contexts
    3. Places the product in various environments
    4. Applies realistic lighting and styling
    5. Returns a collection of product mockups

    Returns:
        Graph: A graph object representing the workflow
    """
    # --- Inputs ---
    product_image = ImageInput(
        name="product_image",
        description="The product image to place in scenes",
        value=ImageRef(type="image", uri=""),
    )

    product_name = StringInput(
        name="product_name",
        description="Name of the product",
        value="Premium Wireless Headphones",
    )

    product_description = StringInput(
        name="product_description",
        description="Brief product description for context",
        value="Sleek, modern wireless headphones with noise cancellation. Matte black finish with silver accents.",
    )

    target_audience = StringInput(
        name="target_audience",
        description="Target customer demographic",
        value="Tech-savvy professionals, music enthusiasts, remote workers",
    )

    num_scenes = IntegerInput(
        name="num_scenes",
        description="Number of scene variations to generate",
        value=6,
        min=2,
        max=12,
    )

    # --- Generate scene concepts ---
    scene_concept_prompt = FormatText(
        template="""
You are a product photography director.

Product: {{ name }}
Description: {{ description }}
Target Audience: {{ audience }}

Create {{ count }} unique scene concepts for product photography mockups.

Include a variety of settings:
- Studio shots (clean backgrounds, professional lighting)
- Lifestyle scenes (product in use)
- Environmental/outdoor settings
- Flat lay compositions
- Detail/close-up angles

For each scene, describe:
1. Setting/environment
2. Lighting style
3. Props and complementary elements
4. Mood/atmosphere
5. Camera angle suggestion

Output format: Return each complete scene description on a separate line, no numbering.
""",
        name=product_name.output,
        description=product_description.output,
        audience=target_audience.output,
        count=num_scenes.output,
    )

    scene_concepts = Agent(
        model=LanguageModel(
            type="language_model",
            provider=Provider.OpenAI,
            id="gpt-4o-mini",
        ),
        system="You are a professional product photographer. Create compelling scene concepts.",
        prompt=scene_concept_prompt.output,
        max_tokens=2048,
    )

    # --- Generate image prompts for each scene ---
    prompt_generator = FormatText(
        template="""
Convert these scene concepts into image generation prompts.

Scene Concepts: {{ scenes }}
Product: {{ name }} - {{ description }}

For each scene, create a detailed image prompt that:
- Describes the complete scene with the product
- Specifies lighting (soft, dramatic, natural, studio)
- Includes realistic shadows and reflections
- Mentions surface materials and textures
- Uses professional product photography style

Output format: One complete prompt per line, no numbering.
Make each prompt suitable for high-quality product photography.
""",
        scenes=scene_concepts.out.text,
        name=product_name.output,
        description=product_description.output,
    )

    image_prompts = ListGenerator(
        model=LanguageModel(
            type="language_model",
            provider=Provider.OpenAI,
            id="gpt-4o-mini",
        ),
        prompt=prompt_generator.output,
        max_tokens=2048,
    )

    mockup_image = ImageToImage(
        model=ImageModel(
            type="image_model",
            provider=Provider.OpenAI,
            id="gpt-image-1.5",
        ),
        image=product_image.output,
        prompt=image_prompts.out.item,
        strength=0.65,
        target_width=1200,
        target_height=1200,
    )

    enhanced = Brightness(
        image=mockup_image.output,
        factor=1.05,
    )

    polished = Contrast(
        image=enhanced.output,
        factor=1.1,
    )

    alt_mockup = TextToImage(
        model=ImageModel(
            type="image_model",
            provider=Provider.OpenAI,
            id="gpt-image-1.5",
        ),
        prompt=image_prompts.out.item,
        width=1200,
        height=1200,
        guidance_scale=7.0,
        num_inference_steps=25,
    )

    # --- Outputs ---
    mockups_out = Output(
        name="product_mockups",
        value=polished.output,
        description="Product mockups in various scenes (image-to-image)",
    )

    text_mockups_out = Output(
        name="generated_scenes",
        value=alt_mockup.output,
        description="Generated product scenes (text-to-image)",
    )

    concepts_out = Output(
        name="scene_concepts",
        value=scene_concepts.out.text,
        description="Scene concept descriptions for reference",
    )

    return create_graph(mockups_out, text_mockups_out, concepts_out)


# Build the graph
graph = build_product_mockup_generator()


if __name__ == "__main__":
    """
    To run this example:

    1. Prepare a product image (preferably with clean background)
    2. Ensure you have API keys configured for OpenAI and FAL AI
    3. Run:

        python examples/product_mockup_generator.py

    The workflow generates product mockups in multiple scene variations.
    """

    print("Product Photography Mockup Generator Workflow")
    print("=" * 50)
    print(f"Nodes: {len(graph.nodes)}")
    print(f"Edges: {len(graph.edges)}")
    print()
    print("Workflow pattern:")
    print("  [Product Image + Details]")
    print("      -> [Agent] (scene concepts)")
    print("          -> [ListGenerator] (image prompts)")
    print("              -> [ForEach] (iterate)")
    print("                  -> [ImageToImage] (product in scene)")
    print("                  -> [TextToImage] (generated scenes)")
    print("                      -> [Brightness/Contrast] (enhance)")
    print("                          -> [Collect]")
    print("                              -> [Outputs]")
    print()
    print("Scene Types:")
    print("  - Studio shots with clean backgrounds")
    print("  - Lifestyle scenes showing product in use")
    print("  - Outdoor/environmental settings")
    print("  - Flat lay compositions")
    print("  - Detail/close-up angles")
    print()

    # Uncomment to run:
    result = run_graph(graph, asset_output_mode=AssetOutputMode.WORKSPACE)
    print(result)
