"""
Example: YouTube Thumbnail Pipeline DSL Workflow

This workflow generates multiple eye-catching YouTube thumbnail variations from
a video script or keyframe description. It demonstrates:

1. Input video script/topic and keyframe description
2. Generate multiple thumbnail concepts using LLM
3. Create thumbnail images with bold text and dramatic styling
4. Batch process variations for A/B testing
5. Output ranked thumbnails based on composition

The workflow pattern:
    [StringInputs] -> [Agent] -> [ListGenerator] -> [ForEach]
                                    -> [TextToImage] -> [RenderText] -> [Collect] -> [Output]

Ideal for video creators optimizing click-through rates.
"""

from nodetool.dsl.graph import create_graph
from nodetool.dsl.nodetool.input import StringInput, IntegerInput
from nodetool.dsl.nodetool.text import FormatText
from nodetool.dsl.nodetool.generators import ListGenerator
from nodetool.dsl.nodetool.image import TextToImage
from nodetool.dsl.lib.pillow.draw import RenderText
from nodetool.dsl.lib.pillow.enhance import Contrast, Brightness
from nodetool.dsl.nodetool.control import ForEach, Collect
from nodetool.dsl.nodetool.output import Output
from nodetool.metadata.types import (
    LanguageModel,
    Provider,
    ImageModel,
    ColorRef,
    FontRef,
)


def build_youtube_thumbnail_pipeline():
    """
    Generate multiple YouTube thumbnail variations.

    This function builds a workflow graph that:
    1. Accepts video title and script/topic description
    2. Uses LLM to generate dramatic thumbnail concepts
    3. Creates multiple thumbnail variations with text-to-image
    4. Applies bold text overlays with YouTube-style formatting
    5. Returns a collection of thumbnail options

    Returns:
        Graph: A graph object representing the workflow
    """
    # --- Inputs ---
    video_title = StringInput(
        name="video_title",
        description="The YouTube video title",
        value="10 AI Tools That Will Change How You Work in 2024",
    )

    video_topic = StringInput(
        name="video_topic",
        description="Brief description of the video content",
        value="A tech review video covering productivity AI tools including ChatGPT, Claude, and automation tools. Target audience: tech enthusiasts and professionals.",
    )

    thumbnail_text = StringInput(
        name="thumbnail_text",
        description="Main text to overlay on thumbnails (short, punchy)",
        value="AI TOOLS 2024",
    )

    num_variations = IntegerInput(
        name="num_variations",
        description="Number of thumbnail variations to generate",
        value=6,
        min=1,
        max=20,
    )

    # --- Generate thumbnail concepts ---
    concept_generator = FormatText(
        template="""
You are a YouTube thumbnail expert. Generate {{ count }} different thumbnail image prompts.

Video Title: {{ title }}
Topic: {{ topic }}

For each thumbnail concept:
- Focus on creating visually striking, clickable images
- Include dramatic lighting and high contrast
- Suggest faces with expressive emotions when appropriate
- Use bold colors that pop (reds, yellows, blues)
- Keep backgrounds clean but impactful
- Make it suitable for 1280x720 format

Output format: Return each prompt on a separate line, no numbering.
Each prompt should be a detailed text-to-image prompt.
""",
        count=num_variations.output,
        title=video_title.output,
        topic=video_topic.output,
    )

    # --- Generate prompts using ListGenerator ---
    thumbnail_prompts = ListGenerator(
        model=LanguageModel(
            type="language_model",
            provider=Provider.OpenAI,
            id="gpt-4o-mini",
        ),
        prompt=concept_generator.output,
        max_tokens=2048,
    )

    # --- Iterate and generate images ---
    prompt_iterator = ForEach(
        input_list=thumbnail_prompts.out.item,
    )

    # --- Generate base thumbnail ---
    base_thumbnail = TextToImage(
        model=ImageModel(
            type="image_model",
            provider=Provider.HuggingFaceFalAI,
            id="fal-ai/flux/schnell",
            name="FLUX.1 Schnell",
        ),
        prompt=prompt_iterator.out.output,
        width=1280,
        height=720,
        guidance_scale=8.0,
        num_inference_steps=35,
    )

    # --- Enhance for YouTube style ---
    enhanced = Contrast(
        image=base_thumbnail.output,
        factor=1.2,
    )

    brightened = Brightness(
        image=enhanced.output,
        factor=1.1,
    )

    # --- Add bold text overlay ---
    # Main text (large, top portion)
    with_main_text = RenderText(
        image=brightened.output,
        text=thumbnail_text.output,
        x=50,
        y=100,
        size=96,
        color=ColorRef(type="color", value="#FFFF00"),  # Yellow for visibility
        font=FontRef(type="font", name="DejaVuSans-Bold"),
    )

    # Subtitle/call to action
    final_thumbnail = RenderText(
        image=with_main_text.output,
        text="WATCH NOW ▶",
        x=50,
        y=620,
        size=48,
        color=ColorRef(type="color", value="#FFFFFF"),
        font=FontRef(type="font", name="DejaVuSans"),
    )

    # --- Collect all variations ---
    collected_thumbnails = Collect(
        input_item=final_thumbnail.output,
    )

    # --- Output ---
    output = Output(
        name="thumbnail_variations",
        value=collected_thumbnails.out.output,
        description="Collection of YouTube thumbnail variations for A/B testing",
    )

    return create_graph(output)


# Build the graph
graph = build_youtube_thumbnail_pipeline()


if __name__ == "__main__":
    """
    To run this example:

    1. Ensure you have API keys configured for OpenAI and FAL AI
    2. Run:

        python examples/youtube_thumbnail_pipeline.py

    The workflow generates multiple thumbnail variations for YouTube videos.
    """

    print("YouTube Thumbnail Pipeline Workflow")
    print("=" * 50)
    print(f"Nodes: {len(graph.nodes)}")
    print(f"Edges: {len(graph.edges)}")
    print()
    print("Workflow pattern:")
    print("  [Video Title/Topic Inputs]")
    print("      -> [FormatText] (thumbnail concept prompt)")
    print("          -> [ListGenerator] (generate multiple concepts)")
    print("              -> [ForEach] (iterate concepts)")
    print("                  -> [TextToImage] (generate thumbnails)")
    print("                      -> [Contrast/Brightness] (enhance)")
    print("                          -> [RenderText] (add text overlays)")
    print("                              -> [Collect]")
    print("                                  -> [Output]")
    print()

    # Uncomment to run:
    # import asyncio
    # result = asyncio.run(run_graph(graph, user_id="example_user", auth_token="token"))
    # print(f"Generated {len(result['thumbnail_variations'])} thumbnail variations")
