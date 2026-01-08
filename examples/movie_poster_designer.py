"""
Example: Cinematic Movie Poster Designer DSL Workflow

This workflow creates professional movie poster variations from a film synopsis. It demonstrates:

1. Input film synopsis and genre
2. Generate dramatic poster compositions
3. Create multiple style variations
4. Apply typography integration
5. Upscale to print resolution

The workflow pattern:
    [StringInputs] -> [Agent] (creative direction) -> [ListGenerator] (poster concepts)
                        -> [ForEach] -> [TextToImage] -> [RenderText] -> [Scale] -> [Collect] -> [Output]

Suited for designers crafting promotional materials.
"""

from nodetool.dsl.graph import create_graph
from nodetool.dsl.nodetool.input import StringInput, IntegerInput
from nodetool.dsl.nodetool.text import FormatText
from nodetool.dsl.nodetool.agents import Agent
from nodetool.dsl.nodetool.generators import ListGenerator
from nodetool.dsl.nodetool.image import TextToImage, Scale
from nodetool.dsl.lib.pillow.draw import RenderText
from nodetool.dsl.lib.pillow.enhance import Contrast
from nodetool.dsl.nodetool.control import ForEach, Collect
from nodetool.dsl.nodetool.output import Output
from nodetool.metadata.types import (
    LanguageModel,
    Provider,
    ImageModel,
    ColorRef,
    FontRef,
)


def build_movie_poster_designer():
    """
    Generate cinematic movie poster variations.

    This function builds a workflow graph that:
    1. Accepts film synopsis, title, and genre
    2. Uses LLM to develop poster concepts
    3. Generates multiple poster variations
    4. Applies title and credit text
    5. Returns high-resolution poster options

    Returns:
        Graph: A graph object representing the workflow
    """
    # --- Inputs ---
    movie_title = StringInput(
        name="movie_title",
        description="The film's title",
        value="ECHOES OF TOMORROW",
    )

    movie_synopsis = StringInput(
        name="movie_synopsis",
        description="Brief plot synopsis",
        value="In a world where memories can be extracted and sold, a former memory thief must confront their own erased past to save the person they once loved.",
    )

    genre = StringInput(
        name="genre",
        description="Film genre",
        value="Sci-fi thriller",
    )

    visual_style = StringInput(
        name="visual_style",
        description="Desired poster visual style",
        value="Neon-noir cyberpunk, atmospheric, moody lighting, dramatic compositions",
    )

    tagline = StringInput(
        name="tagline",
        description="Movie tagline for the poster",
        value="Some memories are worth dying for",
    )

    num_variations = IntegerInput(
        name="num_variations",
        description="Number of poster variations",
        value=4,
        min=2,
        max=8,
    )

    # --- Generate creative direction ---
    direction_prompt = FormatText(
        template="""
You are a movie poster designer for major Hollywood studios.

Title: {{ title }}
Synopsis: {{ synopsis }}
Genre: {{ genre }}
Visual Style: {{ style }}
Tagline: {{ tagline }}

Develop a creative brief for poster design including:
1. Key visual motifs that represent the story
2. Color palette recommendations
3. Composition ideas (close-up, wide shot, symbolic, etc.)
4. Mood and atmosphere direction
5. Typography style suggestions

Be cinematic and dramatic in your vision.
""",
        title=movie_title.output,
        synopsis=movie_synopsis.output,
        genre=genre.output,
        style=visual_style.output,
        tagline=tagline.output,
    )

    creative_direction = Agent(
        model=LanguageModel(
            type="language_model",
            provider=Provider.OpenAI,
            id="gpt-4o-mini",
        ),
        system="You are an award-winning movie poster designer. Create compelling visual concepts.",
        prompt=direction_prompt.output,
        max_tokens=1024,
    )

    # --- Generate poster prompts ---
    poster_prompt_generator = FormatText(
        template="""
Based on this creative direction, generate {{ count }} unique movie poster image prompts.

Creative Direction: {{ direction }}
Genre: {{ genre }}
Style: {{ style }}

Each prompt should:
- Describe a complete poster composition (without text)
- Be dramatic and cinematic
- Work well as a background for title overlay
- Use professional movie poster aesthetics
- Leave space at top and bottom for text

Output format: One detailed prompt per line, no numbering.
""",
        count=num_variations.output,
        direction=creative_direction.out.text,
        genre=genre.output,
        style=visual_style.output,
    )

    poster_prompts = ListGenerator(
        model=LanguageModel(
            type="language_model",
            provider=Provider.OpenAI,
            id="gpt-4o-mini",
        ),
        prompt=poster_prompt_generator.output,
        max_tokens=2048,
    )

    # --- Generate poster images ---
    prompt_iterator = ForEach(
        input_list=poster_prompts.out.item,
    )

    base_poster = TextToImage(
        model=ImageModel(
            type="image_model",
            provider=Provider.HuggingFaceFalAI,
            id="fal-ai/flux/dev",
            name="FLUX.1 Dev",
        ),
        prompt=prompt_iterator.out.output,
        width=768,
        height=1152,  # Standard poster ratio (2:3)
        guidance_scale=8.0,
        num_inference_steps=35,
    )

    # --- Enhance for print ---
    enhanced = Contrast(
        image=base_poster.output,
        factor=1.15,
    )

    # --- Add title text ---
    with_title = RenderText(
        image=enhanced.output,
        text=movie_title.output,
        x=100,
        y=50,
        size=72,
        color=ColorRef(type="color", value="#FFFFFF"),
        font=FontRef(type="font", name="DejaVuSans-Bold"),
    )

    # --- Add tagline ---
    with_tagline = RenderText(
        image=with_title.output,
        text=tagline.output,
        x=100,
        y=1050,
        size=28,
        color=ColorRef(type="color", value="#CCCCCC"),
        font=FontRef(type="font", name="DejaVuSans"),
    )

    # --- Upscale for print ---
    upscaled = Scale(
        image=with_tagline.output,
        scale=2.0,  # 2x upscale for print resolution
    )

    # --- Collect all poster variations ---
    collected_posters = Collect(
        input_item=upscaled.output,
    )

    # --- Outputs ---
    posters_out = Output(
        name="poster_variations",
        value=collected_posters.out.output,
        description="High-resolution movie poster variations",
    )

    direction_out = Output(
        name="creative_direction",
        value=creative_direction.out.text,
        description="Creative direction document for reference",
    )

    return create_graph(posters_out, direction_out)


# Build the graph
graph = build_movie_poster_designer()


if __name__ == "__main__":
    """
    To run this example:

    1. Ensure you have API keys configured for OpenAI and FAL AI
    2. Run:

        python examples/movie_poster_designer.py

    The workflow generates multiple cinematic poster variations.
    """

    print("Cinematic Movie Poster Designer Workflow")
    print("=" * 50)
    print(f"Nodes: {len(graph.nodes)}")
    print(f"Edges: {len(graph.edges)}")
    print()
    print("Workflow pattern:")
    print("  [Film Details Inputs]")
    print("      -> [Agent] (creative direction)")
    print("          -> [ListGenerator] (poster concepts)")
    print("              -> [ForEach] (iterate)")
    print("                  -> [TextToImage] (generate posters)")
    print("                      -> [Contrast] (enhance)")
    print("                          -> [RenderText] (add title)")
    print("                              -> [RenderText] (add tagline)")
    print("                                  -> [Scale] (upscale for print)")
    print("                                      -> [Collect]")
    print("                                          -> [Output]")
    print()
    print("Features:")
    print("  - Multiple composition variations")
    print("  - Dramatic cinematic styling")
    print("  - Typography integration")
    print("  - Print-ready resolution (2x upscale)")
    print()

    # Uncomment to run:
    # import asyncio
    # result = asyncio.run(run_graph(graph, user_id="example_user", auth_token="token"))
    # print(f"Generated {len(result['poster_variations'])} poster variations")
