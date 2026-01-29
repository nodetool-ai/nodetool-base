"""
Example: Concept Art Iteration Board DSL Workflow

This workflow generates concept art from creative briefs with rapid iteration. It demonstrates:

1. Input creative brief with style requirements
2. Generate initial concepts via text-to-image
3. Use LLM feedback loops for refinement suggestions
4. Create mood boards with grid layouts
5. Support iterative variations based on feedback

The workflow pattern:
    [StringInputs] -> [Agent] (expand brief) -> [ListGenerator] (concepts)
                        -> [ForEach] -> [TextToImage] -> [CombineImageGrid] -> [Output]
                                                             -> [Agent] (feedback) -> [Output]

Excellent for illustrators and concept artists in rapid ideation.
"""

from nodetool.dsl.graph import create_graph, run_graph, AssetOutputMode
from nodetool.dsl.nodetool.input import StringInput, IntegerInput
from nodetool.dsl.nodetool.text import FormatText
from nodetool.dsl.nodetool.agents import Agent
from nodetool.dsl.nodetool.generators import ListGenerator
from nodetool.dsl.nodetool.image import TextToImage
from nodetool.dsl.lib.grid import CombineImageGrid
from nodetool.dsl.nodetool.control import ForEach, Collect
from nodetool.dsl.nodetool.output import Output
from nodetool.metadata.types import (
    LanguageModel,
    Provider,
    ImageModel,
)


def build_concept_art_iteration_board():
    """
    Generate concept art iterations from a creative brief.

    This function builds a workflow graph that:
    1. Accepts a creative brief describing the desired concept
    2. Expands the brief into detailed art direction using LLM
    3. Generates multiple concept variations
    4. Combines them into a mood board grid
    5. Provides AI feedback for iteration suggestions

    Returns:
        Graph: A graph object representing the workflow
    """
    # --- Inputs ---
    creative_brief = StringInput(
        name="creative_brief",
        description="The creative brief describing the concept",
        value="A mystical forest guardian - ancient tree spirit protecting enchanted woods. Fantasy RPG character design.",
    )

    art_style = StringInput(
        name="art_style",
        description="Desired art style and medium",
        value="Digital painting, concept art style, painterly, dramatic lighting, rich colors",
    )

    mood_keywords = StringInput(
        name="mood_keywords",
        description="Key mood/atmosphere words",
        value="mysterious, ancient, powerful, serene, magical",
    )

    num_variations = IntegerInput(
        name="num_variations",
        description="Number of concept variations (4, 6, 9 for good grids)",
        value=6,
        min=2,
        max=12,
    )

    # --- Expand creative brief with art direction ---
    brief_expansion = FormatText(
        template="""
You are a creative director for concept art.

Creative Brief: {{ brief }}
Art Style: {{ style }}
Mood: {{ mood }}

Expand this brief into detailed art direction including:
1. Visual composition suggestions
2. Color palette recommendations
3. Key design elements to explore
4. Silhouette and shape language ideas
5. Reference inspirations (describe, don't link)

Be specific and actionable for a concept artist.
""",
        brief=creative_brief.output,
        style=art_style.output,
        mood=mood_keywords.output,
    )

    expanded_direction = Agent(
        model=LanguageModel(
            provider=Provider.OpenAI,
            id="gpt-5-mini",
        ),
        system="You are an experienced concept art director. Provide detailed, inspiring art direction.",
        prompt=brief_expansion.output,
        max_tokens=1024,
    )

    # --- Generate variation prompts ---
    variation_generator = FormatText(
        template="""
Based on this art direction, generate {{ count }} unique concept art prompts.

Art Direction: {{ direction }}
Base Style: {{ style }}

Each prompt should:
- Explore a different interpretation or angle
- Maintain the core concept but vary composition/pose/mood
- Be detailed enough for high-quality image generation
- Include specific lighting and atmosphere descriptions

Output format: One prompt per line, no numbering or bullets.
""",
        count=num_variations.output,
        direction=expanded_direction.out.text,
        style=art_style.output,
    )

    variation_prompts = ListGenerator(
        model=LanguageModel(
            provider=Provider.OpenAI,
            id="gpt-5-mini",
        ),
        prompt=variation_generator.output,
        max_tokens=2048,
    )

    concept_image = TextToImage(
        model=ImageModel(
            provider=Provider.OpenAI,
            id="gpt-image-1.5",
        ),
        prompt=variation_prompts.out.item,
        width=1024,
        height=1024,
        guidance_scale=7.5,
        num_inference_steps=30,
    )

    collected_concepts = Collect(
        input_item=concept_image.output,
    )

    mood_board = CombineImageGrid(
        tiles=collected_concepts.out.output,
        columns=3,
    )

    # --- Generate feedback and iteration suggestions ---
    feedback_prompt = FormatText(
        template="""
Review these concept art explorations as a creative director.

Original Brief: {{ brief }}
Art Direction Used: {{ direction }}
Number of Variations: {{ count }}

Provide constructive feedback including:
1. Which directions show the most promise and why
2. Elements that work well across variations
3. Areas that could be pushed further
4. Specific suggestions for the next iteration round
5. Color and composition notes

Be specific and actionable for guiding the next iteration.
""",
        brief=creative_brief.output,
        direction=expanded_direction.out.text,
        count=num_variations.output,
    )

    iteration_feedback = Agent(
        model=LanguageModel(
            type="language_model",
            provider=Provider.OpenAI,
            id="gpt-4o-mini",
        ),
        system="You are a senior concept art director providing feedback for iteration.",
        prompt=feedback_prompt.output,
        max_tokens=1024,
    )

    # --- Outputs ---
    mood_board_out = Output(
        name="mood_board",
        value=mood_board.output,
        description="Combined mood board grid of all concept variations",
    )

    individual_concepts_out = Output(
        name="concept_variations",
        value=collected_concepts.out.output,
        description="Individual concept images for further editing",
    )

    art_direction_out = Output(
        name="art_direction",
        value=expanded_direction.out.text,
        description="Expanded art direction document",
    )

    feedback_out = Output(
        name="iteration_feedback",
        value=iteration_feedback.out.text,
        description="Creative feedback and suggestions for next iteration",
    )

    return create_graph(
        mood_board_out, individual_concepts_out, art_direction_out, feedback_out
    )


# Build the graph
graph = build_concept_art_iteration_board()


if __name__ == "__main__":
    """
    To run this example:

    1. Ensure you have API keys configured for OpenAI and FAL AI
    2. Run:

        python examples/concept_art_iteration_board.py

    The workflow generates a mood board with multiple concept variations
    and provides feedback for iteration.
    """

    print("Concept Art Iteration Board Workflow")
    print("=" * 50)
    print(f"Nodes: {len(graph.nodes)}")
    print(f"Edges: {len(graph.edges)}")
    print()
    print("Workflow pattern:")
    print("  [Creative Brief Inputs]")
    print("      -> [Agent] (expand art direction)")
    print("          -> [ListGenerator] (variation prompts)")
    print("              -> [ForEach] (iterate)")
    print("                  -> [TextToImage] (generate concepts)")
    print("                      -> [Collect]")
    print("                          -> [CombineImageGrid] (mood board)")
    print("                          -> [Agent] (iteration feedback)")
    print("                              -> [Outputs]")
    print()
    print("Outputs:")
    print("  - mood_board: Combined grid of all concepts")
    print("  - concept_variations: Individual images")
    print("  - art_direction: Detailed direction document")
    print("  - iteration_feedback: Suggestions for next round")
    print()

    result = run_graph(graph, asset_output_mode=AssetOutputMode.WORKSPACE)
    print("Mood board generatekpd!")
    print(result)
