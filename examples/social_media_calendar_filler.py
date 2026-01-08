"""
Example: Social Media Content Calendar Filler DSL Workflow

This workflow generates a batch of social media content for a monthly calendar. It demonstrates:

1. Input monthly themes and brand guidelines
2. Generate diverse content ideas for the month
3. Create images and video concepts with brand filters
4. Auto-generate captions and hashtags
5. Export scheduled assets with metadata

The workflow pattern:
    [StringInputs] -> [Agent] (content plan) -> [ListGenerator] (content ideas)
                        -> [ForEach] -> [TextToImage] -> [TextToSpeech] -> [Collect] -> [Output]

Streamlines workflow for marketers and content creators.
"""

from nodetool.dsl.graph import create_graph
from nodetool.dsl.nodetool.input import StringInput, IntegerInput
from nodetool.dsl.nodetool.text import FormatText
from nodetool.dsl.nodetool.agents import Agent
from nodetool.dsl.nodetool.generators import ListGenerator
from nodetool.dsl.nodetool.image import TextToImage
from nodetool.dsl.nodetool.control import ForEach, Collect
from nodetool.dsl.nodetool.output import Output
from nodetool.metadata.types import (
    LanguageModel,
    Provider,
    ImageModel,
)


def build_social_media_calendar_filler():
    """
    Generate a month's worth of social media content.

    This function builds a workflow graph that:
    1. Accepts monthly themes and posting frequency
    2. Uses LLM to plan content for the month
    3. Generates images for each post
    4. Creates audio narration for video content
    5. Generates captions and hashtags
    6. Returns a complete content calendar

    Returns:
        Graph: A graph object representing the workflow
    """
    # --- Inputs ---
    brand_name = StringInput(
        name="brand_name",
        description="Brand or account name",
        value="TechStartup Pro",
    )

    monthly_theme = StringInput(
        name="monthly_theme",
        description="Main theme for the month's content",
        value="AI productivity tools and workflow automation",
    )

    target_audience = StringInput(
        name="target_audience",
        description="Who the content is for",
        value="Tech professionals, entrepreneurs, and startup founders aged 25-45",
    )

    brand_voice = StringInput(
        name="brand_voice",
        description="Brand voice and tone",
        value="Professional yet approachable, educational, inspiring, data-driven",
    )

    posts_per_week = IntegerInput(
        name="posts_per_week",
        description="Number of posts to generate per week",
        value=4,
        min=1,
        max=7,
    )

    # --- Generate content calendar plan ---
    calendar_prompt = FormatText(
        template="""
You are a social media content strategist.

Brand: {{ brand }}
Theme: {{ theme }}
Target Audience: {{ audience }}
Voice: {{ voice }}
Posts per Week: {{ frequency }}

Create a 4-week content calendar with {{ frequency }} posts per week ({{ total }} posts total).

For each post, provide:
1. Post type (image, carousel, reel concept, story)
2. Topic/angle
3. Key message
4. Best posting day/time suggestion
5. Content pillar (educational, promotional, engagement, behind-scenes)

Output format: Return as a structured list, one post per line with details separated by |
Format: PostType|Topic|KeyMessage|SuggestedDay|Pillar
""",
        brand=brand_name.output,
        theme=monthly_theme.output,
        audience=target_audience.output,
        voice=brand_voice.output,
        frequency=posts_per_week.output,
        total="16",  # 4 posts/week * 4 weeks
    )

    content_plan = Agent(
        model=LanguageModel(
            type="language_model",
            provider=Provider.OpenAI,
            id="gpt-4o-mini",
        ),
        system="You are a social media strategist. Create detailed, actionable content plans.",
        prompt=calendar_prompt.output,
        max_tokens=2048,
    )

    # --- Generate detailed content for each post ---
    content_generator = FormatText(
        template="""
Based on this content plan, generate 16 detailed social media post prompts.

Content Plan: {{ plan }}
Brand Voice: {{ voice }}
Theme: {{ theme }}

For each post prompt:
- Create an image generation prompt (visual description)
- Be brand-consistent and on-theme
- Make visuals scroll-stopping and engaging

Output format: One image prompt per line, no numbering.
""",
        plan=content_plan.out.text,
        voice=brand_voice.output,
        theme=monthly_theme.output,
    )

    post_prompts = ListGenerator(
        model=LanguageModel(
            type="language_model",
            provider=Provider.OpenAI,
            id="gpt-4o-mini",
        ),
        prompt=content_generator.output,
        max_tokens=3000,
    )

    # --- Generate images for each post ---
    prompt_iterator = ForEach(
        input_list=post_prompts.out.item,
    )

    post_image = TextToImage(
        model=ImageModel(
            type="image_model",
            provider=Provider.HuggingFaceFalAI,
            id="fal-ai/flux/schnell",
            name="FLUX.1 Schnell",
        ),
        prompt=prompt_iterator.out.output,
        width=1080,  # Instagram square
        height=1080,
        guidance_scale=7.0,
        num_inference_steps=25,
    )

    # --- Collect all post images ---
    collected_images = Collect(
        input_item=post_image.output,
    )

    # --- Generate captions and hashtags ---
    caption_generator = FormatText(
        template="""
Generate captions and hashtags for social media posts.

Content Plan: {{ plan }}
Brand: {{ brand }}
Voice: {{ voice }}

For each planned post, create:
1. An engaging caption (2-3 sentences)
2. A call-to-action
3. 5-10 relevant hashtags

Output format: One complete caption block per line (caption + hashtags combined).
""",
        plan=content_plan.out.text,
        brand=brand_name.output,
        voice=brand_voice.output,
    )

    captions = ListGenerator(
        model=LanguageModel(
            type="language_model",
            provider=Provider.OpenAI,
            id="gpt-4o-mini",
        ),
        prompt=caption_generator.output,
        max_tokens=2000,
    )

    # --- Collect captions ---
    collected_captions = Collect(
        input_item=captions.out.item,
    )

    # --- Outputs ---
    images_out = Output(
        name="post_images",
        value=collected_images.out.output,
        description="Generated images for each social media post",
    )

    captions_out = Output(
        name="captions_and_hashtags",
        value=collected_captions.out.output,
        description="Captions and hashtags for each post",
    )

    plan_out = Output(
        name="content_calendar",
        value=content_plan.out.text,
        description="Full content calendar plan for the month",
    )

    return create_graph(images_out, captions_out, plan_out)


# Build the graph
graph = build_social_media_calendar_filler()


if __name__ == "__main__":
    """
    To run this example:

    1. Ensure you have API keys configured for OpenAI and FAL AI
    2. Run:

        python examples/social_media_calendar_filler.py

    The workflow generates a complete month of social media content.
    """

    print("Social Media Content Calendar Filler Workflow")
    print("=" * 50)
    print(f"Nodes: {len(graph.nodes)}")
    print(f"Edges: {len(graph.edges)}")
    print()
    print("Workflow pattern:")
    print("  [Brand/Theme Inputs]")
    print("      -> [Agent] (create content calendar)")
    print("          -> [ListGenerator] (post prompts)")
    print("              -> [ForEach] (iterate)")
    print("                  -> [TextToImage] (generate post images)")
    print("          -> [ListGenerator] (captions/hashtags)")
    print("              -> [Collect]")
    print("                  -> [Outputs]")
    print()
    print("Outputs:")
    print("  - post_images: Generated visuals for 16 posts")
    print("  - captions_and_hashtags: Ready-to-use copy")
    print("  - content_calendar: Strategic monthly plan")
    print()

    # Uncomment to run:
    # import asyncio
    # result = asyncio.run(run_graph(graph, user_id="example_user", auth_token="token"))
    # print(f"Generated {len(result['post_images'])} post images")
