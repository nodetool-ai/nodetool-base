"""
Example: Video B-Roll Automation DSL Workflow

This workflow generates complementary B-roll footage from a main video. It demonstrates:

1. Feed a main video clip
2. Extract frames and analyze content
3. Generate complementary B-roll (transitions, overlays, effects)
4. Apply consistent visual styles
5. Compile seamlessly with the main footage

The workflow pattern:
    [VideoInput] -> [FrameIterator] -> [Agent] (analyze content)
                        -> [ListGenerator] (B-roll concepts) -> [ForEach] -> [TextToImage]
                            -> [FrameToVideo] -> [Transition] -> [Output]

Accelerates editing for video creators and filmmakers.
"""

from typing import List

from nodetool.dsl.graph import create_graph, run_graph
from nodetool.dsl.nodetool.input import VideoInput, StringInput, IntegerInput
from nodetool.dsl.nodetool.text import FormatText
from nodetool.dsl.nodetool.agents import Agent
from nodetool.dsl.nodetool.generators import ListGenerator
from nodetool.dsl.nodetool.image import TextToImage
from nodetool.dsl.nodetool.video import (
    FrameIterator,
    FrameToVideo,
    Concat,
    Transition,
    ExtractAudio,
    AddAudio,
)
from nodetool.dsl.lib.pillow.enhance import Contrast
from nodetool.dsl.nodetool.control import ForEach, Collect
from nodetool.dsl.nodetool.output import Output
from nodetool.metadata.types import (
    LanguageModel,
    Provider,
    ImageModel,
    VideoRef,
)


def build_video_broll_automation():
    """
    Generate complementary B-roll for video editing.

    This function builds a workflow graph that:
    1. Accepts a main video clip
    2. Extracts key frames for content analysis
    3. Analyzes the video content using LLM
    4. Generates B-roll concepts that complement the main footage
    5. Creates B-roll clips and transitions

    Returns:
        Graph: A graph object representing the workflow
    """
    # --- Inputs ---
    main_video = VideoInput(
        name="main_video",
        description="The main video clip to generate B-roll for",
        value=VideoRef(type="video", uri=""),
    )

    video_topic = StringInput(
        name="video_topic",
        description="Brief description of the video content",
        value="Tech product review showcasing a new smartphone with close-up shots and demonstrations",
    )

    broll_style = StringInput(
        name="broll_style",
        description="Desired B-roll visual style",
        value="Clean, modern, tech aesthetic, smooth transitions, subtle motion",
    )

    num_broll_clips = IntegerInput(
        name="num_broll_clips",
        description="Number of B-roll clips to generate",
        value=5,
        min=2,
        max=10,
    )

    # --- Extract frames from main video for analysis ---
    frame_extractor = FrameIterator(
        video=main_video.output,
        start=0,
        end=10,  # Extract first 10 frames for analysis
    )

    # --- Extract audio for later ---
    original_audio = ExtractAudio(
        video=main_video.output,
    )

    # --- Analyze video content ---
    analysis_prompt = FormatText(
        template="""
Analyze this video content for B-roll generation.

Video Topic: {{ topic }}
Style Preference: {{ style }}

Based on the video topic, identify:
1. Key subjects/objects that need visual support
2. Actions or processes that could use cutaway shots
3. Mood and atmosphere of the main content
4. Color palette and visual tone
5. Types of B-roll that would enhance the narrative
   - Establishing shots
   - Detail/close-up shots
   - Motion graphics/overlays
   - Transition elements
   - Abstract/mood shots

Provide specific B-roll suggestions that would complement this content.
""",
        topic=video_topic.output,
        style=broll_style.output,
    )

    content_analysis = Agent(
        model=LanguageModel(
            type="language_model",
            provider=Provider.OpenAI,
            id="gpt-4o-mini",
        ),
        system="You are a professional video editor. Analyze content and suggest compelling B-roll concepts.",
        prompt=analysis_prompt.output,
        max_tokens=1024,
    )

    # --- Generate B-roll clip prompts ---
    broll_prompt_generator = FormatText(
        template="""
Based on this content analysis, generate {{ count }} B-roll image prompts.

Content Analysis: {{ analysis }}
Style: {{ style }}
Topic: {{ topic }}

Each prompt should describe a scene/image that:
- Complements the main video content
- Works as a short B-roll clip (will be made into video)
- Matches the visual style and mood
- Could be used as a cutaway or transition shot
- Is suitable for looping or short clip use

Types to include:
- Establishing/wide shots
- Detail close-ups
- Abstract patterns or textures
- Motion-implied compositions
- Atmosphere/mood shots

Output format: One detailed prompt per line, no numbering.
""",
        count=num_broll_clips.output,
        analysis=content_analysis.out.text,
        style=broll_style.output,
        topic=video_topic.output,
    )

    broll_prompts = ListGenerator(
        model=LanguageModel(
            type="language_model",
            provider=Provider.OpenAI,
            id="gpt-4o-mini",
        ),
        prompt=broll_prompt_generator.output,
        max_tokens=2048,
    )

    # --- Generate B-roll frames ---
    prompt_iterator = ForEach(
        input_list=broll_prompts.out.item,
    )

    broll_frame = TextToImage(
        model=ImageModel(
            type="image_model",
            provider=Provider.HuggingFaceFalAI,
            id="fal-ai/flux/schnell",
            name="FLUX.1 Schnell",
        ),
        prompt=prompt_iterator.out.output,
        width=1920,
        height=1080,  # Standard HD video resolution
        guidance_scale=7.0,
        num_inference_steps=25,
    )

    # --- Enhance frames ---
    enhanced_frame = Contrast(
        image=broll_frame.output,
        factor=1.05,
    )

    # --- Collect all B-roll frames ---
    collected_broll = Collect(
        input_item=enhanced_frame.output,
    )

    # --- Convert frames to video clips ---
    # Note: In a full implementation, each frame would be
    # duplicated or interpolated to create short clips
    broll_video = FrameToVideo(
        frame=enhanced_frame.output,
        fps=30.0,
    )

    # --- Create transition between main and B-roll ---
    # Transition.TransitionType provides fade, wipe, etc.
    transitioned = Transition(
        video_a=main_video.output,
        video_b=broll_video.output,
        transition_type=Transition.TransitionType.fade,
        duration=0.5,
    )

    # --- Outputs ---
    broll_frames_out = Output(
        name="broll_frames",
        value=collected_broll.out.output,
        description="Individual B-roll frames for manual editing",
    )

    broll_video_out = Output(
        name="broll_clips",
        value=broll_video.output,
        description="B-roll compiled as video clips",
    )

    transitioned_out = Output(
        name="with_transition",
        value=transitioned.output,
        description="Main video with B-roll transition example",
    )

    analysis_out = Output(
        name="content_analysis",
        value=content_analysis.out.text,
        description="Content analysis for editing reference",
    )

    return create_graph(broll_frames_out, broll_video_out, transitioned_out, analysis_out)


# Build the graph
graph = build_video_broll_automation()


if __name__ == "__main__":
    """
    To run this example:

    1. Prepare a main video file
    2. Ensure you have API keys configured for OpenAI and FAL AI
    3. Run:

        python examples/video_broll_automation.py

    The workflow generates complementary B-roll for your video content.
    """

    print("Video B-Roll Automation Workflow")
    print("=" * 50)
    print(f"Nodes: {len(graph.nodes)}")
    print(f"Edges: {len(graph.edges)}")
    print()
    print("Workflow pattern:")
    print("  [VideoInput]")
    print("      -> [FrameIterator] (extract frames)")
    print("      -> [ExtractAudio] (preserve audio)")
    print("          -> [Agent] (analyze content)")
    print("              -> [ListGenerator] (B-roll concepts)")
    print("                  -> [ForEach] (iterate)")
    print("                      -> [TextToImage] (generate B-roll)")
    print("                          -> [Contrast] (enhance)")
    print("                              -> [FrameToVideo] (create clips)")
    print("                                  -> [Transition] (blend)")
    print("                                      -> [Outputs]")
    print()
    print("B-Roll Types:")
    print("  - Establishing/wide shots")
    print("  - Detail close-ups")
    print("  - Abstract patterns/textures")
    print("  - Motion-implied compositions")
    print("  - Atmosphere/mood shots")
    print()

    # Uncomment to run:
    # import asyncio
    # result = asyncio.run(run_graph(graph, user_id="example_user", auth_token="token"))
    # print(f"Generated {len(result['broll_frames'])} B-roll frames")
