"""
Example: Audio-Reactive Album Cover Creator DSL Workflow

This workflow creates album covers that visually represent the music. It demonstrates:

1. Analyze uploaded music for mood and energy
2. Extract audio characteristics using LLM analysis
3. Generate matching abstract or illustrative covers
4. Create multiple variations synced to musical themes
5. Support animated versions for digital releases

The workflow pattern:
    [AudioInput] -> [AutomaticSpeechRecognition] -> [Agent] (analyze music)
                        -> [ListGenerator] (cover concepts) -> [ForEach] -> [TextToImage]
                            -> [Collect] -> [Output]

Fantastic for music producers visualizing their tracks.
"""

from typing import List

from nodetool.dsl.graph import create_graph, run_graph
from nodetool.dsl.nodetool.input import AudioInput, StringInput, IntegerInput
from nodetool.dsl.nodetool.text import FormatText, AutomaticSpeechRecognition
from nodetool.dsl.nodetool.agents import Agent, Extractor
from nodetool.dsl.nodetool.generators import ListGenerator
from nodetool.dsl.nodetool.image import TextToImage
from nodetool.dsl.lib.pillow.draw import RenderText
from nodetool.dsl.lib.pillow.enhance import Contrast
from nodetool.dsl.nodetool.control import ForEach, Collect
from nodetool.dsl.nodetool.output import Output
from nodetool.metadata.types import (
    LanguageModel,
    Provider,
    ImageModel,
    AudioRef,
    ASRModel,
    ColorRef,
    FontRef,
)


def build_album_cover_creator():
    """
    Generate album covers that match the music's mood and energy.

    This function builds a workflow graph that:
    1. Accepts an audio track as input
    2. Transcribes any vocals/speech in the audio
    3. Analyzes the music's mood, energy, and genre
    4. Generates album cover concepts that visually represent the sound
    5. Creates multiple cover variations

    Returns:
        Graph: A graph object representing the workflow
    """
    # --- Inputs ---
    audio_track = AudioInput(
        name="audio_track",
        description="The music track to create a cover for",
        value=AudioRef(type="audio", uri=""),
    )

    artist_name = StringInput(
        name="artist_name",
        description="Artist or band name",
        value="Neon Dreams",
    )

    album_title = StringInput(
        name="album_title",
        description="Album or single title",
        value="Electric Horizons",
    )

    genre_hint = StringInput(
        name="genre_hint",
        description="Music genre for context",
        value="Electronic / Synthwave",
    )

    cover_style = StringInput(
        name="cover_style",
        description="Preferred visual style",
        value="abstract, geometric, neon colors, retrofuturistic",
    )

    num_variations = IntegerInput(
        name="num_variations",
        description="Number of cover variations",
        value=4,
        min=2,
        max=8,
    )

    # --- Transcribe audio for lyrical content ---
    transcription = AutomaticSpeechRecognition(
        model=ASRModel(
            type="asr_model",
            provider=Provider.FalAI,
            id="openai/whisper-large-v3",
            name="Whisper Large V3",
        ),
        audio=audio_track.output,
    )

    # --- Analyze music characteristics ---
    music_analysis_prompt = FormatText(
        template="""
Analyze this music for album cover design purposes.

Genre: {{ genre }}
Lyrics/Vocals (if any): {{ lyrics }}
Artist: {{ artist }}
Title: {{ title }}
Preferred Style: {{ style }}

Provide a detailed analysis including:
1. Overall mood and emotion (energetic, melancholic, euphoric, dark, uplifting)
2. Energy level (calm, moderate, intense, explosive)
3. Visual themes that match the sound
4. Color palette that represents the music
5. Imagery suggestions (abstract, figurative, symbolic)
6. Texture and pattern ideas
7. Key visual metaphors for the music's feeling

Be creative and specific for album cover design.
""",
        genre=genre_hint.output,
        lyrics=transcription.out.text,
        artist=artist_name.output,
        title=album_title.output,
        style=cover_style.output,
    )

    music_analysis = Agent(
        model=LanguageModel(
            type="language_model",
            provider=Provider.OpenAI,
            id="gpt-4o-mini",
        ),
        system="You are a creative director specializing in album artwork. Analyze music and suggest compelling visual concepts.",
        prompt=music_analysis_prompt.output,
        max_tokens=1024,
    )

    # --- Generate cover concept prompts ---
    cover_prompt_generator = FormatText(
        template="""
Based on this music analysis, generate {{ count }} unique album cover prompts.

Music Analysis: {{ analysis }}
Artist: {{ artist }}
Album: {{ title }}
Style: {{ style }}

Each prompt should:
- Create a complete album cover composition (no text)
- Be visually striking and memorable
- Reflect the music's mood and energy
- Work as a square 1:1 ratio image
- Use suggested colors and visual themes
- Leave appropriate space for artist/album text overlay

Output format: One detailed prompt per line, no numbering.
Make each variation distinct but thematically cohesive.
""",
        count=num_variations.output,
        analysis=music_analysis.out.text,
        artist=artist_name.output,
        title=album_title.output,
        style=cover_style.output,
    )

    cover_prompts = ListGenerator(
        model=LanguageModel(
            type="language_model",
            provider=Provider.OpenAI,
            id="gpt-4o-mini",
        ),
        prompt=cover_prompt_generator.output,
        max_tokens=2048,
    )

    # --- Generate cover images ---
    prompt_iterator = ForEach(
        input_list=cover_prompts.out.item,
    )

    base_cover = TextToImage(
        model=ImageModel(
            type="image_model",
            provider=Provider.HuggingFaceFalAI,
            id="fal-ai/flux/dev",
            name="FLUX.1 Dev",
        ),
        prompt=prompt_iterator.out.output,
        width=1400,  # High-res square for album covers
        height=1400,
        guidance_scale=7.5,
        num_inference_steps=35,
    )

    # --- Enhance for final output ---
    enhanced = Contrast(
        image=base_cover.output,
        factor=1.1,
    )

    # --- Add artist name ---
    with_artist = RenderText(
        image=enhanced.output,
        text=artist_name.output,
        x=70,
        y=70,
        size=56,
        color=ColorRef(type="color", value="#FFFFFF"),
        font=FontRef(type="font", name="DejaVuSans-Bold"),
    )

    # --- Add album title ---
    final_cover = RenderText(
        image=with_artist.output,
        text=album_title.output,
        x=70,
        y=1280,
        size=48,
        color=ColorRef(type="color", value="#FFFFFF"),
        font=FontRef(type="font", name="DejaVuSans"),
    )

    # --- Collect all cover variations ---
    collected_covers = Collect(
        input_item=final_cover.output,
    )

    # --- Also collect base covers without text ---
    collected_base = Collect(
        input_item=enhanced.output,
    )

    # --- Outputs ---
    covers_out = Output(
        name="album_covers",
        value=collected_covers.out.output,
        description="Album covers with artist/title text",
    )

    base_covers_out = Output(
        name="base_artwork",
        value=collected_base.out.output,
        description="Base artwork without text (for custom typography)",
    )

    analysis_out = Output(
        name="music_analysis",
        value=music_analysis.out.text,
        description="Music analysis and visual concept brief",
    )

    return create_graph(covers_out, base_covers_out, analysis_out)


# Build the graph
graph = build_album_cover_creator()


if __name__ == "__main__":
    """
    To run this example:

    1. Prepare an audio file (music track)
    2. Ensure you have API keys configured for OpenAI and FAL AI
    3. Run:

        python examples/album_cover_creator.py

    The workflow creates album covers that visually represent your music.
    """

    print("Audio-Reactive Album Cover Creator Workflow")
    print("=" * 50)
    print(f"Nodes: {len(graph.nodes)}")
    print(f"Edges: {len(graph.edges)}")
    print()
    print("Workflow pattern:")
    print("  [AudioInput]")
    print("      -> [AutomaticSpeechRecognition] (extract lyrics)")
    print("          -> [Agent] (analyze mood/energy)")
    print("              -> [ListGenerator] (cover concepts)")
    print("                  -> [ForEach] (iterate)")
    print("                      -> [TextToImage] (generate covers)")
    print("                          -> [Contrast] (enhance)")
    print("                              -> [RenderText] (add text)")
    print("                                  -> [Collect]")
    print("                                      -> [Outputs]")
    print()
    print("Outputs:")
    print("  - album_covers: Complete covers with text")
    print("  - base_artwork: Clean versions for custom typography")
    print("  - music_analysis: Visual concept brief")
    print()

    # Uncomment to run:
    # import asyncio
    # result = asyncio.run(run_graph(graph, user_id="example_user", auth_token="token"))
    # print(f"Generated {len(result['album_covers'])} album cover variations")
