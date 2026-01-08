"""
Example: Music Video Visualizer DSL Workflow

This workflow creates reactive visuals from an audio track. It demonstrates:

1. Upload audio track
2. Transcribe with speech recognition (Whisper)
3. Analyze audio for mood/energy using LLM
4. Generate reactive visuals matching the audio mood
5. Create style transfers for genre-specific aesthetics

The workflow pattern:
    [AudioInput] -> [AutomaticSpeechRecognition] -> [Agent] (analyze mood)
                                                       -> [TextToImage] -> [FrameToVideo] -> [AddAudio] -> [Output]

Perfect for music producers creating promotional clips.
"""

from nodetool.dsl.graph import create_graph
from nodetool.dsl.nodetool.input import AudioInput, StringInput
from nodetool.dsl.nodetool.text import FormatText, AutomaticSpeechRecognition
from nodetool.dsl.nodetool.agents import Agent
from nodetool.dsl.nodetool.image import TextToImage
from nodetool.dsl.nodetool.video import FrameToVideo, AddAudio
from nodetool.dsl.nodetool.generators import ListGenerator
from nodetool.dsl.nodetool.control import ForEach, Collect
from nodetool.dsl.nodetool.output import Output
from nodetool.metadata.types import (
    LanguageModel,
    Provider,
    ImageModel,
    AudioRef,
    ASRModel,
)


def build_music_video_visualizer():
    """
    Generate music video visuals from an audio track.

    This function builds a workflow graph that:
    1. Accepts an audio track as input
    2. Transcribes the audio to extract lyrics/content
    3. Analyzes mood, energy, and genre using LLM
    4. Generates visual frames that match the audio's vibe
    5. Compiles frames into a video with the original audio

    Returns:
        Graph: A graph object representing the workflow
    """
    # --- Inputs ---
    audio_track = AudioInput(
        name="audio_track",
        description="The audio track to create visuals for",
        value=AudioRef(type="audio", uri=""),
    )

    genre_hint = StringInput(
        name="genre_hint",
        description="Music genre for style guidance (e.g., electronic, rock, jazz)",
        value="electronic ambient",
    )

    visual_style = StringInput(
        name="visual_style",
        description="Desired visual aesthetic",
        value="abstract geometric patterns, neon colors, flowing energy",
    )

    num_frames = StringInput(
        name="num_frames",
        description="Number of visual frames to generate",
        value="8",
    )

    # --- Transcribe audio ---
    transcription = AutomaticSpeechRecognition(
        model=ASRModel(
            type="asr_model",
            provider=Provider.FalAI,
            id="openai/whisper-large-v3",
            name="Whisper Large V3",
        ),
        audio=audio_track.output,
    )

    # --- Analyze mood and energy ---
    mood_analyzer = FormatText(
        template="""
Analyze this audio content and provide visual inspiration.

Transcribed Lyrics/Speech: {{ transcription }}
Genre: {{ genre }}
Desired Style: {{ style }}

Based on the content, identify:
1. Overall mood (energetic, melancholic, peaceful, intense, etc.)
2. Key emotional themes
3. Visual metaphors that represent the music
4. Color palette suggestions
5. Movement/animation style suggestions

Output a detailed analysis to guide visual generation.
""",
        transcription=transcription.out.text,
        genre=genre_hint.output,
        style=visual_style.output,
    )

    mood_analysis = Agent(
        model=LanguageModel(
            type="language_model",
            provider=Provider.OpenAI,
            id="gpt-4o-mini",
        ),
        system="You are a creative director specializing in music visualization. Analyze audio content and suggest compelling visual concepts.",
        prompt=mood_analyzer.output,
        max_tokens=1024,
    )

    # --- Generate frame prompts ---
    frame_prompt_generator = FormatText(
        template="""
Based on this mood analysis, generate {{ count }} distinct image prompts for video frames.

Mood Analysis: {{ analysis }}
Genre: {{ genre }}
Visual Style: {{ style }}

Each prompt should:
- Create a seamless visual that could loop or transition
- Reflect the music's energy and mood
- Use the suggested color palette
- Be suitable for abstract music visualization
- Progress from one visual theme to another for variety

Output format: One prompt per line, no numbering.
""",
        count=num_frames.output,
        analysis=mood_analysis.out.text,
        genre=genre_hint.output,
        style=visual_style.output,
    )

    frame_prompts = ListGenerator(
        model=LanguageModel(
            type="language_model",
            provider=Provider.OpenAI,
            id="gpt-4o-mini",
        ),
        prompt=frame_prompt_generator.output,
        max_tokens=2048,
    )

    # --- Generate visual frames ---
    prompt_iterator = ForEach(
        input_list=frame_prompts.out.item,
    )

    visual_frame = TextToImage(
        model=ImageModel(
            type="image_model",
            provider=Provider.HuggingFaceFalAI,
            id="fal-ai/flux/schnell",
            name="FLUX.1 Schnell",
        ),
        prompt=prompt_iterator.out.output,
        width=1920,
        height=1080,
        guidance_scale=7.5,
        num_inference_steps=30,
    )

    # --- Collect frames ---
    collected_frames = Collect(
        input_item=visual_frame.output,
    )

    # --- Compile to video ---
    # Note: FrameToVideo expects a stream, this is a simplified representation
    video_output = FrameToVideo(
        frame=visual_frame.output,
        fps=24.0,
    )

    # --- Add original audio ---
    final_video = AddAudio(
        video=video_output.output,
        audio=audio_track.output,
        volume=1.0,
        mix=False,
    )

    # --- Outputs ---
    video_out = Output(
        name="music_video",
        value=final_video.output,
        description="Generated music video with reactive visuals",
    )

    frames_out = Output(
        name="visual_frames",
        value=collected_frames.out.output,
        description="Individual visual frames for further editing",
    )

    mood_out = Output(
        name="mood_analysis",
        value=mood_analysis.out.text,
        description="Mood and energy analysis of the audio",
    )

    return create_graph(video_out, frames_out, mood_out)


# Build the graph
graph = build_music_video_visualizer()


if __name__ == "__main__":
    """
    To run this example:

    1. Ensure you have API keys configured for OpenAI and FAL AI
    2. Provide an audio file
    3. Run:

        python examples/music_video_visualizer.py

    The workflow creates reactive visuals from your audio track.
    """

    print("Music Video Visualizer Workflow")
    print("=" * 50)
    print(f"Nodes: {len(graph.nodes)}")
    print(f"Edges: {len(graph.edges)}")
    print()
    print("Workflow pattern:")
    print("  [AudioInput]")
    print("      -> [AutomaticSpeechRecognition] (transcribe)")
    print("          -> [Agent] (analyze mood/energy)")
    print("              -> [ListGenerator] (frame prompts)")
    print("                  -> [ForEach] (iterate)")
    print("                      -> [TextToImage] (generate frames)")
    print("                          -> [FrameToVideo] (compile)")
    print("                              -> [AddAudio] (add soundtrack)")
    print("                                  -> [Output]")
    print()

    # Uncomment to run:
    # import asyncio
    # result = asyncio.run(run_graph(graph, user_id="example_user", auth_token="token"))
    # print(f"Generated video: {result['music_video']}")
