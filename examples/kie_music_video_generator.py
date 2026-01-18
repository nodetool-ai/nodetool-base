"""
Example: Kie Music Video Generator

This workflow creates complete music videos by generating AI music with Suno
and matching video content with Kie.ai's video models. Similar to Weavy-style
audio-visual creation.

1. Generate original music track with Suno
2. Create visual concept images matching the mood
3. Transform images into dynamic video sequences
4. Combine music and video for final output

The workflow pattern:
    [MusicPrompt] -> [Suno] (generate track)
                        -> [Imagen4/Seedream] (visual concepts)
                            -> [Veo31ImageToVideo] (animate)
                                -> [AddAudio] -> [Output]

Perfect for musicians, content creators, and music video producers.

Note: If imports fail, run 'nodetool package scan && nodetool codegen' to regenerate DSL.
"""

from nodetool.dsl.graph import create_graph, run_graph
from nodetool.dsl.nodetool.input import StringInput, IntegerInput
from nodetool.dsl.nodetool.output import Output
from nodetool.dsl.nodetool.text import FormatText
from nodetool.dsl.nodetool.agents import Agent
from nodetool.dsl.nodetool.generators import ListGenerator
from nodetool.dsl.nodetool.video import AddAudio
from nodetool.dsl.kie.image import (
    Imagen4,
    Seedream45TextToImage,
    FluxKontext,
    NanoBananaPro,
)
from nodetool.dsl.kie.video import (
    Veo31ImageToVideo,
    Kling25TurboImageToVideo,
    HailuoImageToVideoPro,
    Sora2ProImageToVideo,
)
from nodetool.dsl.kie.audio import Suno
from nodetool.metadata.types import LanguageModel, Provider


def build_music_video_generator():
    """
    Generate a complete music video with AI music and video.

    This function builds a workflow graph that:
    1. Creates original music based on genre and mood
    2. Generates visual concepts that match the music's style
    3. Animates images into video sequences
    4. Combines everything into a cohesive music video

    Returns:
        Graph: A graph object representing the workflow
    """
    # --- Music Inputs ---
    song_concept = StringInput(
        name="song_concept",
        description="Concept for the song",
        value="An energetic electronic anthem about chasing dreams and never giving up",
    )

    music_genre = StringInput(
        name="music_genre",
        description="Music genre/style",
        value="synthwave, electronic, uplifting, energetic beats",
    )

    song_lyrics = StringInput(
        name="song_lyrics",
        description="Optional lyrics for the song",
        value="""[Verse 1]
Running through the neon lights
Chasing stars into the night
Every dream within my sight
I won't stop until it's right

[Chorus]
We rise, we fall, we rise again
Through the fire, through the rain
Nothing's gonna break our stride
With the future on our side""",
    )

    # --- Visual Inputs ---
    visual_style = StringInput(
        name="visual_style",
        description="Visual aesthetic for the video",
        value="Cyberpunk neon cityscape, vibrant colors, dramatic lighting, "
        "cinematic composition, futuristic urban environment",
    )

    num_scenes = IntegerInput(
        name="num_scenes",
        description="Number of visual scenes to generate",
        value=4,
        min=2,
        max=8,
    )

    # --- Generate Music Track ---
    music_track = Suno(
        prompt=f"{song_concept.output}. Style: {music_genre.output}",
        lyrics=song_lyrics.output,
        style=Suno.Style.ELECTRONIC,
        instrumental=False,
        duration=120,  # 2 minutes
        model=Suno.Model.V4_5_PLUS,
    )

    # --- Generate Instrumental Version ---
    instrumental_track = Suno(
        prompt=f"{song_concept.output}. Style: {music_genre.output}",
        lyrics="",
        style=Suno.Style.ELECTRONIC,
        instrumental=True,
        duration=120,
        model=Suno.Model.V4_5_PLUS,
    )

    # --- Generate Visual Scene Concepts ---
    scene_prompt = FormatText(
        template="""
Create {{ count }} distinct visual scene prompts for a music video.

Song Concept: {{ concept }}
Music Genre: {{ genre }}
Visual Style: {{ style }}

Each scene should:
- Match the energy and mood of the music
- Be visually striking and cinematic
- Work as a standalone image that can be animated
- Progress through the song narrative
- Be suitable for text-to-image generation

Output format: One detailed image prompt per line, no numbering.
""",
        count=num_scenes.output,
        concept=song_concept.output,
        genre=music_genre.output,
        style=visual_style.output,
    )

    scene_concepts = Agent(
        model=LanguageModel(
            type="language_model",
            provider=Provider.OpenAI,
            id="gpt-4o-mini",
        ),
        system="You are a music video director. Create compelling visual concepts.",
        prompt=scene_prompt.output,
        max_tokens=1024,
    )

    # --- Generate Hero Scene Images ---
    # Scene 1: Opening - Imagen 4
    scene_1_image = Imagen4(
        prompt=f"{visual_style.output}, opening scene, wide establishing shot, "
        "dramatic atmosphere, cinematic lighting, 8K quality",
        aspect_ratio=Imagen4.AspectRatio.LANDSCAPE,
    )

    # Scene 2: Build-up - Seedream
    scene_2_image = Seedream45TextToImage(
        prompt=f"{visual_style.output}, intense action scene, dynamic composition, "
        "energy and movement, vibrant colors, high detail",
        aspect_ratio=Seedream45TextToImage.AspectRatio.LANDSCAPE,
        quality=Seedream45TextToImage.Quality.HIGH,
    )

    # Scene 3: Climax - Flux Kontext
    scene_3_image = FluxKontext(
        prompt=f"{visual_style.output}, climactic moment, peak energy, "
        "explosive visuals, dramatic lighting, powerful composition",
        aspect_ratio=FluxKontext.AspectRatio.LANDSCAPE,
        mode=FluxKontext.Mode.MAX,
    )

    # Scene 4: Resolution - Nano Banana Pro
    scene_4_image = NanoBananaPro(
        prompt=f"{visual_style.output}, triumphant resolution, hopeful atmosphere, "
        "beautiful cinematography, emotional impact, sunset lighting",
        aspect_ratio=NanoBananaPro.AspectRatio.LANDSCAPE,
        resolution=NanoBananaPro.Resolution.RES_4K,
    )

    # --- Animate Scenes to Video ---
    scene_1_video = Veo31ImageToVideo(
        image1=scene_1_image.output,
        prompt="Slow cinematic camera movement, establishing shot, ambient motion",
        model=Veo31ImageToVideo.Model.VEO3,
        aspect_ratio=Veo31ImageToVideo.AspectRatio.RATIO_16_9,
    )

    scene_2_video = Kling25TurboImageToVideo(
        image=scene_2_image.output,
        prompt="Dynamic camera movement, energy and action, building intensity",
        duration=Kling25TurboImageToVideo.Duration.D5,
        cfg_scale=0.6,
    )

    scene_3_video = HailuoImageToVideoPro(
        image=scene_3_image.output,
        prompt="Explosive movement, climactic action, powerful motion",
        duration=HailuoImageToVideoPro.Duration.D6,
        resolution=HailuoImageToVideoPro.Resolution.R1080P,
    )

    scene_4_video = Sora2ProImageToVideo(
        image=scene_4_image.output,
        prompt="Gentle camera pull-back, triumphant resolution, emotional finale",
        n_frames=Sora2ProImageToVideo.Sora2Frames._10s,
        remove_watermark=True,
    )

    # --- Combine with Music ---
    # For demo, we combine one video with the music
    # In production, you'd concatenate all scenes
    final_music_video = AddAudio(
        video=scene_1_video.output,
        audio=music_track.output,
        volume=1.0,
        mix=False,
    )

    instrumental_video = AddAudio(
        video=scene_1_video.output,
        audio=instrumental_track.output,
        volume=1.0,
        mix=False,
    )

    # --- Outputs ---
    main_output = Output(
        name="music_video",
        value=final_music_video.output,
        description="Final music video with vocals",
    )

    instrumental_output = Output(
        name="instrumental_video",
        value=instrumental_video.output,
        description="Music video with instrumental track",
    )

    music_output = Output(
        name="music_track",
        value=music_track.output,
        description="Generated music track with vocals",
    )

    instrumental_audio = Output(
        name="instrumental_audio",
        value=instrumental_track.output,
        description="Instrumental version of the track",
    )

    scene_1_out = Output(
        name="scene_1_video",
        value=scene_1_video.output,
        description="Opening scene video (Veo 3.1)",
    )

    scene_2_out = Output(
        name="scene_2_video",
        value=scene_2_video.output,
        description="Build-up scene video (Kling 2.5)",
    )

    scene_3_out = Output(
        name="scene_3_video",
        value=scene_3_video.output,
        description="Climax scene video (Hailuo Pro)",
    )

    scene_4_out = Output(
        name="scene_4_video",
        value=scene_4_video.output,
        description="Resolution scene video (Sora 2 Pro)",
    )

    scene_concepts_out = Output(
        name="scene_concepts",
        value=scene_concepts.out.text,
        description="Generated scene concepts for reference",
    )

    return create_graph(
        main_output,
        instrumental_output,
        music_output,
        instrumental_audio,
        scene_1_out,
        scene_2_out,
        scene_3_out,
        scene_4_out,
        scene_concepts_out,
    )


# Build the graph
graph = build_music_video_generator()


if __name__ == "__main__":
    """
    To run this example:

    1. Ensure you have KIE_API_KEY configured
    2. Run:

        python examples/kie_music_video_generator.py

    The workflow creates complete AI-generated music videos.
    """

    print("Kie Music Video Generator")
    print("=" * 50)
    print(f"Nodes: {len(graph.nodes)}")
    print(f"Edges: {len(graph.edges)}")
    print()
    print("Kie Models Used:")
    print("  Audio:")
    print("    - Suno V4.5+ - Music generation with vocals")
    print()
    print("  Image Generation:")
    print("    - Imagen 4 (Google) - Opening scene")
    print("    - Seedream 4.5 (Bytedance) - Build-up scene")
    print("    - Flux Kontext (Black Forest Labs) - Climax")
    print("    - Nano Banana Pro (Google/Gemini) - Resolution")
    print()
    print("  Video Generation:")
    print("    - Veo 3.1 (Google) - Opening animation")
    print("    - Kling 2.5 Turbo (Kuaishou) - Dynamic scene")
    print("    - Hailuo Pro (MiniMax) - Action sequence")
    print("    - Sora 2 Pro (OpenAI) - Final scene")
    print()
    print("Workflow pattern:")
    print("  [Song Concept + Lyrics]")
    print("      -> [Suno] (generate music)")
    print("          -> [Imagen4/Seedream/Flux] (scene images)")
    print("              -> [Veo/Kling/Hailuo/Sora] (animate)")
    print("                  -> [AddAudio] (combine)")
    print("                      -> [Output]")
    print()

    # Uncomment to run:
    # result = run_graph(graph)
    # print(result)
