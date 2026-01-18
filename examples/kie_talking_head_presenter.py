"""
Example: Kie Talking Head Presenter

This workflow creates AI-powered talking head videos using Kie.ai's avatar
generation. Perfect for creating explainer videos, tutorials, and presentations.

1. Generate or provide a presenter image
2. Provide or generate audio for the script
3. Animate the presenter with lip-sync using Kling Avatar
4. Add background music with Suno

The workflow pattern:
    [ImageInput/TextToImage] -> [AudioInput] (script audio)
                                   -> [KlingAIAvatarPro] (lip-sync)
                                       -> [Suno] (background music)
                                           -> [AddAudio] -> [Output]

Ideal for corporate training, YouTube creators, and marketing presentations.

Note: If you need AI text-to-speech, run 'nodetool package scan && nodetool codegen'
to get access to ElevenLabsTextToSpeech after DSL regeneration.
"""

from nodetool.dsl.graph import create_graph, run_graph
from nodetool.dsl.nodetool.input import StringInput, ImageInput, AudioInput
from nodetool.dsl.nodetool.output import Output
from nodetool.dsl.kie.image import (
    Flux2ProTextToImage,
    RecraftRemoveBackground,
)
from nodetool.dsl.kie.video import (
    KlingAIAvatarPro,
    KlingAIAvatarStandard,
    KlingImageToVideo,
)
from nodetool.dsl.kie.audio import Suno
from nodetool.dsl.nodetool.video import AddAudio
from nodetool.metadata.types import ImageRef, AudioRef


def build_talking_head_presenter():
    """
    Create an AI talking head presenter video.

    This function builds a workflow graph that:
    1. Generates a professional presenter portrait or uses provided image
    2. Uses provided audio narration for lip-sync
    3. Animates the presenter with Kling AI Avatar
    4. Adds background music with Suno
    5. Outputs the final presentation video

    Returns:
        Graph: A graph object representing the workflow
    """
    # --- Inputs ---
    presenter_image = ImageInput(
        name="presenter_image",
        description="Portrait image of the presenter (optional - will generate if not provided)",
        value=ImageRef(type="image", uri=""),
    )

    presenter_description = StringInput(
        name="presenter_description",
        description="Description of the presenter to generate",
        value="Professional female business presenter, mid-30s, friendly smile, "
        "wearing a navy blazer, neutral office background, high quality portrait, "
        "looking directly at camera, professional headshot style",
    )

    script_audio = AudioInput(
        name="script_audio",
        description="Pre-recorded audio narration for the presenter",
        value=AudioRef(type="audio", uri=""),
    )

    music_style = StringInput(
        name="music_style",
        description="Style of background music",
        value="Corporate motivational, uplifting, subtle, professional background music",
    )

    # --- Generate Presenter Image (if not provided) ---
    generated_presenter = Flux2ProTextToImage(
        prompt=presenter_description.output,
        aspect_ratio=Flux2ProTextToImage.AspectRatio.PORTRAIT,
        resolution=Flux2ProTextToImage.Resolution.RES_2K,
        steps=30,
        guidance_scale=8.0,
    )

    # --- Remove Background for cleaner composite ---
    clean_presenter = RecraftRemoveBackground(
        image=generated_presenter.output,
    )

    # --- Create Talking Head Video with Kling Avatar Pro ---
    talking_head_pro = KlingAIAvatarPro(
        image=generated_presenter.output,
        audio=script_audio.output,
        prompt="Professional presenter speaking confidently, natural gestures, "
        "slight head movements, engaging eye contact, business presentation style",
        mode=KlingAIAvatarPro.Mode.PRO,
    )

    # --- Alternative: Standard quality for faster processing ---
    talking_head_standard = KlingAIAvatarStandard(
        image=generated_presenter.output,
        audio=script_audio.output,
        prompt="Professional presenter speaking naturally, slight movements, "
        "engaging delivery, corporate presentation",
        mode=KlingAIAvatarStandard.Mode.STANDARD,
    )

    # --- Alternative: Simple image animation without audio ---
    simple_animation = KlingImageToVideo(
        prompt="Professional presenter with subtle head movements, "
        "slight nodding, engaging eye contact, professional pose",
        image1=generated_presenter.output,
        duration=5,
        sound=False,
    )

    # --- Generate Background Music ---
    background_music = Suno(
        prompt=music_style.output,
        style=Suno.Style.AMBIENT,
        instrumental=True,
        duration=60,  # Match approximate video length
        model=Suno.Model.V4_5_PLUS,
    )

    # --- Combine Video with Background Music ---
    final_video_with_music = AddAudio(
        video=talking_head_pro.output,
        audio=background_music.output,
        volume=0.15,  # Low volume for background
        mix=True,  # Mix with existing audio
    )

    # --- Outputs ---
    main_output = Output(
        name="presentation_video",
        value=final_video_with_music.output,
        description="Final presentation video with talking head and background music",
    )

    video_only = Output(
        name="video_without_music",
        value=talking_head_pro.output,
        description="Talking head video without background music",
    )

    standard_quality = Output(
        name="video_standard_quality",
        value=talking_head_standard.output,
        description="Standard quality version (faster processing)",
    )

    simple_output = Output(
        name="simple_animation",
        value=simple_animation.output,
        description="Simple presenter animation without lip-sync",
    )

    presenter_portrait = Output(
        name="presenter_image",
        value=generated_presenter.output,
        description="Generated presenter portrait for reuse",
    )

    music_track = Output(
        name="background_music",
        value=background_music.output,
        description="Background music track",
    )

    return create_graph(
        main_output,
        video_only,
        standard_quality,
        simple_output,
        presenter_portrait,
        music_track,
    )


# Build the graph
graph = build_talking_head_presenter()


if __name__ == "__main__":
    """
    To run this example:

    1. Ensure you have KIE_API_KEY configured
    2. Provide a pre-recorded audio file for the narration
    3. Run:

        python examples/kie_talking_head_presenter.py

    The workflow creates AI talking head presentation videos.
    """

    print("Kie Talking Head Presenter")
    print("=" * 50)
    print(f"Nodes: {len(graph.nodes)}")
    print(f"Edges: {len(graph.edges)}")
    print()
    print("Kie Models Used:")
    print("  - Flux 2 Pro Text-to-Image - Presenter generation")
    print("  - Recraft Remove Background - Clean composite")
    print("  - Kling AI Avatar Pro/Standard - Lip-sync animation")
    print("  - Kling Image-to-Video - Simple animation")
    print("  - Suno - Background music")
    print()
    print("Workflow pattern:")
    print("  [Presenter Description]")
    print("      -> [Flux2ProTextToImage] (generate presenter)")
    print("          -> [AudioInput] (pre-recorded narration)")
    print("              -> [KlingAIAvatarPro] (lip-sync)")
    print("                  -> [Suno] (background music)")
    print("                      -> [AddAudio] (combine)")
    print("                          -> [Output]")
    print()
    print("Use Cases:")
    print("  - Corporate training videos")
    print("  - YouTube explainer content")
    print("  - Product demos and tutorials")
    print("  - Marketing presentations")
    print("  - Educational content")
    print()

    # Uncomment to run:
    # result = run_graph(graph)
    # print(result)
