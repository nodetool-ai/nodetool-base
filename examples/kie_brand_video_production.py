"""
Example: Kie Brand Video Production Pipeline

This workflow creates professional brand videos using Kie.ai's full suite
of video generation capabilities. Demonstrates storyboard-to-video, multi-shot
editing, and motion control features.

1. Generate storyboard frames from concept
2. Create video sequences using multiple models
3. Apply video-to-video style transfer
4. Combine scenes with transitions and audio

The workflow pattern:
    [BrandConcept] -> [Image Generation] (storyboard frames)
                         -> [Sora2ProStoryboard] (multi-shot video)
                         -> [Wan26VideoToVideo] (style transfer)
                             -> [AddAudio] -> [Output]

Perfect for brand agencies, video producers, and creative directors.

Note: If imports fail, run 'nodetool package scan && nodetool codegen' to regenerate DSL.
"""

from nodetool.dsl.graph import create_graph, run_graph
from nodetool.dsl.nodetool.input import StringInput, VideoInput
from nodetool.dsl.nodetool.output import Output
from nodetool.dsl.nodetool.text import FormatText
from nodetool.dsl.nodetool.video import AddAudio, Transition
from nodetool.dsl.kie.image import (
    Seedream45TextToImage,
    NanoBananaPro,
    Imagen4,
)
from nodetool.dsl.kie.video import (
    Sora2ProStoryboard,
    Sora2ProTextToVideo,
    WanMultiShotTextToVideoPro,
    Veo31ReferenceToVideo,
    TopazVideoUpscale,
)
from nodetool.dsl.kie.audio import Suno
from nodetool.metadata.types import ImageRef


def build_brand_video_production():
    """
    Create professional brand videos with Kie AI.

    This function builds a workflow graph that:
    1. Generates storyboard frames for brand narrative
    2. Creates cohesive multi-shot videos
    3. Applies style transfer and enhancement
    4. Adds professional audio

    Returns:
        Graph: A graph object representing the workflow
    """
    # --- Inputs ---
    brand_name = StringInput(
        name="brand_name",
        description="Brand name",
        value="Horizon Ventures",
    )

    brand_story = StringInput(
        name="brand_story",
        description="Brand narrative for the video",
        value="A journey of innovation and discovery. From bold ideas to breakthrough solutions. "
        "We help visionaries transform their dreams into reality. "
        "Building tomorrow's possibilities, today.",
    )

    visual_identity = StringInput(
        name="visual_identity",
        description="Brand visual identity",
        value="Modern corporate aesthetic, deep blue and gold color palette, "
        "clean geometric shapes, inspirational imagery, professional yet dynamic",
    )

    video_tone = StringInput(
        name="video_tone",
        description="Emotional tone of the video",
        value="Inspiring, professional, forward-thinking, confident, warm",
    )

    narration_script = StringInput(
        name="narration_script",
        description="Voice-over narration",
        value="At Horizon Ventures, we believe in the power of vision. "
        "Every great achievement starts with a bold idea. "
        "We partner with innovators to turn possibilities into reality. "
        "Join us on the journey to tomorrow.",
    )

    # --- Generate Storyboard Frames ---
    # Frame 1: Opening - Brand essence
    frame_1 = Seedream45TextToImage(
        prompt=f"{brand_name.output} brand opening, abstract geometric design, "
        f"{visual_identity.output}, dramatic lighting, cinematic composition, "
        "dawn horizon, new beginnings, 4K quality",
        aspect_ratio=Seedream45TextToImage.AspectRatio.LANDSCAPE,
        quality=Seedream45TextToImage.Quality.HIGH,
    )

    # Frame 2: Challenge - The problem
    frame_2 = Imagen4(
        prompt=f"Abstract representation of complex challenges transforming into opportunities, "
        f"{visual_identity.output}, dynamic motion blur, transitional imagery, "
        "professional corporate photography style",
        aspect_ratio=Imagen4.AspectRatio.LANDSCAPE,
    )

    # Frame 3: Solution - The approach
    frame_3 = NanoBananaPro(
        prompt=f"Innovation and collaboration, hands working together, technology and humanity, "
        f"{visual_identity.output}, warm lighting, connection and partnership, "
        "high-end corporate imagery",
        aspect_ratio=NanoBananaPro.AspectRatio.LANDSCAPE,
        resolution=NanoBananaPro.Resolution.RES_4K,
    )

    # Frame 4: Resolution - The future
    frame_4 = Seedream45TextToImage(
        prompt=f"Triumphant vision of the future, expansive horizon, golden light, "
        f"{visual_identity.output}, success and achievement, inspirational, "
        "epic cinematic landscape, brand finale",
        aspect_ratio=Seedream45TextToImage.AspectRatio.LANDSCAPE,
        quality=Seedream45TextToImage.Quality.HIGH,
    )

    # --- Create Multi-Shot Video with Sora 2 Pro Storyboard ---
    storyboard_video = Sora2ProStoryboard(
        prompt=f"Cinematic brand video for {brand_name.output}, {brand_story.output}, "
        f"{video_tone.output}, smooth transitions between scenes, "
        "professional corporate video, cohesive narrative flow",
        image1=frame_1.output,
        image2=frame_2.output,
        image3=frame_3.output,
        aspect_ratio=Sora2ProStoryboard.AspectRatio.LANDSCAPE,
        n_frames=Sora2ProStoryboard.Sora2Frames._15s,
        remove_watermark=True,
    )

    # --- Alternative: Wan 2.1 Multi-Shot ---
    multishot_video = WanMultiShotTextToVideoPro(
        prompt=f"Professional brand video for {brand_name.output}, "
        f"{brand_story.output}, cinematic quality, multiple scenes, "
        f"{video_tone.output}, corporate advertisement, smooth transitions",
        aspect_ratio=WanMultiShotTextToVideoPro.AspectRatio.V16_9,
        resolution=WanMultiShotTextToVideoPro.Resolution.R1080P,
        duration=WanMultiShotTextToVideoPro.Duration.D10,
        remove_watermark=True,
    )

    # --- Reference-Based Video with Veo 3.1 ---
    reference_video = Veo31ReferenceToVideo(
        prompt=f"Brand narrative video inspired by storyboard frames, "
        f"{brand_story.output}, {video_tone.output}, cinematic motion, "
        "professional corporate production",
        image1=frame_1.output,
        image2=frame_3.output,
        image3=frame_4.output,
        model=Veo31ReferenceToVideo.Model.VEO3_FAST,
        aspect_ratio=Veo31ReferenceToVideo.AspectRatio.RATIO_16_9,
    )

    # --- Text-to-Video Alternative ---
    pure_text_video = Sora2ProTextToVideo(
        prompt=f"Cinematic brand manifesto video for {brand_name.output}, "
        f"{brand_story.output}, {visual_identity.output}, "
        "epic corporate video, inspirational imagery, professional production, "
        f"{video_tone.output}, dawn to success journey narrative",
        aspect_ratio=Sora2ProTextToVideo.AspectRatio.LANDSCAPE,
        n_frames=Sora2ProTextToVideo.Sora2Frames._15s,
        remove_watermark=True,
    )

    # --- Video Upscaling ---
    upscaled_storyboard = TopazVideoUpscale(
        video=storyboard_video.output,
        resolution=TopazVideoUpscale.Resolution.R4K,
        denoise=True,
    )

    # --- Audio Production ---
    # Background music
    brand_music = Suno(
        prompt=f"Corporate brand music for {brand_name.output}, {video_tone.output}, "
        "inspirational orchestral, modern corporate, emotional build, "
        "professional advertising music",
        style=Suno.Style.AMBIENT,
        instrumental=True,
        duration=60,
        model=Suno.Model.V4_5_PLUS,
    )

    # Epic version for dramatic videos
    epic_music = Suno(
        prompt=f"Epic corporate anthem for {brand_name.output}, {video_tone.output}, "
        "cinematic, building crescendo, powerful emotional impact",
        style=Suno.Style.CLASSICAL,
        instrumental=True,
        duration=60,
        model=Suno.Model.V4_5_PLUS,
    )

    # --- Combine Video and Audio ---
    final_with_music = AddAudio(
        video=upscaled_storyboard.output,
        audio=brand_music.output,
        volume=0.9,
        mix=False,
    )

    final_with_epic = AddAudio(
        video=storyboard_video.output,
        audio=epic_music.output,
        volume=0.9,
        mix=False,
    )

    # --- Create Transition Demo ---
    transition_demo = Transition(
        video_a=storyboard_video.output,
        video_b=multishot_video.output,
        transition_type=Transition.TransitionType.fade,
        duration=1.0,
    )

    # --- Outputs ---
    main_video = Output(
        name="brand_video_final",
        value=final_with_music.output,
        description="Final brand video with music (4K, Sora Storyboard)",
    )

    music_video = Output(
        name="brand_video_epic",
        value=final_with_epic.output,
        description="Brand video with epic music",
    )

    storyboard_out = Output(
        name="storyboard_video",
        value=storyboard_video.output,
        description="Multi-shot storyboard video (Sora 2 Pro)",
    )

    multishot_out = Output(
        name="multishot_video",
        value=multishot_video.output,
        description="Multi-shot video (Wan 2.1 Pro)",
    )

    reference_out = Output(
        name="reference_video",
        value=reference_video.output,
        description="Reference-based video (Veo 3.1)",
    )

    text_video_out = Output(
        name="text_to_video",
        value=pure_text_video.output,
        description="Pure text-to-video output (Sora 2 Pro)",
    )

    # Storyboard frames
    frame_1_out = Output(
        name="storyboard_frame_1",
        value=frame_1.output,
        description="Opening frame (Seedream 4.5)",
    )

    frame_2_out = Output(
        name="storyboard_frame_2",
        value=frame_2.output,
        description="Challenge frame (Imagen 4)",
    )

    frame_3_out = Output(
        name="storyboard_frame_3",
        value=frame_3.output,
        description="Solution frame (Nano Banana Pro)",
    )

    frame_4_out = Output(
        name="storyboard_frame_4",
        value=frame_4.output,
        description="Resolution frame (Seedream 4.5)",
    )

    music_out = Output(
        name="brand_music",
        value=brand_music.output,
        description="Original brand music track",
    )

    epic_out = Output(
        name="epic_music",
        value=epic_music.output,
        description="Epic brand music track",
    )

    return create_graph(
        main_video,
        music_video,
        storyboard_out,
        multishot_out,
        reference_out,
        text_video_out,
        frame_1_out,
        frame_2_out,
        frame_3_out,
        frame_4_out,
        music_out,
        epic_out,
    )


# Build the graph
graph = build_brand_video_production()


if __name__ == "__main__":
    """
    To run this example:

    1. Ensure you have KIE_API_KEY configured
    2. Run:

        python examples/kie_brand_video_production.py

    The workflow creates professional brand videos with storyboard-to-video.
    """

    print("Kie Brand Video Production Pipeline")
    print("=" * 50)
    print(f"Nodes: {len(graph.nodes)}")
    print(f"Edges: {len(graph.edges)}")
    print()
    print("Production Workflow:")
    print()
    print("  1. Storyboard Generation:")
    print("     - Seedream 4.5 - Opening/Resolution frames")
    print("     - Imagen 4 - Challenge frame")
    print("     - Nano Banana Pro - Solution frame")
    print()
    print("  2. Video Generation:")
    print("     - Sora 2 Pro Storyboard - Multi-frame to video")
    print("     - Wan 2.1 Multi-Shot - Scene transitions")
    print("     - Veo 3.1 Reference - Material-based video")
    print("     - Sora 2 Pro Text-to-Video - Pure generation")
    print()
    print("  3. Enhancement:")
    print("     - Topaz Video Upscale - 4K quality")
    print()
    print("  4. Audio Production:")
    print("     - Suno - Original brand music (ambient + epic)")
    print()
    print("Use Cases:")
    print("  - Brand manifesto videos")
    print("  - Corporate presentations")
    print("  - Product launch videos")
    print("  - About us content")
    print("  - Social media brand content")
    print()

    # Uncomment to run:
    # result = run_graph(graph)
    # print(result)
