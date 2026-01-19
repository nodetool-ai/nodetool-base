"""
Example: Kie Photo to Cinematic Video Pipeline

This workflow transforms static photos into cinematic video sequences using
Kie.ai's advanced video generation models. Similar to tools like Krea and Flora.

1. Input a photo (product, portrait, landscape)
2. Apply AI enhancement and upscaling
3. Generate cinematic video motion using multiple Kie models
4. Compare different video generation engines
5. Output high-quality cinematic clips

The workflow pattern:
    [ImageInput] -> [TopazUpscale] -> [Veo31ImageToVideo]
                                   -> [HailuoImageToVideo]
                                   -> [Kling25TurboImageToVideo]
                                       -> [TopazVideoUpscale] -> [Outputs]

Perfect for photographers, filmmakers, and content creators.

Note: If imports fail, run 'nodetool package scan && nodetool codegen' to regenerate DSL.
"""

from nodetool.dsl.graph import create_graph
from nodetool.dsl.nodetool.input import StringInput, ImageInput
from nodetool.dsl.nodetool.output import Output
from nodetool.dsl.kie.image import (
    TopazImageUpscale,
    RecraftCrispUpscale,
    IdeogramV3Reframe,
)
from nodetool.dsl.kie.video import (
    Veo31ImageToVideo,
    HailuoImageToVideoPro,
    Kling25TurboImageToVideo,
    SeedanceV1ProImageToVideo,
    TopazVideoUpscale,
    Wan26ImageToVideo,
)
from nodetool.metadata.types import ImageRef


def build_photo_to_cinematic_video():
    """
    Transform photos into cinematic video sequences.

    This function builds a workflow graph that:
    1. Accepts a static photo
    2. Enhances and upscales the image
    3. Generates cinematic motion with multiple AI models
    4. Upscales final videos for high resolution
    5. Provides multiple output options for comparison

    Returns:
        Graph: A graph object representing the workflow
    """
    # --- Inputs ---
    source_image = ImageInput(
        name="source_image",
        description="The photo to transform into video",
        value=ImageRef(type="image", uri=""),
    )

    motion_prompt = StringInput(
        name="motion_prompt",
        description="Description of the desired motion/animation",
        value="Gentle camera push-in with subtle parallax, soft ambient movement, "
        "cinematic depth of field, natural lighting transitions",
    )

    style_prompt = StringInput(
        name="style_prompt",
        description="Cinematic style direction",
        value="Cinematic, filmic quality, professional color grading, smooth motion, "
        "high production value, 24fps film look",
    )

    video_duration = StringInput(
        name="video_duration",
        description="Desired video length",
        value="5",  # seconds
    )

    # --- Enhance Source Image ---
    upscaled_source = TopazImageUpscale(
        image=source_image.output,
        upscale_factor=TopazImageUpscale.UpscaleFactor.X2,
    )

    # --- Alternative: Crisp upscale for sharper details ---
    crisp_upscaled = RecraftCrispUpscale(
        image=source_image.output,
    )

    # --- Reframe for different aspect ratios ---
    reframed_landscape = IdeogramV3Reframe(
        image=source_image.output,
        image_size=IdeogramV3Reframe.ImageSize.LANDSCAPE_16_9,
        rendering_speed=IdeogramV3Reframe.RenderingSpeed.QUALITY,
        style=IdeogramV3Reframe.Style.REALISTIC,
    )

    reframed_portrait = IdeogramV3Reframe(
        image=source_image.output,
        image_size=IdeogramV3Reframe.ImageSize.PORTRAIT_16_9,
        rendering_speed=IdeogramV3Reframe.RenderingSpeed.QUALITY,
        style=IdeogramV3Reframe.Style.REALISTIC,
    )

    # --- Generate Videos with Multiple Kie Models ---

    # Option 1: Google Veo 3.1 - Highest quality
    veo_video = Veo31ImageToVideo(
        image1=upscaled_source.output,
        prompt=f"{motion_prompt.output}. {style_prompt.output}. Duration: {video_duration.output} seconds.",
        model=Veo31ImageToVideo.Model.VEO3,
        aspect_ratio=Veo31ImageToVideo.AspectRatio.RATIO_16_9,
    )

    # Option 2: Hailuo Pro - Excellent motion
    hailuo_video = HailuoImageToVideoPro(
        image=upscaled_source.output,
        prompt=f"{motion_prompt.output}. {style_prompt.output}. Duration: {video_duration.output} seconds.",
        duration=HailuoImageToVideoPro.Duration.D6,
        resolution=HailuoImageToVideoPro.Resolution.R1080P,
    )

    # Option 3: Kling 2.5 Turbo - Fast and high quality
    kling_video = Kling25TurboImageToVideo(
        image=upscaled_source.output,
        prompt=f"{motion_prompt.output}. {style_prompt.output}. Duration: {video_duration.output} seconds.",
        duration=Kling25TurboImageToVideo.Duration.D5,
        cfg_scale=0.5,
    )

    # Option 4: Seedance Pro - Bytedance quality
    seedance_video = SeedanceV1ProImageToVideo(
        image1=upscaled_source.output,
        prompt=f"{motion_prompt.output}. {style_prompt.output}. Duration: {video_duration.output} seconds.",
        duration=SeedanceV1ProImageToVideo.Duration.D5,
        resolution=SeedanceV1ProImageToVideo.Resolution.R720P,
        remove_watermark=True,
    )

    # Option 5: Wan 2.6 - Alibaba quality
    wan_video = Wan26ImageToVideo(
        image1=upscaled_source.output,
        prompt=f"{motion_prompt.output}. {style_prompt.output}. Duration: {video_duration.output} seconds.",
        duration=Wan26ImageToVideo.Duration.D5,
        resolution=Wan26ImageToVideo.Resolution.R1080P,
    )

    # --- Upscale Videos for 4K Output ---
    veo_4k = TopazVideoUpscale(
        video=veo_video.output,
        resolution=TopazVideoUpscale.Resolution.R4K,
        denoise=True,
    )

    hailuo_4k = TopazVideoUpscale(
        video=hailuo_video.output,
        resolution=TopazVideoUpscale.Resolution.R4K,
        denoise=True,
    )

    # --- Outputs ---
    veo_output = Output(
        name="veo_cinematic",
        value=veo_4k.output,
        description="Google Veo 3.1 cinematic video (4K upscaled)",
    )

    hailuo_output = Output(
        name="hailuo_cinematic",
        value=hailuo_4k.output,
        description="Hailuo Pro cinematic video (4K upscaled)",
    )

    kling_output = Output(
        name="kling_cinematic",
        value=kling_video.output,
        description="Kling 2.5 Turbo cinematic video",
    )

    seedance_output = Output(
        name="seedance_cinematic",
        value=seedance_video.output,
        description="Seedance Pro cinematic video",
    )

    wan_output = Output(
        name="wan_cinematic",
        value=wan_video.output,
        description="Wan 2.6 cinematic video",
    )

    enhanced_source = Output(
        name="enhanced_image",
        value=upscaled_source.output,
        description="Enhanced source image (2x upscaled)",
    )

    crisp_source = Output(
        name="crisp_upscaled_image",
        value=crisp_upscaled.output,
        description="Alternative enhancement (Recraft Crisp Upscale)",
    )

    landscape_frame = Output(
        name="landscape_reframe",
        value=reframed_landscape.output,
        description="Reframed to 16:9 landscape",
    )

    portrait_frame = Output(
        name="portrait_reframe",
        value=reframed_portrait.output,
        description="Reframed to 9:16 portrait",
    )

    return create_graph(
        veo_output,
        hailuo_output,
        kling_output,
        seedance_output,
        wan_output,
        enhanced_source,
        crisp_source,
        landscape_frame,
        portrait_frame,
    )


# Build the graph
graph = build_photo_to_cinematic_video()


if __name__ == "__main__":
    """
    To run this example:

    1. Ensure you have KIE_API_KEY configured
    2. Provide a high-quality source image
    3. Run:

        python examples/kie_photo_to_cinematic_video.py

    The workflow transforms photos into cinematic video sequences.
    """

    print("Kie Photo to Cinematic Video Pipeline")
    print("=" * 50)
    print(f"Nodes: {len(graph.nodes)}")
    print(f"Edges: {len(graph.edges)}")
    print()
    print("Kie Models Compared:")
    print("  - Google Veo 3.1 - Highest quality, natural motion")
    print("  - Hailuo 2.3 Pro - Excellent temporal consistency")
    print("  - Kling 2.5 Turbo - Fast, high-quality results")
    print("  - Seedance Pro - Bytedance's advanced model")
    print("  - Wan 2.6 - Alibaba's video generation")
    print("  - Topaz Video Upscale - 4K enhancement")
    print()
    print("Workflow pattern:")
    print("  [Source Photo]")
    print("      -> [TopazImageUpscale] (enhance)")
    print("          -> [Veo31ImageToVideo] (Google)")
    print("          -> [HailuoImageToVideoPro] (MiniMax)")
    print("          -> [Kling25TurboImageToVideo] (Kuaishou)")
    print("          -> [SeedanceV1ProImageToVideo] (Bytedance)")
    print("          -> [Wan26ImageToVideo] (Alibaba)")
    print("              -> [TopazVideoUpscale] (4K)")
    print("                  -> [Outputs]")
    print()
    print("Use Cases:")
    print("  - Portfolio presentations")
    print("  - Real estate virtual tours")
    print("  - Product showcases")
    print("  - Social media content")
    print("  - Documentary b-roll")
    print()

    # Uncomment to run:
    # result = run_graph(graph)
    # print(result)
