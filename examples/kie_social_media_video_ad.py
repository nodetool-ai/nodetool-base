"""
Example: Kie Social Media Video Ad Generator

This workflow creates engaging social media video ads using Kie.ai's powerful
AI video generation capabilities. It demonstrates:

1. Generate concept visuals with text-to-image
2. Transform images into dynamic video clips
3. Add background music with AI-generated tracks
4. Combine audio and video for final ad

The workflow pattern:
    [StringInputs] -> [Imagen4/Flux] (hero image) -> [KlingImageToVideo] (animate)
                                                        -> [Suno] (background music)
                                                            -> [AddAudio] -> [Output]

Perfect for marketing teams and content creators generating social media ads.

Note: If imports fail, run 'nodetool package scan && nodetool codegen' to regenerate DSL.
"""

from nodetool.dsl.graph import create_graph, run_graph
from nodetool.dsl.nodetool.input import StringInput
from nodetool.dsl.nodetool.output import Output
from nodetool.dsl.kie.image import (
    Imagen4Fast,
    FluxKontext,
    TopazImageUpscale,
)
from nodetool.dsl.kie.video import (
    KlingImageToVideo,
    Kling25TurboTextToVideo,
    HailuoImageToVideoPro,
)
from nodetool.dsl.kie.audio import Suno
from nodetool.dsl.nodetool.video import AddAudio


def build_social_media_video_ad():
    """
    Generate a social media video ad with Kie AI models.

    This function builds a workflow graph that:
    1. Accepts product/service description and ad copy
    2. Generates a hero image using Imagen 4 Fast
    3. Transforms the image into a video clip with Kling 2.6
    4. Creates voice-over narration with ElevenLabs
    5. Combines everything into a final video ad

    Returns:
        Graph: A graph object representing the workflow
    """
    # --- Inputs ---
    product_name = StringInput(
        name="product_name",
        description="Name of the product or service",
        value="AuraFlow Pro",
    )

    product_description = StringInput(
        name="product_description",
        description="Brief description of the product",
        value="A revolutionary smart water bottle that tracks hydration and syncs with your fitness apps",
    )

    ad_style = StringInput(
        name="ad_style",
        description="Visual style for the ad",
        value="Modern, minimalist, tech-forward, with soft gradient lighting and floating water droplets",
    )

    target_platform = StringInput(
        name="target_platform",
        description="Target social media platform (affects aspect ratio)",
        value="Instagram Reels",  # 9:16 for vertical video
    )

    voice_over_text = StringInput(
        name="voice_over_text",
        description="The narration text for the ad",
        value="Stay hydrated, stay connected. AuraFlow Pro - your smart companion for peak performance. Available now.",
    )

    # --- Generate Hero Image with Imagen 4 Fast ---
    hero_image = Imagen4Fast(
        prompt=f"Professional product photography of {product_name.output}, "
        f"{product_description.output}, {ad_style.output}, "
        "clean background, dramatic lighting, 8K quality, advertising photography",
        aspect_ratio=Imagen4Fast.AspectRatio.PORTRAIT,  # 9:16 for social
    )

    # --- Upscale for better quality ---
    upscaled_hero = TopazImageUpscale(
        image=hero_image.output,
        upscale_factor=TopazImageUpscale.UpscaleFactor.X2,
    )

    # --- Animate Image to Video with Kling 2.6 ---
    animated_video = KlingImageToVideo(
        prompt=f"Cinematic product reveal, {product_name.output} floating with "
        "subtle rotation, water droplets gently falling, soft bokeh background, "
        "smooth motion, professional commercial quality",
        image1=upscaled_hero.output,
        duration=5,
        sound=False,  # We'll add our own audio
    )

    # --- Generate Background Music with Suno ---
    background_music = Suno(
        prompt="Modern electronic advertising music, upbeat, professional, "
        "product showcase energy, corporate tech vibes",
        style=Suno.Style.ELECTRONIC,
        instrumental=True,
        duration=30,
        model=Suno.Model.V4_5_PLUS,
    )

    # --- Combine Video and Audio ---
    final_ad = AddAudio(
        video=animated_video.output,
        audio=background_music.output,
        volume=0.8,
        mix=False,
    )

    # --- Alternative: Text-to-Video Only ---
    # For comparison, also generate a direct text-to-video version
    text_to_video_ad = Kling25TurboTextToVideo(
        prompt=f"Professional advertisement for {product_name.output}, "
        f"{product_description.output}, {ad_style.output}, "
        "cinematic product showcase, smooth camera movement, "
        "professional lighting, high-end commercial quality",
        duration=Kling25TurboTextToVideo.Duration.D5,
        aspect_ratio=Kling25TurboTextToVideo.AspectRatio.V9_16,
        cfg_scale=0.6,
    )

    # --- Outputs ---
    main_ad = Output(
        name="video_ad_with_music",
        value=final_ad.output,
        description="Final video ad with background music",
    )

    hero_output = Output(
        name="hero_image",
        value=upscaled_hero.output,
        description="High-quality hero image for static ads",
    )

    alternative_ad = Output(
        name="text_to_video_ad",
        value=text_to_video_ad.output,
        description="Alternative ad generated directly from text",
    )

    audio_output = Output(
        name="background_music",
        value=background_music.output,
        description="Background music track for reuse",
    )

    return create_graph(main_ad, hero_output, alternative_ad, audio_output)


# Build the graph
graph = build_social_media_video_ad()


if __name__ == "__main__":
    """
    To run this example:

    1. Ensure you have KIE_API_KEY configured
    2. Run:

        python examples/kie_social_media_video_ad.py

    The workflow generates social media video ads using Kie.ai models.
    """

    print("Kie Social Media Video Ad Generator")
    print("=" * 50)
    print(f"Nodes: {len(graph.nodes)}")
    print(f"Edges: {len(graph.edges)}")
    print()
    print("Kie Models Used:")
    print("  - Imagen 4 Fast (Google) - Hero image generation")
    print("  - Topaz Image Upscale - Image enhancement")
    print("  - Kling 2.6 Image-to-Video - Animation")
    print("  - Kling 2.5 Turbo Text-to-Video - Alternative")
    print("  - Suno - Background music generation")
    print()
    print("Workflow pattern:")
    print("  [Product Description]")
    print("      -> [Imagen4Fast] (hero image)")
    print("          -> [TopazImageUpscale] (enhance)")
    print("              -> [KlingImageToVideo] (animate)")
    print("                  -> [Suno] (background music)")
    print("                      -> [AddAudio] (combine)")
    print("                          -> [Output]")
    print()

    # Uncomment to run:
    # result = run_graph(graph)
    # print(result)
