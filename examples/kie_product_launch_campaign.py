"""
Example: Kie Product Launch Campaign Generator

This workflow creates a complete visual campaign for product launches using
Kie.ai's image and video generation capabilities. Generates hero images,
product videos, social media assets, and promotional content.

1. Generate product hero images in multiple styles
2. Create promotional video clips
3. Generate social media assets for different platforms
4. Produce background music for videos

The workflow pattern:
    [ProductInfo] -> [Imagen4/Flux/Seedream] (hero images)
                        -> [KlingTextToVideo] (promo video)
                            -> [Suno] (background music)
                                -> [AddAudio] -> [Outputs]

Perfect for marketing teams, product managers, and brand agencies.

Note: If imports fail, run 'nodetool package scan && nodetool codegen' to regenerate DSL.
"""

from nodetool.dsl.graph import create_graph, run_graph
from nodetool.dsl.nodetool.input import StringInput, ImageInput
from nodetool.dsl.nodetool.output import Output
from nodetool.dsl.nodetool.text import FormatText
from nodetool.dsl.kie.image import (
    Imagen4Ultra,
    Flux2ProTextToImage,
    Seedream45TextToImage,
    ZImage,
    FluxKontext,
    TopazImageUpscale,
    RecraftRemoveBackground,
    QwenTextToImage,
    GrokImagineTextToImage,
)
from nodetool.dsl.kie.video import (
    KlingTextToVideo,
    Veo31TextToVideo,
    HailuoTextToVideoPro,
    Sora2ProTextToVideo,
)
from nodetool.dsl.kie.audio import Suno
from nodetool.dsl.nodetool.video import AddAudio
from nodetool.metadata.types import ImageRef


def build_product_launch_campaign():
    """
    Generate a complete product launch visual campaign.

    This function builds a workflow graph that:
    1. Creates hero images using multiple AI models
    2. Generates promotional video content
    3. Produces platform-specific social media assets
    4. Creates voice-over announcements
    5. Generates background music for videos

    Returns:
        Graph: A graph object representing the workflow
    """
    # --- Inputs ---
    product_name = StringInput(
        name="product_name",
        description="Name of the product being launched",
        value="NovaSphere Pro",
    )

    product_description = StringInput(
        name="product_description",
        description="Detailed product description",
        value="Revolutionary AI-powered smart home hub with voice control, "
        "holographic display, and seamless integration with all your devices. "
        "Sleek spherical design in matte white with subtle RGB lighting.",
    )

    brand_values = StringInput(
        name="brand_values",
        description="Brand values and aesthetic",
        value="Innovative, minimalist, premium, futuristic, approachable technology",
    )

    target_audience = StringInput(
        name="target_audience",
        description="Target customer profile",
        value="Tech-savvy homeowners, early adopters, design-conscious consumers, ages 25-45",
    )

    announcement_text = StringInput(
        name="announcement_text",
        description="Voice-over announcement script",
        value="Introducing NovaSphere Pro. The future of smart home living is here. "
        "Experience seamless control, stunning design, and intelligent automation. "
        "Available now. Transform your home today.",
    )

    # --- Hero Image Generation ---
    # Using multiple models for variety and comparison

    # Premium hero - Imagen 4 Ultra (Google's best)
    hero_imagen = Imagen4Ultra(
        prompt=f"Professional product photography of {product_name.output}, "
        f"{product_description.output}, studio lighting, clean white background, "
        "dramatic shadows, 8K quality, advertising campaign hero shot, "
        "ultra high resolution, photorealistic",
        aspect_ratio=Imagen4Ultra.AspectRatio.SQUARE,
    )

    # Lifestyle context - Flux 2 Pro
    hero_lifestyle = Flux2ProTextToImage(
        prompt=f"{product_name.output} in a modern living room setting, "
        f"{product_description.output}, natural lighting, interior design context, "
        "lifestyle product photography, cozy home environment, premium aesthetic",
        aspect_ratio=Flux2ProTextToImage.AspectRatio.LANDSCAPE,
        resolution=Flux2ProTextToImage.Resolution.RES_2K,
        steps=30,
    )

    # Dramatic product shot - Seedream 4.5
    hero_dramatic = Seedream45TextToImage(
        prompt=f"Cinematic product reveal of {product_name.output}, "
        f"{product_description.output}, dramatic lighting, dark background, "
        "spotlight effect, luxury tech aesthetic, high contrast, premium feel",
        aspect_ratio=Seedream45TextToImage.AspectRatio.LANDSCAPE,
        quality=Seedream45TextToImage.Quality.HIGH,
    )

    # Fast iteration - Z-Image
    hero_fast = ZImage(
        prompt=f"{product_name.output} product shot, {product_description.output}, "
        "clean design, professional photography, tech product showcase",
        aspect_ratio=ZImage.AspectRatio.SQUARE,
    )

    # Artistic interpretation - Flux Kontext
    hero_artistic = FluxKontext(
        prompt=f"Artistic product visualization of {product_name.output}, "
        f"{product_description.output}, abstract tech background, "
        "futuristic environment, creative product photography, award-winning design",
        aspect_ratio=FluxKontext.AspectRatio.LANDSCAPE,
        mode=FluxKontext.Mode.MAX,
    )

    # Social media vertical - Qwen
    hero_vertical = QwenTextToImage(
        prompt=f"{product_name.output} vertical product shot for social media, "
        f"{product_description.output}, Instagram-ready, bold composition",
        aspect_ratio=QwenTextToImage.AspectRatio.PORTRAIT,
    )

    # Alternative style - Grok Imagine
    hero_grok = GrokImagineTextToImage(
        prompt=f"{product_name.output} creative product visualization, "
        f"{product_description.output}, innovative composition, striking visuals",
        aspect_ratio=GrokImagineTextToImage.AspectRatio.SQUARE,
    )

    # --- Upscale Hero Images ---
    hero_upscaled = TopazImageUpscale(
        image=hero_imagen.output,
        upscale_factor=TopazImageUpscale.UpscaleFactor.X4,
    )

    # --- Remove Background for Composite Work ---
    hero_isolated = RecraftRemoveBackground(
        image=hero_imagen.output,
    )

    # --- Promotional Videos ---
    # Main campaign video - Veo 3.1
    promo_video_veo = Veo31TextToVideo(
        prompt=f"Cinematic product reveal video for {product_name.output}, "
        f"{product_description.output}, smooth camera movement, "
        "dramatic lighting transitions, premium tech commercial, "
        "high production value, professional advertising quality",
        model=Veo31TextToVideo.Model.VEO3,
        aspect_ratio=Veo31TextToVideo.AspectRatio.RATIO_16_9,
    )

    # Fast social video - Kling
    promo_video_kling = KlingTextToVideo(
        prompt=f"Dynamic product showcase of {product_name.output}, "
        f"{product_description.output}, energetic motion, modern tech aesthetic, "
        "social media advertisement, eye-catching visuals",
        aspect_ratio=KlingTextToVideo.AspectRatio.V16_9,
        duration=5,
        resolution=KlingTextToVideo.Resolution.R768P,
    )

    # Premium quality - Hailuo Pro
    promo_video_hailuo = HailuoTextToVideoPro(
        prompt=f"Premium product commercial for {product_name.output}, "
        f"{product_description.output}, luxurious presentation, "
        "smooth motion, cinematic quality, high-end advertising",
        duration=HailuoTextToVideoPro.Duration.D6,
        resolution=HailuoTextToVideoPro.Resolution.R1080P,
    )

    # Story format - Sora 2 Pro
    promo_video_sora = Sora2ProTextToVideo(
        prompt=f"Instagram Stories style product teaser for {product_name.output}, "
        f"{product_description.output}, vertical format, trendy transitions, "
        "social media optimized, engaging hook",
        aspect_ratio=Sora2ProTextToVideo.AspectRatio.PORTRAIT,
        n_frames=Sora2ProTextToVideo.Sora2Frames._10s,
        remove_watermark=True,
    )

    # --- Audio Assets ---
    # Background music
    campaign_music = Suno(
        prompt=f"Modern tech product launch music, {brand_values.output}, "
        "inspiring, innovative, futuristic corporate background music",
        style=Suno.Style.ELECTRONIC,
        instrumental=True,
        duration=60,
        model=Suno.Model.V4_5_PLUS,
    )

    # Energetic promo music
    promo_music = Suno(
        prompt=f"Upbeat product announcement jingle, {brand_values.output}, "
        "catchy, modern, corporate advertising music",
        style=Suno.Style.POP,
        instrumental=True,
        duration=30,
        model=Suno.Model.V4_5_PLUS,
    )

    # --- Combine Video and Audio ---
    final_promo_video = AddAudio(
        video=promo_video_veo.output,
        audio=promo_music.output,
        volume=0.9,
        mix=False,
    )

    promo_with_music = AddAudio(
        video=promo_video_hailuo.output,
        audio=campaign_music.output,
        volume=0.8,
        mix=False,
    )

    # --- Outputs ---
    # Hero Images
    hero_main = Output(
        name="hero_image_main",
        value=hero_upscaled.output,
        description="Main hero image (Imagen 4 Ultra, 4x upscaled)",
    )

    hero_lifestyle_out = Output(
        name="hero_lifestyle",
        value=hero_lifestyle.output,
        description="Lifestyle context hero (Flux 2 Pro)",
    )

    hero_dramatic_out = Output(
        name="hero_dramatic",
        value=hero_dramatic.output,
        description="Dramatic product shot (Seedream 4.5)",
    )

    hero_artistic_out = Output(
        name="hero_artistic",
        value=hero_artistic.output,
        description="Artistic interpretation (Flux Kontext)",
    )

    hero_isolated_out = Output(
        name="hero_isolated",
        value=hero_isolated.output,
        description="Product with removed background",
    )

    hero_vertical_out = Output(
        name="hero_vertical",
        value=hero_vertical.output,
        description="Vertical social media hero (Qwen)",
    )

    # Videos
    video_main = Output(
        name="promo_video_main",
        value=final_promo_video.output,
        description="Main promo video with music (Veo 3.1)",
    )

    video_music = Output(
        name="promo_video_music",
        value=promo_with_music.output,
        description="Promo video with background music (Hailuo)",
    )

    video_social = Output(
        name="promo_video_social",
        value=promo_video_kling.output,
        description="Social media promo (Kling)",
    )

    video_stories = Output(
        name="promo_video_stories",
        value=promo_video_sora.output,
        description="Stories format vertical video (Sora 2)",
    )

    # Audio
    music_track = Output(
        name="background_music",
        value=campaign_music.output,
        description="Campaign background music",
    )

    promo_jingle = Output(
        name="promo_jingle",
        value=promo_music.output,
        description="Promo music jingle",
    )

    return create_graph(
        hero_main,
        hero_lifestyle_out,
        hero_dramatic_out,
        hero_artistic_out,
        hero_isolated_out,
        hero_vertical_out,
        video_main,
        video_music,
        video_social,
        video_stories,
        music_track,
        promo_jingle,
    )


# Build the graph
graph = build_product_launch_campaign()


if __name__ == "__main__":
    """
    To run this example:

    1. Ensure you have KIE_API_KEY configured
    2. Run:

        python examples/kie_product_launch_campaign.py

    The workflow generates a complete product launch visual campaign.
    """

    print("Kie Product Launch Campaign Generator")
    print("=" * 50)
    print(f"Nodes: {len(graph.nodes)}")
    print(f"Edges: {len(graph.edges)}")
    print()
    print("Campaign Assets Generated:")
    print()
    print("  Hero Images (7 variations):")
    print("    - Imagen 4 Ultra - Premium studio shot")
    print("    - Flux 2 Pro - Lifestyle context")
    print("    - Seedream 4.5 - Dramatic lighting")
    print("    - Flux Kontext - Artistic interpretation")
    print("    - Z-Image - Fast iteration")
    print("    - Qwen - Vertical social media")
    print("    - Grok Imagine - Alternative style")
    print()
    print("  Promotional Videos (4 variations):")
    print("    - Veo 3.1 - Main campaign video")
    print("    - Kling - Fast social video")
    print("    - Hailuo Pro - Premium quality")
    print("    - Sora 2 Pro - Stories format")
    print()
    print("  Audio Assets:")
    print("    - Suno - Background music & promo jingles")
    print()
    print("  Additional Processing:")
    print("    - Topaz Upscale - 4x hero enhancement")
    print("    - Recraft - Background removal")
    print()

    # Uncomment to run:
    # result = run_graph(graph)
    # print(result)
