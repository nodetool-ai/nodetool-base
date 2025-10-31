"""
AI Product Launch Video Generator DSL Example

Create a realistic 16:9 marketing video for a new product launch using the TextToVideo node.

Workflow:
1. **Collect Campaign Inputs** - Marketing brief, audience, and tone
2. **Craft Video Prompt** - Assemble a cinematic prompt from structured inputs
3. **Generate Video** - Produce a high-definition clip with AI text-to-video
4. **Output Asset** - Save the generated video for review and editing
"""

from nodetool.dsl.graph import create_graph, run_graph
from nodetool.dsl.nodetool.input import StringInput
from nodetool.dsl.nodetool.text import FormatText
from nodetool.dsl.nodetool.video import TextToVideo
from nodetool.dsl.nodetool.output import VideoOutput
from nodetool.metadata.types import Provider, VideoModel
from nodetool.workflows.processing_context import AssetOutputMode


campaign_brief = StringInput(
    name="campaign_brief",
    description="Short description of the campaign concept",
    value="Launch video for the Aurora Trail smart fitness watch, highlighting outdoor adventure tracking.",
)

target_audience = StringInput(
    name="target_audience",
    description="Primary audience for the marketing video",
    value="active millennials who enjoy weekend hiking and fitness challenges",
)

tone = StringInput(
    name="tone",
    description="Desired tone or mood for the video",
    value="inspiring, cinematic, and energetic",
)

key_features = StringInput(
    name="key_features",
    description="Key product features to highlight",
    value="GPS navigation, heart-rate analytics, adaptive coaching, water resistance",
)

prompt_builder = FormatText(
    template="""
Create a cinematic product reveal sequence for a wearable tech launch.
Product narrative: {{ brief }}
Audience: {{ audience }}
Emphasize features: {{ features }}
Tone: {{ tone }}
Visual style: golden-hour mountain trails, close-up wrist shots, dynamic motion transitions, realistic lighting, natural colors.
""".strip(),
    brief=campaign_brief.output,
    audience=target_audience.output,
    features=key_features.output,
    tone=tone.output,
)

negative_prompt = FormatText(
    template="""
Avoid glitch art, distorted anatomy, floating objects, text overlays, watermarks, or unrealistic facial expressions.
""".strip(),
)

video_generator = TextToVideo(
    model=VideoModel(
        type="video_model",
        provider=Provider.Gemini,
        id="veo-3.0-fast-generate-001",
        name="Veo 3.0 Fast",
    ),
    prompt=prompt_builder.output,
    negative_prompt=negative_prompt.output,
    aspect_ratio=TextToVideo.AspectRatio.RATIO_16_9,
    resolution=TextToVideo.Resolution.FHD,
    num_frames=120,
    guidance_scale=9.0,
    num_inference_steps=40,
    seed=12345,
)

video_output = VideoOutput(
    name="product_launch_video",
    description="Generated marketing video for the Aurora Trail watch",
    value=video_generator.output,
)

# Create the graph
graph = create_graph(video_output)


if __name__ == "__main__":
    result = run_graph(graph, asset_output_mode=AssetOutputMode.WORKSPACE)
    print(f"Generated video asset: {result['product_launch_video']}")
