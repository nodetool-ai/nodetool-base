"""
Image to Video Animation DSL Example

Animate a still image into a short cinematic clip using the ImageToVideo node.

Workflow:
1. **Image Input** – Provide a reference frame to drive the animation
2. **Creative Direction** – Capture a short story beat and motion cues from the user
3. **Prompt Assembly** – Blend scene description with motion guidance for the generator
4. **Image-to-Video Generation** – Animate the image with the selected video model
5. **Video Output** – Export the rendered clip to the workspace for preview or download
"""

from nodetool.dsl.graph import create_graph, run_graph
from nodetool.workflows.processing_context import AssetOutputMode
from nodetool.dsl.nodetool.input import ImageInput, StringInput
from nodetool.dsl.nodetool.output import VideoOutput
from nodetool.dsl.nodetool.text import FormatText
from nodetool.dsl.nodetool.video import ImageToVideo
from nodetool.metadata.types import ImageRef, VideoModel, Provider


# --- Reference Image ---------------------------------------------------------
reference_image = ImageInput(
    name="reference_frame",
    description="Hero frame or concept art to animate into motion",
    value=ImageRef(
        type="image",
        uri="https://upload.wikimedia.org/wikipedia/commons/8/89/Stormy_sunset_over_the_ocean.jpg",
    ),
)

# --- Creative Direction ------------------------------------------------------
scene_prompt = StringInput(
    name="scene_prompt",
    description="High-level description of the mood, setting, and visual tone",
    value="Moody twilight ocean scene with cinematic lighting",
)

motion_prompt = StringInput(
    name="motion_prompt",
    description="Movement cues that the animation should follow",
    value="Slow dolly push toward the horizon as waves shimmer with reflective highlights",
)

duration_prompt = StringInput(
    name="duration_prompt",
    description="Short descriptor for pacing and clip duration",
    value="8 second dramatic reveal",
)

# Combine the creative directives into a single generator prompt.
video_prompt = FormatText(
    template=(
        "Transform the reference frame into a cinematic ocean scene. "
        "Scene: {{ scene }}. Motion: {{ motion }}. Duration: {{ duration }}. "
        "Keep the animation cohesive, with gentle camera movement and atmospheric particles."
    ),
    scene=scene_prompt.output,
    motion=motion_prompt.output,
    duration=duration_prompt.output,
)

# Optional negative prompt to avoid unwanted artifacts (e.g., text overlays).
negative_prompt = StringInput(
    name="negative_prompt",
    description="Elements to avoid in the generated video",
    value="no watermarks, no text overlays, maintain realistic wave motion",
)

# --- Image to Video Generation -----------------------------------------------
video_model = VideoModel(
    type="video_model",
    provider=Provider.Gemini,
    id="veo-3.0-fast-generate-001",
    name="Veo 3.0 Fast",
)

animated_video = ImageToVideo(
    image=reference_image.output,
    model=video_model,
    prompt=video_prompt.output,
    negative_prompt=negative_prompt.output,
    aspect_ratio=ImageToVideo.AspectRatio.RATIO_16_9,
    resolution=ImageToVideo.Resolution.FHD,
    num_frames=120,
    guidance_scale=8.0,
    num_inference_steps=28,
)

# --- Output ------------------------------------------------------------------
video_output = VideoOutput(
    name="animated_clip",
    description="AI-generated cinematic video based on the supplied reference frame",
    value=animated_video.output,
)

# Build the graph so the workflow can be executed or exported.
graph = create_graph(video_output)


if __name__ == "__main__":
    result = run_graph(graph, asset_output_mode=AssetOutputMode.WORKSPACE)
    print(f"Animated video saved to workspace: {result}")
