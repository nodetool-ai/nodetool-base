"""
Video Frame Extract and Text Overlay DSL Example

Extract frames from a video file and render customizable text captions/overlays on top
of the extracted frames using font rendering.

Workflow:
1. **Video Input** - Load the source video file
2. **Frame Iterator** - Extract individual frames from the video
3. **Generate Caption** - Use AI to generate descriptive captions for the frames
4. **Render Text** - Draw caption text onto each frame with custom font and styling
5. **Image Output** - Save the frames with text overlay
"""

import os
from nodetool.dsl.graph import create_graph, run_graph
from nodetool.workflows.processing_context import AssetOutputMode
from nodetool.dsl.nodetool.input import VideoInput, StringInput
from nodetool.dsl.nodetool.output import ImageOutput
from nodetool.dsl.nodetool.video import FrameIterator
from nodetool.dsl.lib.pillow.draw import RenderText
from nodetool.dsl.nodetool.text import FormatText
from nodetool.dsl.nodetool.agents import Agent
from nodetool.metadata.types import FontRef, VideoRef, LanguageModel, Provider, ColorRef


font_file = os.path.join(os.path.dirname(__file__), "DejaVuSans.ttf")

# Configure LLM for caption generation
LLM = LanguageModel(
    type="language_model",
    id="gpt-4o",
    provider=Provider.OpenAI
)

# --- Inputs ---
# Video input: User provides a video file
video_input = VideoInput(
    name="video",
    description="Input video to extract frames from",
    value=VideoRef(
        uri="https://upload.wikimedia.org/wikipedia/commons/c/c0/Abstracte_werkers_-_Open_Beelden_-_19089.ogv",
        type="video",
    ),
)

# Context/description: User provides context about what the video is about
context = StringInput(
    name="video_context",
    description="Description or topic of the video (used for caption generation)",
    value="A crane bird in natural habitat",
)

# Text overlay settings
overlay_text_input = StringInput(
    name="overlay_text",
    description="Custom text to overlay on frames",
    value="Frame Caption",
)

# --- Extract frames from video ---
# FrameIterator streams individual frames from the video
# It outputs: frame (ImageRef), index (int), fps (float)
frame_iterator = FrameIterator(
    video=video_input.output,
    start=0,
    end=10,  # Extract first 10 frames for demo
)

# --- Generate AI captions (optional) ---
# Format a prompt for the AI to generate captions based on video context
caption_prompt = FormatText(
    template="Generate a short, catchy one-line caption for a video frame about: {{ context }}. Make it engaging and under 30 characters.",
    context=context.output,
)

# Use an agent to generate the caption
caption_agent = Agent(
    prompt=caption_prompt.output,
    model=LLM,
    max_tokens=50
)

# --- Render text on each frame ---
# RenderText adds text overlay to each extracted frame
# This executes for each frame from the FrameIterator (automatic iteration)
rendered_frame = RenderText(
    image=frame_iterator.out.frame,  # Each frame from the iterator
    text=caption_agent.out.text,  # Generated caption as text
    font=FontRef(type="font", name="Arial.ttf"),  # Font family
    size=32,  # Font size in pixels
    color=ColorRef(value="#FFFFFF"),  # White text color
    x=10,  # X coordinate (10 pixels from left)
    y=10,  # Y coordinate (10 pixels from top)
    align=RenderText.TextAlignment.LEFT,  # Text alignment
)

# --- Output frames with text overlay ---
# Preview nodes capture results from iterating operations
output = ImageOutput(
    name="frame_with_overlay",
    description="Video frames with text overlay rendered on top",
    value=rendered_frame.output,
)

# Create the graph
graph = create_graph(output)


if __name__ == "__main__":
    result = run_graph(graph, asset_output_mode=AssetOutputMode.WORKSPACE)
    print(f"Frames with text overlay saved: {result}")
