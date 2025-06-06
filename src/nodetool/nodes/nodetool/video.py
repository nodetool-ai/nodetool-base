from datetime import datetime
import enum
import os
import tempfile
import uuid
import ffmpeg
import cv2
import logging

import PIL.ImageFilter
import PIL.ImageOps
import PIL.Image
import PIL.ImageFont
import PIL.ImageEnhance
import PIL.ImageDraw
import numpy as np
from pydantic import Field
from nodetool.metadata.types import AudioChunk, AudioRef, ColorRef, FolderRef
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.metadata.types import ImageRef, Event
from nodetool.workflows.base_node import BaseNode
from nodetool.metadata.types import VideoRef, FontRef

logger = logging.getLogger(__name__)


def safe_unlink(path: str):
    try:
        safe_unlink(path)
    except Exception:
        pass


class LoadVideoAssets(BaseNode):
    """Load video files from an asset folder.

    video, assets, load

    Use cases:
    - Provide videos for batch processing
    - Iterate over stored video assets
    - Prepare clips for editing or analysis
    """

    folder: FolderRef = Field(
        default=FolderRef(),
        description="The asset folder to load the video files from.",
    )

    @classmethod
    def get_title(cls):
        return "Load Video Folder"

    def required_inputs(self):
        return ["folder"]

    @classmethod
    def return_type(cls):
        return {
            "video": VideoRef,
            "name": str,
        }

    async def gen_process(self, context: ProcessingContext):
        if self.folder.is_empty():
            raise ValueError("Please select an asset folder.")

        parent_id = self.folder.asset_id
        list_assets = await context.list_assets(
            parent_id=parent_id, content_type="video"
        )
        for asset in list_assets.assets:
            yield "name", asset.name
            yield "video", VideoRef(
                type="video",
                uri=await context.get_asset_url(asset.id),
                asset_id=asset.id,
            )


class SaveVideo(BaseNode):
    """
    Save a video to an asset folder.
    video, save, file, output

    Use cases:
    1. Export processed video to a specific asset folder
    2. Save video with a custom name
    3. Create a copy of a video in a different location
    """

    video: VideoRef = Field(default=VideoRef(), description="The video to save.")
    folder: FolderRef = Field(
        default=FolderRef(), description="The asset folder to save the video in."
    )
    name: str = Field(
        default="%Y-%m-%d-%H-%M-%S.mp4",
        description="""
        Name of the output video.
        You can use time and date variables to create unique names:
        %Y - Year
        %m - Month
        %d - Day
        %H - Hour
        %M - Minute
        %S - Second
        """,
    )

    @classmethod
    def get_title(cls):
        return "Save Video Asset"

    def required_inputs(self):
        return ["video"]

    async def process(self, context: ProcessingContext) -> VideoRef:
        video = await context.asset_to_io(self.video)
        filename = datetime.now().strftime(self.name)
        return await context.video_from_io(
            buffer=video,
            name=filename,
            parent_id=self.folder.asset_id if self.folder.is_set() else None,
        )


class FrameIterator(BaseNode):
    """
    Extract frames from a video file using OpenCV.
    video, frames, extract, sequence

    Use cases:
    1. Generate image sequences for further processing
    2. Extract specific frame ranges from a video
    3. Create thumbnails or previews from video content
    """

    video: VideoRef = Field(
        default=VideoRef(), description="The input video to extract frames from."
    )
    start: int = Field(default=0, description="The frame to start extracting from.")
    end: int = Field(default=-1, description="The frame to stop extracting from.")

    @classmethod
    def get_title(cls):
        return "Frame Iterator"

    @classmethod
    def return_type(cls):
        return {
            "frame": ImageRef,
            "index": int,
            "fps": float,
            "event": Event,
        }

    async def gen_process(self, context: ProcessingContext):
        video_file = await context.asset_to_io(self.video)

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as temp:
            temp.write(video_file.read())
            temp.flush()

            cap = cv2.VideoCapture(temp.name, apiPreference=0, params=[])
            frame_count = 0
            fps = await self.get_fps(context)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count >= self.start and (
                    self.end == -1 or frame_count < self.end
                ):
                    # Convert BGR to RGB
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = PIL.Image.fromarray(rgb_frame)
                    img_ref = await context.image_from_pil(img)
                    yield "frame", img_ref
                    yield "index", frame_count
                    yield "fps", fps
                    yield "event", Event(name="frame")

                if self.end > -1 and frame_count >= self.end:
                    break

                frame_count += 1

            cap.release()

        yield "event", Event(name="done")

    async def get_fps(self, context: ProcessingContext) -> float:
        video_file = await context.asset_to_io(self.video)
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as temp:
            temp.write(video_file.read())
            temp.flush()

            cap = cv2.VideoCapture(temp.name)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()

            return fps


class Fps(BaseNode):
    """
    Get the frames per second (FPS) of a video file.
    video, analysis, frames, fps

    Use cases:
    1. Analyze video properties for quality assessment
    2. Determine appropriate playback speed for video editing
    3. Ensure compatibility with target display systems
    """

    video: VideoRef = Field(
        default=VideoRef(), description="The input video to analyze for FPS."
    )

    async def process(self, context: ProcessingContext) -> float:
        video_file = await context.asset_to_io(self.video)
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as temp:
            temp.write(video_file.read())
            temp.flush()

            cap = cv2.VideoCapture(temp.name)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()

            return fps


class FrameToVideo(BaseNode):
    """
    Combine a sequence of frames into a single video file.
    video, frames, combine, sequence

    Use cases:
    1. Create time-lapse videos from image sequences
    2. Compile processed frames back into a video
    3. Generate animations from individual images
    """

    frame: ImageRef = Field(default=ImageRef(), description="Collect input frames")
    index: int = Field(
        default=0, description="Index of the current frame. -1 signals end of stream."
    )
    fps: float = Field(default=30, description="The FPS of the output video.")
    event: Event = Field(default=Event(name="done"), description="Signal end of stream")

    async def handle_event(self, context: ProcessingContext, event: Event):
        if not self.frame:
            raise ValueError("No frames provided to create video.")

        if event.name == "done":
            yield "output", await self.create_video(context)
        elif event.name == "frame":
            # Save all frames as images in the temporary directory
            img = await context.image_to_pil(self.frame)
            frame_path = context.resolve_workspace_path(f"frame_{self.index:05d}.png")
            img.save(frame_path)
            logger.debug("Saved frame %s to %s", self.index, frame_path)
        else:
            raise ValueError(f"Unknown event: {event.name}")

    async def create_video(self, context: ProcessingContext) -> VideoRef:
        # Create a temporary file for the output video
        video_path = context.resolve_workspace_path(f"video_{str(uuid.uuid4())}.mp4")
        try:
            # Use FFmpeg to create video from frames
            frame_path = context.resolve_workspace_path("frame_%05d.png")
            logger.debug("Creating video from %s", frame_path)
            (
                ffmpeg.input(frame_path, framerate=self.fps)
                .output(video_path, vcodec="libx264", pix_fmt="yuv420p")
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )

            # Read the created video and return as VideoRef
            with open(video_path, "rb") as f:
                return await context.video_from_io(f)

        except ffmpeg.Error as e:
            logger.error("FFmpeg stdout:\n%s", e.stdout.decode("utf8"))
            logger.error("FFmpeg stderr:\n%s", e.stderr.decode("utf8"))
            raise RuntimeError(f"Error creating video: {e.stderr.decode('utf8')}")


class Concat(BaseNode):
    """
    Concatenate multiple video files into a single video, including audio when available.
    video, concat, merge, combine, audio, +
    """

    video_a: VideoRef = Field(
        default=VideoRef(), description="The first video to concatenate."
    )
    video_b: VideoRef = Field(
        default=VideoRef(), description="The second video to concatenate."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        import ffmpeg

        if self.video_a.is_empty() or self.video_b.is_empty():
            raise ValueError("Both videos must be connected.")

        video_a = await context.asset_to_io(self.video_a)
        video_b = await context.asset_to_io(self.video_b)

        with tempfile.NamedTemporaryFile(
            suffix=".mp4", delete=False
        ) as temp_a, tempfile.NamedTemporaryFile(
            suffix=".mp4", delete=False
        ) as temp_b, tempfile.NamedTemporaryFile(
            suffix=".txt", delete=False
        ) as temp_list, tempfile.NamedTemporaryFile(
            suffix=".mp4", delete=False
        ) as output_temp:
            try:
                temp_a.write(video_a.read())
                temp_b.write(video_b.read())
                temp_a.close()
                temp_b.close()
                output_temp.close()

                # Create a list file for concatenation
                with open(temp_list.name, "w") as f:
                    f.write(f"file '{temp_a.name}'\n")
                    f.write(f"file '{temp_b.name}'\n")
                temp_list.close()

                # Check if both videos have audio streams
                probe_a = ffmpeg.probe(temp_a.name)
                probe_b = ffmpeg.probe(temp_b.name)
                has_audio_a = any(
                    stream["codec_type"] == "audio" for stream in probe_a["streams"]
                )
                has_audio_b = any(
                    stream["codec_type"] == "audio" for stream in probe_b["streams"]
                )

                # Use ffmpeg-python to concatenate videos
                input_stream = ffmpeg.input(temp_list.name, format="concat", safe=0)

                if has_audio_a and has_audio_b:
                    output = ffmpeg.output(input_stream, output_temp.name, c="copy")
                else:
                    output = ffmpeg.output(
                        input_stream.video, output_temp.name, vcodec="libx264"
                    )

                output.overwrite_output().run(
                    quiet=True, capture_stdout=True, capture_stderr=True
                )

                # Read the concatenated video and create a VideoRef
                with open(output_temp.name, "rb") as f:
                    return await context.video_from_io(f)
            except ffmpeg.Error as e:
                logger.error("FFmpeg stdout:\n%s", e.stdout.decode("utf8"))
                logger.error("FFmpeg stderr:\n%s", e.stderr.decode("utf8"))
                raise RuntimeError(
                    f"Error concatenating videos: {e.stderr.decode('utf8')}"
                )
            finally:
                safe_unlink(temp_a.name)
                safe_unlink(temp_b.name)
                safe_unlink(temp_list.name)
                safe_unlink(output_temp.name)


class Trim(BaseNode):
    """
    Trim a video to a specific start and end time.
    video, trim, cut, segment

    Use cases:
    1. Extract specific segments from a longer video
    2. Remove unwanted parts from the beginning or end of a video
    3. Create shorter clips from a full-length video
    """

    video: VideoRef = Field(default=VideoRef(), description="The input video to trim.")
    start_time: float = Field(
        default=0.0, description="The start time in seconds for the trimmed video."
    )
    end_time: float = Field(
        default=-1.0,
        description="The end time in seconds for the trimmed video. Use -1 for the end of the video.",
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        import ffmpeg

        if self.video.is_empty():
            raise ValueError("Input video must be connected.")

        video_file = await context.asset_to_io(self.video)

        with tempfile.NamedTemporaryFile(
            suffix=".mp4", delete=False
        ) as temp, tempfile.NamedTemporaryFile(suffix=".mp4") as output_temp:
            try:
                temp.write(video_file.read())
                temp.close()
                output_temp.close()

                input_stream = ffmpeg.input(temp.name)

                # Check if the video has an audio stream
                probe = ffmpeg.probe(temp.name)
                has_audio = any(
                    stream["codec_type"] == "audio" for stream in probe["streams"]
                )

                # Apply trimming to video stream
                if self.end_time > 0:
                    trimmed_video = input_stream.video.trim(
                        start=self.start_time, end=self.end_time
                    ).setpts("PTS-STARTPTS")
                else:
                    trimmed_video = input_stream.video.trim(
                        start=self.start_time
                    ).setpts("PTS-STARTPTS")

                # Apply trimming to audio stream if it exists
                if has_audio:
                    if self.end_time > 0:
                        trimmed_audio = input_stream.audio.filter_(
                            "atrim", start=self.start_time, end=self.end_time
                        ).filter_("asetpts", "PTS-STARTPTS")
                    else:
                        trimmed_audio = input_stream.audio.filter_(
                            "atrim", start=self.start_time
                        ).filter_("asetpts", "PTS-STARTPTS")

                    # Output both video and audio
                    ffmpeg.output(
                        trimmed_video, trimmed_audio, output_temp.name
                    ).overwrite_output().run(quiet=False)
                else:
                    # Output only video
                    ffmpeg.output(
                        trimmed_video, output_temp.name
                    ).overwrite_output().run(quiet=False)

                # Read the trimmed video and create a VideoRef
                with open(output_temp.name, "rb") as f:
                    return await context.video_from_io(f)
            finally:
                safe_unlink(temp.name)
                safe_unlink(output_temp.name)


class ResizeNode(BaseNode):
    """
    Resize a video to a specific width and height.
    video, resize, scale, dimensions

    Use cases:
    1. Adjust video resolution for different display requirements
    2. Reduce file size by downscaling video
    3. Prepare videos for specific platforms with size constraints
    """

    video: VideoRef = Field(
        default=VideoRef(), description="The input video to resize."
    )
    width: int = Field(
        default=-1, description="The target width. Use -1 to maintain aspect ratio."
    )
    height: int = Field(
        default=-1, description="The target height. Use -1 to maintain aspect ratio."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        import ffmpeg

        if self.video.is_empty():
            raise ValueError("Input video must be connected.")

        video_file = await context.asset_to_io(self.video)

        with tempfile.NamedTemporaryFile(
            suffix=".mp4", delete=False
        ) as temp, tempfile.NamedTemporaryFile(
            suffix=".mp4", delete=False
        ) as output_temp:
            try:
                temp.write(video_file.read())
                temp.close()
                output_temp.close()

                input_stream = ffmpeg.input(temp.name)

                # Apply resizing to video stream
                resized = input_stream.video.filter("scale", self.width, self.height)

                # Check if audio stream exists
                probe = ffmpeg.probe(temp.name)
                has_audio = any(
                    stream["codec_type"] == "audio" for stream in probe["streams"]
                )

                if has_audio:
                    # Keep audio stream unchanged if it exists
                    audio = input_stream.audio
                    ffmpeg.output(
                        resized, audio, output_temp.name
                    ).overwrite_output().run(quiet=False)
                else:
                    # Output only video if no audio stream
                    ffmpeg.output(resized, output_temp.name).overwrite_output().run(
                        quiet=False
                    )

                # Read the resized video and create a VideoRef
                with open(output_temp.name, "rb") as f:
                    return await context.video_from_io(f)
            finally:
                safe_unlink(temp.name)
                safe_unlink(output_temp.name)


class Rotate(BaseNode):
    """
    Rotate a video by a specified angle.
    video, rotate, orientation, transform

    Use cases:
    1. Correct orientation of videos taken with a rotated camera
    2. Create artistic effects by rotating video content
    3. Adjust video for different display orientations
    """

    video: VideoRef = Field(
        default=VideoRef(), description="The input video to rotate."
    )
    angle: float = Field(
        default=0.0, description="The angle of rotation in degrees.", ge=-360, le=360
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        import ffmpeg

        if self.video.is_empty():
            raise ValueError("Input video must be connected.")

        video_file = await context.asset_to_io(self.video)

        with tempfile.NamedTemporaryFile(
            suffix=".mp4", delete=False
        ) as temp, tempfile.NamedTemporaryFile(
            suffix=".mp4", delete=False
        ) as output_temp:
            try:
                temp.write(video_file.read())
                temp.close()
                output_temp.close()

                input_stream = ffmpeg.input(temp.name)

                # Apply rotation to video stream
                angle_rad = np.radians(self.angle)
                rotated = input_stream.video.filter("rotate", angle=angle_rad)

                # Check if audio stream exists
                probe = ffmpeg.probe(temp.name)
                has_audio = any(
                    stream["codec_type"] == "audio" for stream in probe["streams"]
                )

                if has_audio:
                    # Keep audio stream unchanged if it exists
                    audio = input_stream.audio
                    ffmpeg.output(
                        rotated, audio, output_temp.name
                    ).overwrite_output().run(quiet=False)
                else:
                    # Output only video if no audio stream
                    ffmpeg.output(rotated, output_temp.name).overwrite_output().run(
                        quiet=False
                    )

                # Read the rotated video and create a VideoRef
                with open(output_temp.name, "rb") as f:
                    return await context.video_from_io(f)
            finally:
                safe_unlink(temp.name)
                safe_unlink(output_temp.name)


class SetSpeed(BaseNode):
    """
    Adjust the playback speed of a video.
    video, speed, tempo, time

    Use cases:
    1. Create slow-motion effects by decreasing video speed
    2. Generate time-lapse videos by increasing playback speed
    3. Synchronize video duration with audio or other timing requirements
    """

    video: VideoRef = Field(
        default=VideoRef(), description="The input video to adjust speed."
    )
    speed_factor: float = Field(
        default=1.0,
        description="The speed adjustment factor. Values > 1 speed up, < 1 slow down.",
        gt=0,
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        import ffmpeg

        if self.video.is_empty():
            raise ValueError("Input video must be connected.")

        video_file = await context.asset_to_io(self.video)

        with tempfile.NamedTemporaryFile(
            suffix=".mp4", delete=False
        ) as temp, tempfile.NamedTemporaryFile(
            suffix=".mp4", delete=False
        ) as output_temp:
            try:
                temp.write(video_file.read())
                temp.close()
                output_temp.close()

                input_stream = ffmpeg.input(temp.name)

                # Check if video has audio
                probe = ffmpeg.probe(temp.name)
                has_audio = any(
                    stream["codec_type"] == "audio" for stream in probe["streams"]
                )

                # Apply speed adjustment to video
                adjusted_video = input_stream.filter(
                    "setpts", f"{1/self.speed_factor}*PTS"
                )

                if has_audio:
                    # Apply speed adjustment to audio
                    # Note: atempo filter is limited to 0.5-2.0 range, so we chain filters for larger adjustments
                    adjusted_audio = input_stream.audio
                    remaining_tempo = self.speed_factor

                    while remaining_tempo > 2.0:
                        adjusted_audio = adjusted_audio.filter("atempo", 2.0)
                        remaining_tempo /= 2.0
                    while remaining_tempo < 0.5:
                        adjusted_audio = adjusted_audio.filter("atempo", 0.5)
                        remaining_tempo *= 2.0
                    if remaining_tempo != 1.0:
                        adjusted_audio = adjusted_audio.filter(
                            "atempo", remaining_tempo
                        )

                    # Output with adjusted audio
                    ffmpeg.output(
                        adjusted_video, adjusted_audio, output_temp.name
                    ).overwrite_output().run(quiet=False)
                else:
                    # Output video only
                    ffmpeg.output(
                        adjusted_video, output_temp.name
                    ).overwrite_output().run(quiet=False)

                # Read the speed-adjusted video and create a VideoRef
                with open(output_temp.name, "rb") as f:
                    return await context.video_from_io(f)
            finally:
                safe_unlink(temp.name)
                safe_unlink(output_temp.name)


class Overlay(BaseNode):
    """
    Overlay one video on top of another, including audio overlay when available.
    video, overlay, composite, picture-in-picture, audio
    """

    main_video: VideoRef = Field(
        default=VideoRef(), description="The main (background) video."
    )
    overlay_video: VideoRef = Field(
        default=VideoRef(), description="The video to overlay on top."
    )
    x: int = Field(default=0, description="X-coordinate for overlay placement.")
    y: int = Field(default=0, description="Y-coordinate for overlay placement.")
    scale: float = Field(
        default=1.0, description="Scale factor for the overlay video.", gt=0
    )
    overlay_audio_volume: float = Field(
        default=0.5,
        description="Volume of the overlay audio relative to the main audio.",
        ge=0,
        le=1,
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        import ffmpeg

        if self.main_video.is_empty() or self.overlay_video.is_empty():
            raise ValueError("Both main and overlay videos must be connected.")

        main_video_file = await context.asset_to_io(self.main_video)
        overlay_video_file = await context.asset_to_io(self.overlay_video)

        with tempfile.NamedTemporaryFile(
            suffix=".mp4", delete=False
        ) as main_temp, tempfile.NamedTemporaryFile(
            suffix=".mp4", delete=False
        ) as overlay_temp, tempfile.NamedTemporaryFile(
            suffix=".mp4", delete=False
        ) as output_temp:
            try:
                main_temp.write(main_video_file.read())
                overlay_temp.write(overlay_video_file.read())
                main_temp.close()
                overlay_temp.close()
                output_temp.close()

                main_input = ffmpeg.input(main_temp.name)
                overlay_input = ffmpeg.input(overlay_temp.name)

                # Check if videos have audio streams
                probe_main = ffmpeg.probe(main_temp.name)
                probe_overlay = ffmpeg.probe(overlay_temp.name)
                has_audio_main = any(
                    stream["codec_type"] == "audio" for stream in probe_main["streams"]
                )
                has_audio_overlay = any(
                    stream["codec_type"] == "audio"
                    for stream in probe_overlay["streams"]
                )

                # Scale the overlay video
                scaled_overlay = overlay_input.filter(
                    "scale", f"iw*{self.scale}", f"ih*{self.scale}"
                )

                # Apply the video overlay
                video_output = ffmpeg.overlay(
                    main_input.video, scaled_overlay, x=self.x, y=self.y
                )

                # Mix the audio streams if both videos have audio
                if has_audio_main and has_audio_overlay:
                    main_audio = main_input.audio
                    overlay_audio = overlay_input.audio.filter(
                        "volume", volume=self.overlay_audio_volume
                    )
                    audio_output = ffmpeg.filter(
                        [main_audio, overlay_audio],
                        "amix",
                        inputs=2,
                        duration="longest",
                    )
                    ffmpeg.output(
                        video_output, audio_output, output_temp.name
                    ).overwrite_output().run(quiet=False)
                elif has_audio_main:
                    ffmpeg.output(
                        video_output, main_input.audio, output_temp.name
                    ).overwrite_output().run(quiet=False)
                elif has_audio_overlay:
                    overlay_audio = overlay_input.audio.filter(
                        "volume", volume=self.overlay_audio_volume
                    )
                    ffmpeg.output(
                        video_output, overlay_audio, output_temp.name
                    ).overwrite_output().run(quiet=False)
                else:
                    ffmpeg.output(
                        video_output, output_temp.name
                    ).overwrite_output().run(quiet=False)

                # Read the overlaid video and create a VideoRef
                with open(output_temp.name, "rb") as f:
                    return await context.video_from_io(f)
            except ffmpeg.Error as e:
                logger.error("stdout: %s", e.stdout.decode("utf8"))
                logger.error("stderr: %s", e.stderr.decode("utf8"))
                raise RuntimeError(f"ffmpeg error: {e.stderr.decode('utf8')}")
            finally:
                safe_unlink(main_temp.name)
                safe_unlink(overlay_temp.name)
                safe_unlink(output_temp.name)


class ColorBalance(BaseNode):
    """
    Adjust the color balance of a video.
    video, color, balance, adjustment

    Use cases:
    1. Correct color casts in video footage
    2. Enhance specific color tones for artistic effect
    3. Normalize color balance across multiple video clips
    """

    video: VideoRef = Field(
        default=VideoRef(), description="The input video to adjust color balance."
    )
    red_adjust: float = Field(
        default=1.0, description="Red channel adjustment factor.", ge=0, le=2
    )
    green_adjust: float = Field(
        default=1.0, description="Green channel adjustment factor.", ge=0, le=2
    )
    blue_adjust: float = Field(
        default=1.0, description="Blue channel adjustment factor.", ge=0, le=2
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        import ffmpeg
        import tempfile

        if self.video.is_empty():
            raise ValueError("Input video must be connected.")

        video_file = await context.asset_to_io(self.video)

        with tempfile.NamedTemporaryFile(
            suffix=".mp4", delete=False
        ) as temp_input, tempfile.NamedTemporaryFile(
            suffix=".mp4", delete=False
        ) as temp_output:

            # Write input video to temporary file
            temp_input.write(video_file.read())
            temp_input.close()
            temp_output.close()

            try:
                # Get input stream
                input_stream = ffmpeg.input(temp_input.name)

                # Check if video has audio
                probe = ffmpeg.probe(temp_input.name)
                has_audio = any(
                    stream["codec_type"] == "audio" for stream in probe["streams"]
                )

                # Apply color balance adjustment to video stream only
                adjusted_video = input_stream.video.filter(
                    "colorbalance",
                    rs=self.red_adjust - 1,
                    gs=self.green_adjust - 1,
                    bs=self.blue_adjust - 1,
                )

                if has_audio:
                    # If there's audio, include it in the output
                    output = ffmpeg.output(
                        adjusted_video,
                        input_stream.audio,
                        temp_output.name,
                        vcodec="libx264",
                        acodec="copy",
                    )
                else:
                    # Video only output
                    output = ffmpeg.output(
                        adjusted_video,
                        temp_output.name,
                        vcodec="libx264",
                    )

                # Run ffmpeg process
                output.overwrite_output().run(quiet=True)

                # Read the processed video and create a VideoRef
                with open(temp_output.name, "rb") as f:
                    return await context.video_from_io(f)

            except ffmpeg.Error as e:
                raise ValueError(f"Error processing video: {e.stderr.decode()}")

            finally:
                # Clean up temporary files
                safe_unlink(temp_input.name)
                safe_unlink(temp_output.name)


class Denoise(BaseNode):
    """
    Apply noise reduction to a video.
    video, denoise, clean, enhance

    Use cases:
    1. Improve video quality by reducing unwanted noise
    2. Enhance low-light footage
    3. Prepare video for further processing or compression
    """

    video: VideoRef = Field(
        default=VideoRef(), description="The input video to denoise."
    )
    strength: float = Field(
        default=5.0,
        description="Strength of the denoising effect. Higher values mean more denoising.",
        ge=0,
        le=20,
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        import ffmpeg
        import tempfile

        if self.video.is_empty():
            raise ValueError("Input video must be connected.")

        video_file = await context.asset_to_io(self.video)

        with tempfile.NamedTemporaryFile(
            suffix=".mp4", delete=False
        ) as temp_input, tempfile.NamedTemporaryFile(
            suffix=".mp4", delete=False
        ) as temp_output:

            # Write input video to temporary file
            temp_input.write(video_file.read())
            temp_input.close()
            temp_output.close()

            try:
                # Get input stream and check for audio
                input_stream = ffmpeg.input(temp_input.name)
                probe = ffmpeg.probe(temp_input.name)
                has_audio = any(
                    stream["codec_type"] == "audio" for stream in probe["streams"]
                )

                # Apply denoising filter to video stream only
                denoised = input_stream.video.filter("nlmeans", s=self.strength)

                if has_audio:
                    # Combine denoised video with original audio
                    output = ffmpeg.output(
                        denoised,
                        input_stream.audio,
                        temp_output.name,
                        acodec="copy",  # Copy audio without re-encoding
                    )
                else:
                    output = ffmpeg.output(denoised, temp_output.name)

                # Run ffmpeg process
                ffmpeg.run(output, quiet=True, overwrite_output=True)

                # Read the processed video and create a VideoRef
                with open(temp_output.name, "rb") as f:
                    return await context.video_from_io(f)

            except ffmpeg.Error as e:
                raise ValueError(f"Error processing video: {e.stderr.decode()}")

            finally:
                # Clean up temporary files
                safe_unlink(temp_input.name)
                safe_unlink(temp_output.name)


class Stabilize(BaseNode):
    """
    Apply video stabilization to reduce camera shake and jitter.
    video, stabilize, smooth, shake-reduction

    Use cases:
    1. Improve quality of handheld or action camera footage
    2. Smooth out panning and tracking shots
    3. Enhance viewer experience by reducing motion sickness
    """

    video: VideoRef = Field(
        default=VideoRef(), description="The input video to stabilize."
    )
    smoothing: float = Field(
        default=10.0,
        description="Smoothing strength. Higher values result in smoother but potentially more cropped video.",
        ge=1,
        le=100,
    )
    crop_black: bool = Field(
        default=True,
        description="Whether to crop black borders that may appear after stabilization.",
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        import ffmpeg
        import tempfile

        if self.video.is_empty():
            raise ValueError("Input video must be connected.")

        video_file = await context.asset_to_io(self.video)

        with tempfile.NamedTemporaryFile(
            suffix=".mp4", delete=False
        ) as temp_input, tempfile.NamedTemporaryFile(
            suffix=".mp4", delete=False
        ) as temp_output:

            # Write input video to temporary file
            temp_input.write(video_file.read())
            temp_input.close()
            temp_output.close()

            try:
                # Get input stream and check for audio
                input_stream = ffmpeg.input(temp_input.name)
                probe = ffmpeg.probe(temp_input.name)
                has_audio = any(
                    stream["codec_type"] == "audio" for stream in probe["streams"]
                )

                # Apply stabilization to video stream only
                stabilized = input_stream.video.filter("deshake", smooth=self.smoothing)
                if self.crop_black:
                    stabilized = stabilized.filter("cropdetect").filter("crop")

                if has_audio:
                    # Combine stabilized video with original audio
                    output = ffmpeg.output(
                        stabilized, input_stream.audio, temp_output.name, acodec="copy"
                    )
                else:
                    output = ffmpeg.output(stabilized, temp_output.name)

                # Run ffmpeg process
                ffmpeg.run(output, quiet=True, overwrite_output=True)

                # Read the processed video and create a VideoRef
                with open(temp_output.name, "rb") as f:
                    return await context.video_from_io(f)

            except ffmpeg.Error as e:
                raise ValueError(f"Error processing video: {e.stderr.decode()}")

            finally:
                # Clean up temporary files
                safe_unlink(temp_input.name)
                safe_unlink(temp_output.name)


class Sharpness(BaseNode):
    """
    Adjust the sharpness of a video.
    video, sharpen, enhance, detail

    Use cases:
    1. Enhance detail in slightly out-of-focus footage
    2. Correct softness introduced by video compression
    3. Create stylistic effects by over-sharpening
    """

    video: VideoRef = Field(
        default=VideoRef(), description="The input video to sharpen."
    )
    luma_amount: float = Field(
        default=1.0,
        description="Amount of sharpening to apply to luma (brightness) channel.",
        ge=0,
        le=3,
    )
    chroma_amount: float = Field(
        default=0.5,
        description="Amount of sharpening to apply to chroma (color) channels.",
        ge=0,
        le=3,
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        import ffmpeg
        import tempfile

        if self.video.is_empty():
            raise ValueError("Input video must be connected.")

        video_file = await context.asset_to_io(self.video)

        with tempfile.NamedTemporaryFile(
            suffix=".mp4", delete=False
        ) as temp_input, tempfile.NamedTemporaryFile(
            suffix=".mp4", delete=False
        ) as temp_output:

            # Write input video to temporary file
            temp_input.write(video_file.read())
            temp_input.close()
            temp_output.close()

            try:
                # Get input stream and check for audio
                input_stream = ffmpeg.input(temp_input.name)
                probe = ffmpeg.probe(temp_input.name)
                has_audio = any(
                    stream["codec_type"] == "audio" for stream in probe["streams"]
                )

                # Apply sharpening to video stream only
                sharpened = input_stream.video.filter(
                    "unsharp",
                    luma_msize_x=5,
                    luma_msize_y=5,
                    luma_amount=self.luma_amount,
                    chroma_msize_x=5,
                    chroma_msize_y=5,
                    chroma_amount=self.chroma_amount,
                )

                if has_audio:
                    # Combine sharpened video with original audio
                    output = ffmpeg.output(
                        sharpened, input_stream.audio, temp_output.name, acodec="copy"
                    )
                else:
                    output = ffmpeg.output(sharpened, temp_output.name)

                # Run ffmpeg process
                ffmpeg.run(output, quiet=True, overwrite_output=True)

                # Read the processed video and create a VideoRef
                with open(temp_output.name, "rb") as f:
                    return await context.video_from_io(f)

            except ffmpeg.Error as e:
                raise ValueError(f"Error processing video: {e.stderr.decode()}")

            finally:
                # Clean up temporary files
                safe_unlink(temp_input.name)
                safe_unlink(temp_output.name)


class Blur(BaseNode):
    """
    Apply a blur effect to a video.
    video, blur, smooth, soften

    Use cases:
    1. Create a dreamy or soft focus effect
    2. Obscure or censor specific areas of the video
    3. Reduce noise or grain in low-quality footage
    """

    video: VideoRef = Field(
        default=VideoRef(), description="The input video to apply blur effect."
    )
    strength: float = Field(
        default=5.0,
        description="The strength of the blur effect. Higher values create a stronger blur.",
        ge=0,
        le=20,
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        import ffmpeg
        import tempfile

        if self.video.is_empty():
            raise ValueError("Input video must be connected.")

        video_file = await context.asset_to_io(self.video)

        with tempfile.NamedTemporaryFile(
            suffix=".mp4", delete=False
        ) as temp_input, tempfile.NamedTemporaryFile(
            suffix=".mp4", delete=False
        ) as temp_output:

            # Write input video to temporary file
            temp_input.write(video_file.read())
            temp_input.close()
            temp_output.close()

            try:
                # Get input stream and check for audio
                input_stream = ffmpeg.input(temp_input.name)
                probe = ffmpeg.probe(temp_input.name)
                has_audio = any(
                    stream["codec_type"] == "audio" for stream in probe["streams"]
                )

                # Apply blur to video stream only
                blurred = input_stream.video.filter(
                    "boxblur", luma_radius=self.strength
                )

                if has_audio:
                    # Combine blurred video with original audio
                    output = ffmpeg.output(
                        blurred, input_stream.audio, temp_output.name, acodec="copy"
                    )
                else:
                    output = ffmpeg.output(blurred, temp_output.name)

                # Run ffmpeg process
                ffmpeg.run(output, quiet=True, overwrite_output=True)

                # Read the processed video and create a VideoRef
                with open(temp_output.name, "rb") as f:
                    return await context.video_from_io(f)

            except ffmpeg.Error as e:
                raise ValueError(f"Error processing video: {e.stderr.decode()}")

            finally:
                # Clean up temporary files
                safe_unlink(temp_input.name)
                safe_unlink(temp_output.name)


class Saturation(BaseNode):
    """
    Adjust the color saturation of a video.
    video, saturation, color, enhance

    Use cases:
    1. Enhance color vibrancy in dull or flat-looking footage
    2. Create stylistic effects by over-saturating or desaturating video
    3. Correct oversaturated footage from certain cameras
    """

    video: VideoRef = Field(
        default=VideoRef(), description="The input video to adjust saturation."
    )
    saturation: float = Field(
        default=1.0,
        description="Saturation level. 1.0 is original, <1 decreases saturation, >1 increases saturation.",
        ge=0,
        le=3,
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        import ffmpeg
        import tempfile

        if self.video.is_empty():
            raise ValueError("Input video must be connected.")

        video_file = await context.asset_to_io(self.video)

        with tempfile.NamedTemporaryFile(
            suffix=".mp4", delete=False
        ) as temp_input, tempfile.NamedTemporaryFile(
            suffix=".mp4", delete=False
        ) as temp_output:

            # Write input video to temporary file
            temp_input.write(video_file.read())
            temp_input.close()
            temp_output.close()

            try:
                # Get input stream and check for audio
                input_stream = ffmpeg.input(temp_input.name)
                probe = ffmpeg.probe(temp_input.name)
                has_audio = any(
                    stream["codec_type"] == "audio" for stream in probe["streams"]
                )

                # Apply saturation adjustment to video stream only
                saturated = input_stream.video.filter("eq", saturation=self.saturation)

                if has_audio:
                    # Combine saturated video with original audio
                    output = ffmpeg.output(
                        saturated, input_stream.audio, temp_output.name, acodec="copy"
                    )
                else:
                    output = ffmpeg.output(saturated, temp_output.name)

                # Run ffmpeg process
                ffmpeg.run(output, quiet=True, overwrite_output=True)

                # Read the processed video and create a VideoRef
                with open(temp_output.name, "rb") as f:
                    return await context.video_from_io(f)

            except ffmpeg.Error as e:
                raise ValueError(f"Error processing video: {e.stderr.decode()}")

            finally:
                # Clean up temporary files
                safe_unlink(temp_input.name)
                safe_unlink(temp_output.name)


class AddSubtitles(BaseNode):
    """
    Add subtitles to a video.
    video, subtitles, text, caption

    Use cases:
    1. Add translations or closed captions to videos
    2. Include explanatory text or commentary in educational videos
    3. Create lyric videos for music content
    """

    class SubtitleTextAlignment(str, enum.Enum):
        TOP = "top"
        CENTER = "center"
        BOTTOM = "bottom"

    video: VideoRef = Field(
        default=VideoRef(), description="The input video to add subtitles to."
    )
    chunks: list[AudioChunk] = Field(
        default=[], description="Audio chunks to add as subtitles."
    )
    font: FontRef = Field(default=FontRef(name=""), description="The font to use.")
    align: SubtitleTextAlignment = Field(
        default=SubtitleTextAlignment.BOTTOM,
        description="Vertical alignment of subtitles.",
    )
    font_size: int = Field(default=24, ge=1, le=72, description="The font size.")
    font_color: ColorRef = Field(
        default=ColorRef(value="#FFFFFF"), description="The font color."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        if self.video.is_empty():
            raise ValueError("Input video must be connected.")

        video_file = await context.asset_to_io(self.video)

        with tempfile.NamedTemporaryFile(
            suffix=".mp4", delete=False
        ) as temp_input, tempfile.NamedTemporaryFile(
            suffix=".mp4", delete=False
        ) as temp_output:
            try:
                # Write input video to temporary file
                temp_input.write(video_file.read())
                temp_input.close()
                temp_output.close()

                # Get video properties
                cap = cv2.VideoCapture(temp_input.name)
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                # Create VideoWriter
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
                out = cv2.VideoWriter(temp_output.name, fourcc, fps, (width, height))

                # Load font
                font_path = context.get_system_font_path(self.font.name)
                font = PIL.ImageFont.truetype(font_path, self.font_size)

                def wrap_text(
                    text: str, max_width: int, draw: PIL.ImageDraw.ImageDraw
                ) -> list[str]:
                    words = text.split()
                    lines = []
                    current_line = []

                    for word in words:
                        current_line.append(word)
                        line_width = draw.textlength(" ".join(current_line), font=font)

                        if line_width > max_width:
                            if len(current_line) == 1:
                                lines.append(current_line[0])
                                current_line = []
                            else:
                                current_line.pop()
                                lines.append(" ".join(current_line))
                                current_line = [word]

                    if current_line:
                        lines.append(" ".join(current_line))

                    return lines

                # Process each frame
                frame_number = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    current_time = frame_number / fps
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = PIL.Image.fromarray(rgb_frame)
                    draw = PIL.ImageDraw.Draw(pil_image)

                    # Find current subtitle text
                    current_text = ""
                    for chunk in self.chunks:
                        if chunk.timestamp[0] <= current_time <= chunk.timestamp[1]:
                            current_text = chunk.text
                            break

                    if current_text:
                        # Calculate maximum width for text (80% of video width)
                        max_width = int(width * 0.8)

                        # Wrap text into lines
                        lines = wrap_text(current_text, max_width, draw)

                        # Calculate total height of all lines
                        line_spacing = self.font_size * 1.2  # 20% spacing between lines
                        total_height = len(lines) * line_spacing

                        # Calculate starting y position based on alignment
                        padding = 20  # Padding from edges
                        if self.align == self.SubtitleTextAlignment.TOP:
                            y = padding
                        elif self.align == self.SubtitleTextAlignment.CENTER:
                            y = (height - total_height) // 2
                        else:  # BOTTOM
                            y = height - total_height - padding

                        # Draw each line
                        for line in lines:
                            # Calculate line width for centering
                            line_width = draw.textlength(line, font=font)
                            x = (width - line_width) // 2

                            draw.text(
                                (x, y),
                                line,
                                font=font,
                                fill=self.font_color.value,
                            )

                            # Move y position for next line
                            y += line_spacing

                    # Convert back to BGR for OpenCV
                    frame_with_text = cv2.cvtColor(
                        np.array(pil_image), cv2.COLOR_RGB2BGR
                    )
                    out.write(frame_with_text)
                    frame_number += 1

                # Release OpenCV objects
                cap.release()
                out.release()

                # Now combine the processed frames with the original audio using ffmpeg
                import ffmpeg

                # Get the original video with audio
                input_video = ffmpeg.input(temp_input.name)
                # Get the processed frames
                processed_frames = ffmpeg.input(temp_output.name)

                # Create a temporary file for the final output
                with tempfile.NamedTemporaryFile(
                    suffix=".mp4", delete=False
                ) as final_output:
                    final_output.close()

                    # Combine processed frames with original audio
                    ffmpeg.output(
                        processed_frames.video,
                        input_video.audio,
                        final_output.name,
                        acodec="copy",  # Copy audio stream without re-encoding
                    ).overwrite_output().run(quiet=True)

                    # Read the final video and create a VideoRef
                    with open(final_output.name, "rb") as f:
                        return await context.video_from_io(f)

            except Exception as e:
                raise RuntimeError(f"Error processing video: {str(e)}") from e

            finally:
                safe_unlink(temp_input.name)
                safe_unlink(temp_output.name)


class Reverse(BaseNode):
    """
    Reverse the playback of a video.
    video, reverse, backwards, effect

    Use cases:
    1. Create artistic effects by playing video in reverse
    2. Analyze motion or events in reverse order
    3. Generate unique transitions or intros for video projects
    """

    video: VideoRef = Field(
        default=VideoRef(), description="The input video to reverse."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        import ffmpeg

        if self.video.is_empty():
            raise ValueError("Input video must be connected.")

        video_file = await context.asset_to_io(self.video)

        with tempfile.NamedTemporaryFile(
            suffix=".mp4", delete=False
        ) as temp_input, tempfile.NamedTemporaryFile(
            suffix=".mp4", delete=False
        ) as temp_output:
            try:
                temp_input.write(video_file.read())
                temp_input.close()
                temp_output.close()

                input_stream = ffmpeg.input(temp_input.name)
                reversed_video = input_stream.filter("reverse")

                ffmpeg.output(reversed_video, temp_output.name).overwrite_output().run(
                    quiet=False
                )

                # Read the reversed video and create a VideoRef
                with open(temp_output.name, "rb") as f:
                    return await context.video_from_io(f)

            except ffmpeg.Error as e:
                raise RuntimeError(f"ffmpeg error: {e.stderr.decode('utf8')}")

            finally:
                safe_unlink(temp_input.name)
                safe_unlink(temp_output.name)


class Transition(BaseNode):
    """
    Create a transition effect between two videos, including audio transition when available.
    video, transition, effect, merge, audio

    Use cases:
    1. Create smooth transitions between video clips in a montage
    2. Add professional-looking effects to video projects
    3. Blend scenes together for creative storytelling
    4. Smoothly transition between audio tracks of different video clips
    """

    class TransitionType(str, enum.Enum):
        fade = "fade"
        wipeleft = "wipeleft"
        wiperight = "wiperight"
        wipeup = "wipeup"
        wipedown = "wipedown"
        slideleft = "slideleft"
        slideright = "slideright"
        slideup = "slideup"
        slidedown = "slidedown"
        circlecrop = "circlecrop"
        rectcrop = "rectcrop"
        distance = "distance"
        fadeblack = "fadeblack"
        fadewhite = "fadewhite"
        radial = "radial"
        smoothleft = "smoothleft"
        smoothright = "smoothright"
        smoothup = "smoothup"
        smoothdown = "smoothdown"
        circleopen = "circleopen"
        circleclose = "circleclose"
        vertopen = "vertopen"
        vertclose = "vertclose"
        horzopen = "horzopen"
        horzclose = "horzclose"
        dissolve = "dissolve"
        pixelize = "pixelize"
        diagtl = "diagtl"
        diagtr = "diagtr"
        diagbl = "diagbl"
        diagbr = "diagbr"
        hlslice = "hlslice"
        hrslice = "hrslice"
        vuslice = "vuslice"
        vdslice = "vdslice"
        hblur = "hblur"
        fadegrays = "fadegrays"
        wipetl = "wipetl"
        wipetr = "wipetr"
        wipebl = "wipebl"
        wipebr = "wipebr"
        squeezeh = "squeezeh"
        squeezev = "squeezev"
        zoomin = "zoomin"
        fadefast = "fadefast"
        fadeslow = "fadeslow"
        hlwind = "hlwind"
        hrwind = "hrwind"
        vuwind = "vuwind"
        vdwind = "vdwind"
        coverleft = "coverleft"
        coverright = "coverright"
        coverup = "coverup"
        coverdown = "coverdown"
        revealleft = "revealleft"
        revealright = "revealright"
        revealup = "revealup"
        revealdown = "revealdown"

    video_a: VideoRef = Field(
        default=VideoRef(), description="The first video in the transition."
    )
    video_b: VideoRef = Field(
        default=VideoRef(), description="The second video in the transition."
    )
    transition_type: TransitionType = Field(
        default=TransitionType.fade, description="Type of transition effect"
    )
    duration: float = Field(
        default=1.0,
        description="Duration of the transition effect in seconds.",
        ge=0.1,
        le=5.0,
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        import ffmpeg

        if self.video_a.is_empty() or self.video_b.is_empty():
            raise ValueError("Both input videos must be connected.")

        video_a_file = await context.asset_to_io(self.video_a)
        video_b_file = await context.asset_to_io(self.video_b)

        with tempfile.NamedTemporaryFile(
            suffix=".mp4", delete=False
        ) as temp_a, tempfile.NamedTemporaryFile(
            suffix=".mp4", delete=False
        ) as temp_b, tempfile.NamedTemporaryFile(
            suffix=".mp4", delete=False
        ) as temp_output:
            try:
                temp_a.write(video_a_file.read())
                temp_b.write(video_b_file.read())
                temp_a.close()
                temp_b.close()
                temp_output.close()

                input_a = ffmpeg.input(temp_a.name)
                input_b = ffmpeg.input(temp_b.name)

                # Check if videos have audio streams
                probe_a = ffmpeg.probe(temp_a.name)
                probe_b = ffmpeg.probe(temp_b.name)
                has_audio_a = any(
                    stream["codec_type"] == "audio" for stream in probe_a["streams"]
                )
                has_audio_b = any(
                    stream["codec_type"] == "audio" for stream in probe_b["streams"]
                )

                # Get the duration of video_a
                duration_a = float(probe_a["streams"][0]["duration"])

                # Video transition
                video_transition = ffmpeg.filter(
                    [input_a.video, input_b.video],
                    "xfade",
                    transition=self.transition_type.value,
                    duration=self.duration,
                    offset=duration_a - self.duration,
                )

                # Audio transition (crossfade) if both videos have audio
                if has_audio_a and has_audio_b:
                    audio_transition = ffmpeg.filter(
                        [input_a.audio, input_b.audio],
                        "acrossfade",
                        d=self.duration,
                        c1="tri",
                        c2="tri",
                    )
                    output = ffmpeg.output(
                        video_transition, audio_transition, temp_output.name
                    )
                else:
                    output = ffmpeg.output(video_transition, temp_output.name)

                output.overwrite_output().run(quiet=False)

                # Read the transitioned video and create a VideoRef
                with open(temp_output.name, "rb") as f:
                    return await context.video_from_io(f)

            except ffmpeg.Error as e:
                raise RuntimeError(f"ffmpeg error: {e.stderr.decode('utf8')}")

            finally:
                safe_unlink(temp_a.name)
                safe_unlink(temp_b.name)
                safe_unlink(temp_output.name)


class AddAudio(BaseNode):
    """
    Add an audio track to a video, replacing or mixing with existing audio.
    video, audio, soundtrack, merge

    Use cases:
    1. Add background music or narration to a silent video
    2. Replace original audio with a new soundtrack
    3. Mix new audio with existing video sound
    """

    video: VideoRef = Field(
        default=VideoRef(), description="The input video to add audio to."
    )
    audio: AudioRef = Field(
        default=AudioRef(), description="The audio file to add to the video."
    )
    volume: float = Field(
        default=1.0,
        description="Volume adjustment for the added audio. 1.0 is original volume.",
        ge=0,
        le=2,
    )
    mix: bool = Field(
        default=False,
        description="If True, mix new audio with existing. If False, replace existing audio.",
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        import ffmpeg

        if self.video.is_empty() or self.audio.is_empty():
            raise ValueError("Both video and audio inputs must be connected.")

        video_file = await context.asset_to_io(self.video)
        audio_file = await context.asset_to_io(self.audio)

        with tempfile.NamedTemporaryFile(
            suffix=".mp4", delete=False
        ) as temp_video, tempfile.NamedTemporaryFile(
            suffix=".opus", delete=False
        ) as temp_audio, tempfile.NamedTemporaryFile(
            suffix=".mp4", delete=False
        ) as temp_output:
            try:
                temp_video.write(video_file.read())
                temp_audio.write(audio_file.read())
                temp_video.close()
                temp_audio.close()
                temp_output.close()

                # Set permissions for temporary files
                os.chmod(temp_video.name, 0o644)
                os.chmod(temp_audio.name, 0o644)
                os.chmod(temp_output.name, 0o644)

                video_input = ffmpeg.input(temp_video.name)
                audio_input = ffmpeg.input(temp_audio.name)

                audio_input = audio_input.filter("volume", volume=self.volume)

                if self.mix:
                    # Mix new audio with existing video audio
                    audio = ffmpeg.filter(
                        [video_input.audio, audio_input], "amix", inputs=2
                    )
                else:
                    # Replace video audio with new audio
                    audio = audio_input

                ffmpeg.output(
                    video_input.video, audio, temp_output.name, format="mp4"
                ).overwrite_output().run(quiet=False)

                # Read the video with added audio and create a VideoRef
                with open(temp_output.name, "rb") as f:
                    return await context.video_from_io(f)

            except ffmpeg.Error as e:
                raise RuntimeError(f"ffmpeg error: {e.stderr.decode('utf8')}")

            finally:
                safe_unlink(temp_video.name)
                safe_unlink(temp_audio.name)
                safe_unlink(temp_output.name)


class ChromaKey(BaseNode):
    """
    Apply chroma key (green screen) effect to a video.
    video, chroma key, green screen, compositing

    Use cases:
    1. Remove green or blue background from video footage
    2. Create special effects by compositing video onto new backgrounds
    3. Produce professional-looking videos for presentations or marketing
    """

    video: VideoRef = Field(
        default=VideoRef(), description="The input video to apply chroma key effect."
    )
    key_color: ColorRef = Field(
        default=ColorRef(value="#00FF00"),
        description="The color to key out (e.g., '#00FF00' for green).",
    )
    similarity: float = Field(
        default=0.3,
        description="Similarity threshold for the key color.",
        ge=0.0,
        le=1.0,
    )
    blend: float = Field(
        default=0.1, description="Blending of the keyed area edges.", ge=0.0, le=1.0
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        import ffmpeg

        if self.video.is_empty():
            raise ValueError("Input video must be connected.")

        video_file = await context.asset_to_io(self.video)

        with tempfile.NamedTemporaryFile(
            suffix=".mp4", delete=False
        ) as temp_input, tempfile.NamedTemporaryFile(
            suffix=".mp4", delete=False
        ) as temp_output:
            try:
                temp_input.write(video_file.read())
                temp_input.close()
                temp_output.close()

                # Set permissions for temporary files
                os.chmod(temp_input.name, 0o644)
                os.chmod(temp_output.name, 0o644)

                input_stream = ffmpeg.input(temp_input.name)

                # Apply chroma key filter
                keyed = input_stream.filter(
                    "chromakey",
                    color=self.key_color.value,
                    similarity=self.similarity,
                    blend=self.blend,
                )

                ffmpeg.output(keyed, temp_output.name).overwrite_output().run(
                    quiet=False
                )

                # Read the chroma keyed video and create a VideoRef
                with open(temp_output.name, "rb") as f:
                    return await context.video_from_io(f)

            except ffmpeg.Error as e:
                raise RuntimeError(f"ffmpeg error: {e.stderr.decode('utf8')}")

            finally:
                safe_unlink(temp_input.name)
                safe_unlink(temp_output.name)


class ExtractAudio(BaseNode):
    """
    Separate audio from a video file.
    video, audio, extract, separate
    """

    video: VideoRef = Field(
        default=VideoRef(), description="The input video to separate."
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        if self.video.is_empty():
            raise ValueError("Input video must be connected.")

        video_file = await context.asset_to_io(self.video)

        with tempfile.NamedTemporaryFile(
            suffix=".mp4", delete=False
        ) as temp_input, tempfile.NamedTemporaryFile(
            suffix=".opus", delete=False
        ) as temp_audio:
            try:
                temp_input.write(video_file.read())
                temp_input.close()
                temp_audio.close()

                # Set permissions for temporary files
                os.chmod(temp_input.name, 0o644)
                os.chmod(temp_audio.name, 0o644)

                # Extract the audio using Opus codec
                (
                    ffmpeg.input(temp_input.name)
                    .output(
                        temp_audio.name,
                        acodec="libopus",
                        map="0:a",
                        format="opus",
                        loglevel="error",
                    )
                    .overwrite_output()
                    .run(quiet=True)
                )

                # Read the extracted audio and return it
                with open(temp_audio.name, "rb") as f:
                    return await context.audio_from_io(f, content_type="audio/opus")

            except ffmpeg.Error as e:
                error_message = (
                    e.stderr.decode("utf-8") if e.stderr else "Unknown ffmpeg error."
                )
                raise RuntimeError(f"ffmpeg error: {error_message}") from e

            finally:
                safe_unlink(temp_input.name)
                safe_unlink(temp_audio.name)
