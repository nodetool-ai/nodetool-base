"""
Example: Kie Character Animation Studio

This workflow creates animated character videos using Kie.ai's motion control
and character remix capabilities. Similar to tools like Flora for character
animation and consistent character generation.

1. Generate or remix character images
2. Create talking head animations with lip-sync
3. Generate character-consistent video sequences
4. Add background music

    The workflow pattern:
        [CharacterImage] -> [IdeogramCharacterRemix] (style variations)
                               -> [KlingAIAvatarPro] (lip-sync)
                                   -> [GenerateMusic] (theme music)
                                       -> [Output]

Perfect for animators, game developers, and digital content creators.

Note: If imports fail, run 'nodetool package scan && nodetool codegen' to regenerate DSL.
"""

from nodetool.dsl.graph import create_graph
from nodetool.dsl.nodetool.input import StringInput, ImageInput, AudioInput
from nodetool.dsl.nodetool.output import Output
from nodetool.dsl.kie.image import (
    IdeogramCharacterRemix,
    Flux2ProTextToImage,
    Seedream45Edit,
    NanoBananaEdit,
)
from nodetool.dsl.kie.video import (
    KlingAIAvatarPro,
    KlingAIAvatarStandard,
    KlingImageToVideo,
    InfinitalkV1,
)
from nodetool.dsl.kie.audio import GenerateMusic
from nodetool.dsl.nodetool.video import AddAudio
from nodetool.metadata.types import ImageRef, AudioRef


def build_character_animation_studio():
    """
    Create animated character content with Kie AI.

    This function builds a workflow graph that:
    1. Generates or remixes character designs
    2. Applies motion control from reference videos
    3. Creates lip-synced talking animations
    4. Produces character-consistent video content

    Returns:
        Graph: A graph object representing the workflow
    """
    # --- Inputs ---
    character_image = ImageInput(
        name="character_image",
        description="Base character image to animate",
        value=ImageRef(type="image", uri=""),
    )

    character_description = StringInput(
        name="character_description",
        description="Description of the character",
        value="Friendly cartoon mascot character, cute round design, "
        "big expressive eyes, warm smile, vibrant colors, "
        "2D animation style with 3D depth",
    )

    character_audio = AudioInput(
        name="character_audio",
        description="Pre-recorded audio for character lip-sync",
        value=AudioRef(type="audio", uri=""),
    )

    animation_style = StringInput(
        name="animation_style",
        description="Desired animation style",
        value="Smooth, bouncy animation, cartoon physics, "
        "exaggerated expressions, playful movement",
    )

    # --- Generate Base Character ---
    generated_character = Flux2ProTextToImage(
        prompt=character_description.output,
        aspect_ratio=Flux2ProTextToImage.AspectRatio.SQUARE,
        resolution=Flux2ProTextToImage.Resolution.RES_2K,
        steps=30,
        guidance_scale=8.0,
    )

    # --- Character Remix Variations ---
    # Create style variations while maintaining character consistency
    character_action_pose = IdeogramCharacterRemix(
        prompt="Character in dynamic action pose, jumping with joy, "
        "arms raised, excited expression, energetic",
        image=generated_character.output,
        reference_images=[generated_character.output],
        rendering_speed=IdeogramCharacterRemix.RenderingSpeed.QUALITY,
        style=IdeogramCharacterRemix.Style.GENERAL,
        image_size=IdeogramCharacterRemix.ImageSize.SQUARE_HD,
        strength=0.7,
        expand_prompt=True,
    )

    character_waving = IdeogramCharacterRemix(
        prompt="Character waving hello, friendly greeting gesture, "
        "warm welcoming expression, approachable pose",
        image=generated_character.output,
        reference_images=[generated_character.output],
        rendering_speed=IdeogramCharacterRemix.RenderingSpeed.QUALITY,
        style=IdeogramCharacterRemix.Style.GENERAL,
        image_size=IdeogramCharacterRemix.ImageSize.SQUARE_HD,
        strength=0.65,
        expand_prompt=True,
    )

    character_thinking = IdeogramCharacterRemix(
        prompt="Character in thinking pose, hand on chin, "
        "curious expression, contemplative look, slight tilt",
        image=generated_character.output,
        reference_images=[generated_character.output],
        rendering_speed=IdeogramCharacterRemix.RenderingSpeed.QUALITY,
        style=IdeogramCharacterRemix.Style.GENERAL,
        image_size=IdeogramCharacterRemix.ImageSize.SQUARE_HD,
        strength=0.7,
        expand_prompt=True,
    )

    # --- Edit Character with Seedream ---
    character_celebration = Seedream45Edit(
        prompt="Same character celebrating, confetti around, "
        "party atmosphere, joyful expression, hands up in victory",
        image=generated_character.output,
        aspect_ratio=Seedream45Edit.AspectRatio.SQUARE,
        quality=Seedream45Edit.Quality.HIGH,
    )

    # --- Edit Character with Nano Banana ---
    character_night_mode = NanoBananaEdit(
        prompt="Same character in night/evening version, "
        "cozy lighting, sleepy expression, peaceful mood",
        image_input=[generated_character.output],
        image_size=NanoBananaEdit.ImageSize.SQUARE,
    )

    # --- Talking Head Animation ---
    talking_character = KlingAIAvatarPro(
        image=generated_character.output,
        audio=character_audio.output,
        prompt=f"Character speaking expressively, {animation_style.output}, "
        "natural lip sync, engaging personality, direct eye contact",
        mode=KlingAIAvatarPro.Mode.PRO,
    )

    # --- Standard Quality Avatar ---
    talking_standard = KlingAIAvatarStandard(
        image=generated_character.output,
        audio=character_audio.output,
        prompt=f"Animated character speaking, {animation_style.output}, "
        "cartoon-style lip movement",
        mode=KlingAIAvatarStandard.Mode.STANDARD,
    )

    # --- Alternative: Infinitalk for talking animation ---
    infinitalk_character = InfinitalkV1(
        image=generated_character.output,
        audio=character_audio.output,
        prompt=f"Animated character speaking, {animation_style.output}, "
        "cartoon-style lip movement",
        resolution=InfinitalkV1.Resolution.R480P,
    )

    # --- Simple Image-to-Video Animation ---
    waving_animation = KlingImageToVideo(
        prompt=f"Character waving hello with friendly gesture, {animation_style.output}, "
        "looping animation, smooth wave motion",
        image1=character_waving.output,
        duration=5,
        sound=False,
    )

    action_animation = KlingImageToVideo(
        prompt=f"Character in dynamic action, {animation_style.output}, "
        "jumping, bouncing, energetic movement",
        image1=character_action_pose.output,
        duration=5,
        sound=False,
    )

    # --- Generate Character Theme Music ---
    character_theme = GenerateMusic(
        prompt="Cute character theme music (~30 seconds), playful melody, "
        "upbeat tempo, kid-friendly, mascot jingle, memorable tune",
        style="pop",
        instrumental=True,
        model=GenerateMusic.Model.V4_5PLUS,
    )

    # --- Combine Talking Animation with Music ---
    final_with_music = AddAudio(
        video=talking_character.output,
        audio=character_theme.output,
        volume=0.2,  # Low background music
        mix=True,
    )

    # --- Outputs ---
    # Animations
    talking_output = Output(
        name="talking_animation",
        value=talking_character.output,
        description="Talking character animation (Kling AI Avatar Pro)",
    )

    talking_with_music = Output(
        name="talking_with_music",
        value=final_with_music.output,
        description="Talking animation with background music",
    )

    talking_standard_output = Output(
        name="talking_standard",
        value=talking_standard.output,
        description="Standard quality talking animation",
    )

    waving_output = Output(
        name="waving_animation",
        value=waving_animation.output,
        description="Waving gesture animation",
    )

    action_output = Output(
        name="action_animation",
        value=action_animation.output,
        description="Action pose animation",
    )

    infinitalk_output = Output(
        name="infinitalk_animation",
        value=infinitalk_character.output,
        description="Infinitalk talking animation",
    )

    # Character Images
    base_character = Output(
        name="base_character",
        value=generated_character.output,
        description="Base character design",
    )

    action_pose = Output(
        name="character_action",
        value=character_action_pose.output,
        description="Character in action pose (remix)",
    )

    waving_pose = Output(
        name="character_waving",
        value=character_waving.output,
        description="Character waving (remix)",
    )

    thinking_pose = Output(
        name="character_thinking",
        value=character_thinking.output,
        description="Character thinking (remix)",
    )

    celebration_pose = Output(
        name="character_celebration",
        value=character_celebration.output,
        description="Character celebrating (Seedream edit)",
    )

    night_pose = Output(
        name="character_night",
        value=character_night_mode.output,
        description="Character night version (Nano Banana edit)",
    )

    # Audio
    theme_output = Output(
        name="character_theme",
        value=character_theme.output,
        description="Character theme music",
    )

    character_image_input = Output(
        name="character_image_input",
        value=character_image.output,
        description="Optional input character image (if provided)",
    )

    return create_graph(
        talking_output,
        talking_with_music,
        talking_standard_output,
        waving_output,
        action_output,
        infinitalk_output,
        character_image_input,
        base_character,
        action_pose,
        waving_pose,
        thinking_pose,
        celebration_pose,
        night_pose,
        theme_output,
    )


# Build the graph
graph = build_character_animation_studio()


if __name__ == "__main__":
    """
    To run this example:

    1. Ensure you have KIE_API_KEY configured
    2. Provide a pre-recorded audio file for character dialogue
    3. Run:

        python examples/kie_character_animation_studio.py

    The workflow creates animated character content.
    """

    print("Kie Character Animation Studio")
    print("=" * 50)
    print(f"Nodes: {len(graph.nodes)}")
    print(f"Edges: {len(graph.edges)}")
    print()
    print("Character Generation:")
    print("  - Flux 2 Pro - Base character generation")
    print("  - Ideogram Character Remix - Pose variations")
    print("  - Seedream 4.5 Edit - Character edits")
    print("  - Nano Banana Edit - Style variations")
    print()
    print("Animation Capabilities:")
    print("  - Kling AI Avatar Pro/Standard - Lip-sync talking head")
    print("  - Kling Image-to-Video - General animation")
    print("  - Infinitalk V1 - Alternative talking animation")
    print()
    print("Audio Generation:")
    print("  - GenerateMusic (Suno via Kie.ai) - Character theme music")
    print()
    print("Use Cases:")
    print("  - YouTube mascot animations")
    print("  - Educational character content")
    print("  - Game character previews")
    print("  - Social media mascot videos")
    print("  - Brand ambassador content")
    print()

    # Uncomment to run:
    # result = run_graph(graph)
    # print(result)
