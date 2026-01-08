"""
Example: Photo Enhancement Suite DSL Workflow

This workflow provides batch photo enhancement with AI-powered improvements. It demonstrates:

1. Batch upload photos from a folder
2. Apply AI upscaling and enhancement
3. Color correction and noise reduction
4. Style transfers (vintage, cinematic, etc.)
5. Generate before/after comparisons

The workflow pattern:
    [FolderInput] -> [LoadImageFolder] -> [ForEach]
                        -> [AutoContrast] -> [Sharpen] -> [Color] -> [ImageToImage] -> [Collect] -> [Output]

Tailored for photographers processing shoots efficiently.
"""

from nodetool.dsl.graph import create_graph
from nodetool.dsl.nodetool.input import FolderPathInput, FloatInput
from nodetool.dsl.nodetool.image import (
    LoadImageFolder,
    ImageToImage,
)
from nodetool.dsl.lib.pillow.enhance import (
    AutoContrast,
    Sharpen,
    Color,
    Brightness,
    UnsharpMask,
)
from nodetool.dsl.nodetool.control import Collect
from nodetool.dsl.nodetool.output import Output
from nodetool.metadata.types import (
    Provider,
    ImageModel,
)


def build_photo_enhancement_suite():
    """
    Batch enhance photos with AI-powered improvements.

    This function builds a workflow graph that:
    1. Loads images from a specified folder
    2. Applies auto contrast correction
    3. Sharpens details and enhances color vibrancy
    4. Applies AI-based enhancement via image-to-image
    5. Returns enhanced images

    Returns:
        Graph: A graph object representing the workflow
    """
    # --- Inputs ---
    image_folder = FolderPathInput(
        name="image_folder",
        description="Folder containing photos to enhance",
        value="./photos",
    )

    color_boost = FloatInput(
        name="color_boost",
        description="Color saturation boost factor (1.0 = no change)",
        value=1.2,
        min=0.5,
        max=2.0,
    )

    brightness_adjust = FloatInput(
        name="brightness_adjust",
        description="Brightness adjustment factor",
        value=1.05,
        min=0.5,
        max=1.5,
    )

    # --- Load images from folder ---
    loaded_images = LoadImageFolder(
        folder=image_folder.output,
        include_subdirectories=False,
        extensions=[".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"],
    )

    # --- Process each image through enhancement pipeline ---

    # Step 1: Auto contrast correction
    auto_contrast = AutoContrast(
        image=loaded_images.out.image,
        cutoff=2,
    )

    # Step 2: Brightness adjustment
    brightened = Brightness(
        image=auto_contrast.output,
        factor=brightness_adjust.output,
    )

    # Step 3: Color enhancement
    color_enhanced = Color(
        image=brightened.output,
        factor=color_boost.output,
    )

    # Step 4: Sharpen details
    sharpened = Sharpen(
        image=color_enhanced.output,
    )

    # Step 5: Unsharp mask for fine detail
    unsharp = UnsharpMask(
        image=sharpened.output,
        radius=2,
        percent=120,
        threshold=3,
    )

    # Step 6: AI-based style enhancement
    # Create dynamic prompt based on style
    ai_enhanced = ImageToImage(
        model=ImageModel(
            type="image_model",
            provider=Provider.HuggingFaceFalAI,
            id="fal-ai/flux/dev",
            name="FLUX.1 Dev",
        ),
        image=unsharp.output,
        prompt="Enhance this photo with professional cinematic color grading, improved lighting, and subtle film grain. Keep the subject and composition intact, only improve quality.",
        strength=0.3,  # Low strength to preserve original
        guidance_scale=7.0,
        num_inference_steps=25,
    )

    # --- Collect enhanced images ---
    collected_enhanced = Collect(
        input_item=ai_enhanced.output,
    )

    # --- Also collect original paths for reference ---
    collected_paths = Collect(
        input_item=loaded_images.out.path,
    )

    # --- Outputs ---
    enhanced_out = Output(
        name="enhanced_photos",
        value=collected_enhanced.out.output,
        description="Collection of enhanced photos",
    )

    paths_out = Output(
        name="source_paths",
        value=collected_paths.out.output,
        description="Original file paths for reference",
    )

    return create_graph(enhanced_out, paths_out)


# Build the graph
graph = build_photo_enhancement_suite()


if __name__ == "__main__":
    """
    To run this example:

    1. Create a folder with photos to enhance
    2. Ensure you have API keys configured for FAL AI
    3. Run:

        python examples/photo_enhancement_suite.py

    The workflow batch processes all photos in the specified folder.
    """

    print("Photo Enhancement Suite Workflow")
    print("=" * 50)
    print(f"Nodes: {len(graph.nodes)}")
    print(f"Edges: {len(graph.edges)}")
    print()
    print("Enhancement Pipeline:")
    print("  [FolderPathInput]")
    print("      -> [LoadImageFolder] (load all photos)")
    print("          -> [AutoContrast] (fix exposure)")
    print("              -> [Brightness] (adjust levels)")
    print("                  -> [Color] (boost saturation)")
    print("                      -> [Sharpen] (enhance details)")
    print("                          -> [UnsharpMask] (fine detail)")
    print("                              -> [ImageToImage] (AI enhancement)")
    print("                                  -> [Collect]")
    print("                                      -> [Output]")
    print()
    print("Features:")
    print("  - Automatic contrast correction")
    print("  - Brightness and color adjustment")
    print("  - Detail sharpening with unsharp mask")
    print("  - AI-powered style enhancement")
    print("  - Batch processing support")
    print()

    # Uncomment to run:
    # import asyncio
    # result = asyncio.run(run_graph(graph, user_id="example_user", auth_token="token"))
    # print(f"Enhanced {len(result['enhanced_photos'])} photos")
