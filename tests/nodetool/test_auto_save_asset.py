"""
Test that _auto_save_asset attribute is properly set on generative nodes only.
"""
import inspect
import pytest
from nodetool.workflows.base_node import BaseNode
from nodetool.metadata.types import ImageRef, AudioRef, VideoRef


def get_all_node_classes():
    """Get all BaseNode subclasses from the nodetool.nodes package."""
    import importlib
    import pkgutil
    import nodetool.nodes

    node_classes = []

    def walk_packages(package):
        """Recursively walk through all packages and modules."""
        for importer, modname, ispkg in pkgutil.walk_packages(
            path=package.__path__, prefix=package.__name__ + "."
        ):
            try:
                module = importlib.import_module(modname)
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if (
                        issubclass(obj, BaseNode)
                        and obj is not BaseNode
                        and obj.__module__ == modname
                    ):
                        node_classes.append((modname, name, obj))
            except (ImportError, AttributeError):
                # Skip modules that can't be imported
                pass

    walk_packages(nodetool.nodes)
    return node_classes


def node_is_generative(node_name):
    """Check if a node is generative (creates content from scratch)."""
    generative_keywords = [
        "TextToImage",
        "ImageToImage", 
        "TextToVideo",
        "ImageToVideo",
        "FrameToVideo",
        "TextToSpeech",
        "CreateSilence",
        "GenerateMusic",
        "GenerateLyrics",
        "GenerateMusicVideo",
        "CreateImage",
        "ImageGeneration",
        # KIE specific patterns
        "Flux2Pro",
        "Flux2Flex",
        "Seedream45TextToImage",
        "ZImage",
        "NanoBanana",
        "FluxKontext",
        "GrokImagineTextToImage",
        "QwenTextToImage",
        "Imagen4",
        "Kli ngTextToVideo",
        "KlingImageToVideo",
        "KlingAIAvatar",
        "Seedance",
        "Hailuo",
        "Kling25Turbo",
        "Sora2",
        "WanMultiShot",
        "Wan26TextToVideo",
        "Wan26ImageToVideo",
        "Infinitalk",
        "Veo31",
        "RunwayGen3AlphaTextToVideo",
        "RunwayGen3AlphaImageToVideo",
        "RunwayAleph",
        "ElevenLabsTextToSpeech",
    ]
    
    return any(keyword in node_name for keyword in generative_keywords)


def node_is_load_or_save(node_name):
    """Check if a node is a Load* or Save* node."""
    return node_name.startswith("Load") or node_name.startswith("Save")


@pytest.mark.parametrize("module_name,node_name,node_class", get_all_node_classes())
def test_auto_save_asset_attribute(module_name, node_name, node_class):
    """
    Test that only generative nodes have _auto_save_asset = True.

    This test checks that:
    1. Generative nodes (TextToImage, TextToSpeech, etc.) have _auto_save_asset = True
    2. Editing/transformation nodes do NOT have _auto_save_asset = True
    3. Load* and Save* nodes do NOT have _auto_save_asset = True
    4. Library nodes do NOT have _auto_save_asset = True
    """
    # Skip Load* and Save* nodes
    if node_is_load_or_save(node_name):
        return

    # Check if node has _auto_save_asset
    has_auto_save = getattr(node_class, "_auto_save_asset", False)
    
    # Check if it's a library node
    is_library = "nodes.lib." in module_name
    
    if is_library:
        # Library nodes should NOT have auto_save_asset
        assert (
            has_auto_save is False
        ), f"{module_name}.{node_name} is a library node and should NOT have _auto_save_asset = True"
        return
    
    # Check if it's a generative node
    is_generative = node_is_generative(node_name)
    
    if is_generative:
        # Generative nodes SHOULD have auto_save_asset
        assert (
            has_auto_save is True
        ), f"{module_name}.{node_name} is a generative node and should have _auto_save_asset = True"
    else:
        # Non-generative nodes should NOT have auto_save_asset
        # (unless they are exceptions we haven't categorized)
        pass

        if node_name in exceptions:
            return

        assert (
            has_auto_save is True
        ), f"{module_name}.{node_name} generates assets but doesn't have _auto_save_asset = True"


def test_auto_save_asset_sample_nodes():
    """
    Test a sample of known generative nodes to ensure they have _auto_save_asset = True.
    """
    from nodetool.nodes.nodetool.image import TextToImage, ImageToImage
    from nodetool.nodes.nodetool.audio import CreateSilence, TextToSpeech
    from nodetool.nodes.nodetool.video import TextToVideo

    # Test core generative nodes
    assert (
        getattr(TextToImage, "_auto_save_asset", False) is True
    ), "TextToImage should have _auto_save_asset = True"
    assert (
        getattr(ImageToImage, "_auto_save_asset", False) is True
    ), "ImageToImage should have _auto_save_asset = True"
    assert (
        getattr(CreateSilence, "_auto_save_asset", False) is True
    ), "CreateSilence should have _auto_save_asset = True"
    assert (
        getattr(TextToSpeech, "_auto_save_asset", False) is True
    ), "TextToSpeech should have _auto_save_asset = True"
    assert (
        getattr(TextToVideo, "_auto_save_asset", False) is True
    ), "TextToVideo should have _auto_save_asset = True"


def test_auto_save_asset_not_on_editing_nodes():
    """
    Test that editing/transformation nodes do NOT have _auto_save_asset = True.
    """
    from nodetool.nodes.nodetool.image import Scale, Resize, Crop
    from nodetool.nodes.nodetool.audio import Normalize, OverlayAudio
    from nodetool.nodes.nodetool.video import Trim, ColorBalance

    # Test that editing nodes do NOT have auto_save_asset
    assert (
        getattr(Scale, "_auto_save_asset", False) is False
    ), "Scale (editing node) should NOT have _auto_save_asset = True"
    assert (
        getattr(Resize, "_auto_save_asset", False) is False
    ), "Resize (editing node) should NOT have _auto_save_asset = True"
    assert (
        getattr(Crop, "_auto_save_asset", False) is False
    ), "Crop (editing node) should NOT have _auto_save_asset = True"
    assert (
        getattr(Normalize, "_auto_save_asset", False) is False
    ), "Normalize (editing node) should NOT have _auto_save_asset = True"
    assert (
        getattr(OverlayAudio, "_auto_save_asset", False) is False
    ), "OverlayAudio (editing node) should NOT have _auto_save_asset = True"
    assert (
        getattr(Trim, "_auto_save_asset", False) is False
    ), "Trim (editing node) should NOT have _auto_save_asset = True"
    assert (
        getattr(ColorBalance, "_auto_save_asset", False) is False
    ), "ColorBalance (editing node) should NOT have _auto_save_asset = True"


def test_auto_save_asset_provider_nodes():
    """
    Test provider nodes to ensure generative ones have _auto_save_asset = True.
    """
    from nodetool.nodes.openai.image import CreateImage
    from nodetool.nodes.openai.audio import TextToSpeech as OpenAITextToSpeech
    from nodetool.nodes.gemini.image import ImageGeneration
    from nodetool.nodes.gemini.video import TextToVideo as GeminiTextToVideo

    # Test OpenAI nodes
    assert (
        getattr(CreateImage, "_auto_save_asset", False) is True
    ), "OpenAI CreateImage should have _auto_save_asset = True"
    assert (
        getattr(OpenAITextToSpeech, "_auto_save_asset", False) is True
    ), "OpenAI TextToSpeech should have _auto_save_asset = True"

    # Test Gemini nodes
    assert (
        getattr(ImageGeneration, "_auto_save_asset", False) is True
    ), "Gemini ImageGeneration should have _auto_save_asset = True"
    assert (
        getattr(GeminiTextToVideo, "_auto_save_asset", False) is True
    ), "Gemini TextToVideo should have _auto_save_asset = True"


def test_auto_save_asset_not_on_library_nodes():
    """
    Test that library nodes do NOT have _auto_save_asset = True.
    """
    from nodetool.nodes.lib.pillow.enhance import AutoContrast, Sharpness
    from nodetool.nodes.lib.pillow.filter import Blur, Invert

    # Test that library nodes do NOT have auto_save_asset
    assert (
        getattr(AutoContrast, "_auto_save_asset", False) is False
    ), "AutoContrast (library node) should NOT have _auto_save_asset = True"
    assert (
        getattr(Sharpness, "_auto_save_asset", False) is False
    ), "Sharpness (library node) should NOT have _auto_save_asset = True"
    assert (
        getattr(Blur, "_auto_save_asset", False) is False
    ), "Blur (library node) should NOT have _auto_save_asset = True"
    assert (
        getattr(Invert, "_auto_save_asset", False) is False
    ), "Invert (library node) should NOT have _auto_save_asset = True"

