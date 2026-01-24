"""
Test that _auto_save_asset attribute is properly set on nodes that generate assets.
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


def node_generates_asset(node_class):
    """Check if a node generates an asset based on its return type."""
    # Get the process or gen_process method
    if hasattr(node_class, "process"):
        method = node_class.process
    elif hasattr(node_class, "gen_process"):
        method = node_class.gen_process
    else:
        return False

    # Get the return type annotation
    sig = inspect.signature(method)
    return_type = sig.return_annotation

    if return_type is inspect.Signature.empty:
        return False

    # Convert to string for easier checking
    return_type_str = str(return_type)

    # Check if it returns an asset reference type
    return any(
        asset_type in return_type_str
        for asset_type in ["ImageRef", "AudioRef", "VideoRef"]
    )


def node_is_load_or_save(node_name):
    """Check if a node is a Load* or Save* node."""
    return node_name.startswith("Load") or node_name.startswith("Save")


@pytest.mark.parametrize("module_name,node_name,node_class", get_all_node_classes())
def test_auto_save_asset_attribute(module_name, node_name, node_class):
    """
    Test that nodes which generate assets have _auto_save_asset = True.

    This test checks that:
    1. Nodes that return ImageRef, AudioRef, or VideoRef have _auto_save_asset = True
    2. Load* and Save* nodes are excluded (they don't generate new assets)
    """
    # Skip Load* and Save* nodes
    if node_is_load_or_save(node_name):
        return

    # Check if node generates an asset
    if node_generates_asset(node_class):
        # Check if _auto_save_asset is set to True
        has_auto_save = getattr(node_class, "_auto_save_asset", False)

        # Allow some exceptions for specific node types
        exceptions = [
            "BatchToList",  # Just reorganizes references
            "ImagesToList",  # Just reorganizes references
            "AudioToNumpy",  # Converts to numpy array, not an audio asset
            "ConvertToArray",  # Converts to numpy array, not an audio asset
            "GetMetadata",  # Returns metadata, not a new asset
            "FrameIterator",  # Iterates over frames, doesn't generate new ones
            "Translate",  # Returns text, not audio
            "Transcribe",  # Returns text, not audio
        ]

        if node_name in exceptions:
            return

        assert (
            has_auto_save is True
        ), f"{module_name}.{node_name} generates assets but doesn't have _auto_save_asset = True"


def test_auto_save_asset_sample_nodes():
    """
    Test a sample of known asset-generating nodes to ensure they have _auto_save_asset = True.
    """
    from nodetool.nodes.nodetool.image import TextToImage, ImageToImage, Scale
    from nodetool.nodes.nodetool.audio import Normalize, CreateSilence
    from nodetool.nodes.nodetool.video import TextToVideo

    # Test core nodes
    assert (
        getattr(TextToImage, "_auto_save_asset", False) is True
    ), "TextToImage should have _auto_save_asset = True"
    assert (
        getattr(ImageToImage, "_auto_save_asset", False) is True
    ), "ImageToImage should have _auto_save_asset = True"
    assert (
        getattr(Scale, "_auto_save_asset", False) is True
    ), "Scale should have _auto_save_asset = True"
    assert (
        getattr(Normalize, "_auto_save_asset", False) is True
    ), "Normalize should have _auto_save_asset = True"
    assert (
        getattr(CreateSilence, "_auto_save_asset", False) is True
    ), "CreateSilence should have _auto_save_asset = True"
    assert (
        getattr(TextToVideo, "_auto_save_asset", False) is True
    ), "TextToVideo should have _auto_save_asset = True"


def test_auto_save_asset_provider_nodes():
    """
    Test provider nodes to ensure they have _auto_save_asset = True.
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


def test_auto_save_asset_pillow_nodes():
    """
    Test Pillow library nodes to ensure they have _auto_save_asset = True.
    """
    from nodetool.nodes.lib.pillow.enhance import AutoContrast, Sharpness
    from nodetool.nodes.lib.pillow.filter import Blur, Invert
    from nodetool.nodes.lib.pillow.color_grading import Exposure
    from nodetool.nodes.lib.pillow.draw import Background

    # Test enhance nodes
    assert (
        getattr(AutoContrast, "_auto_save_asset", False) is True
    ), "AutoContrast should have _auto_save_asset = True"
    assert (
        getattr(Sharpness, "_auto_save_asset", False) is True
    ), "Sharpness should have _auto_save_asset = True"

    # Test filter nodes
    assert (
        getattr(Blur, "_auto_save_asset", False) is True
    ), "Blur should have _auto_save_asset = True"
    assert (
        getattr(Invert, "_auto_save_asset", False) is True
    ), "Invert should have _auto_save_asset = True"

    # Test color grading nodes
    assert (
        getattr(Exposure, "_auto_save_asset", False) is True
    ), "Exposure should have _auto_save_asset = True"

    # Test draw nodes
    assert (
        getattr(Background, "_auto_save_asset", False) is True
    ), "Background should have _auto_save_asset = True"
