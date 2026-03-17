"""Kie.ai API nodes for AI generation.

This package provides nodes for integrating with Kie.ai's unified API platform,
which offers access to state-of-the-art AI models for:
- Image generation and editing (4O, Flux, Seedream, Z-Image, Nano Banana, Grok, Topaz)
- Video generation (Veo, Wan, Sora, Seedance, Hailuo, Kling)
- Music generation (Suno)

All nodes require a KIE_API_KEY secret to be configured.
"""

# Import submodules so `import nodetool.nodes.kie` eagerly registers all Kie node
# classes. Graph loading only knows the top-level namespace (`kie.*`), so this
# package init must populate the global node registry for reloads in fresh
# processes.
import nodetool.nodes.kie.audio  # noqa: F401
import nodetool.nodes.kie.dynamic_schema  # noqa: F401
import nodetool.nodes.kie.image  # noqa: F401
import nodetool.nodes.kie.video  # noqa: F401