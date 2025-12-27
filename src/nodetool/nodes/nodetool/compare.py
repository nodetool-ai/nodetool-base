"""
Comparison nodes for visual data comparison.
"""

from typing import Any
from pydantic import Field

from nodetool.metadata.types import ImageRef
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import PreviewUpdate


class CompareImages(BaseNode):
    """
    Compare two images side-by-side with an interactive slider.
    image, compare, comparison, diff, before, after, slider

    Use this node to visually compare:
    - Before/after processing results
    - Different model outputs
    - Original vs edited images
    - A/B testing of image variations
    """

    image_a: ImageRef = Field(
        default=ImageRef(),
        description="First image (displayed on left/top)",
    )
    image_b: ImageRef = Field(
        default=ImageRef(),
        description="Second image (displayed on right/bottom)",
    )
    label_a: str = Field(
        default="A",
        description="Label for the first image",
    )
    label_b: str = Field(
        default="B",
        description="Label for the second image",
    )

    # Hide the outputs panel - this is a preview-only node
    _visible: bool = False

    @classmethod
    def get_title(cls) -> str:
        return "Compare Images"

    @classmethod
    def is_cacheable(cls) -> bool:
        return False

    async def process(self, context: ProcessingContext) -> None:
        """Process the comparison and send preview update."""
        if self.image_a.is_empty():
            return
        if self.image_b.is_empty():
            return

        # Normalize the image refs to ensure they have URIs
        image_a = await context.normalize_output_value(self.image_a)
        image_b = await context.normalize_output_value(self.image_b)

        # Send the comparison data as a preview update
        context.post_message(
            PreviewUpdate(
                node_id=self.id,
                value={
                    "type": "image_comparison",
                    "image_a": image_a,
                    "image_b": image_b,
                    "label_a": self.label_a,
                    "label_b": self.label_b,
                },
            )
        )

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["image_a", "image_b"]
