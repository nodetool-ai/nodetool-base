from typing import Any

from nodetool.workflows.base_node import OutputNode
from nodetool.workflows.processing_context import ProcessingContext


class Output(OutputNode):
    """
    Generic output node for any type.
    output, result, sink, return

    Use cases:
    - Output any type of result from a workflow
    - Return strings, integers, floats, booleans, lists, dictionaries
    - Return media references (images, videos, audio, documents, dataframes)
    """

    pass
