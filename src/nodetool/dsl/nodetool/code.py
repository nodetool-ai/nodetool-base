from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode


class EvaluateExpression(GraphNode):
    """
    Evaluates a Python expression with safety restrictions.
    python, expression, evaluate

    Use cases:
    - Calculate values dynamically
    - Transform data with simple expressions
    - Quick data validation

    IMPORTANT: Only enabled in non-production environments
    """

    expression: str | GraphNode | tuple[GraphNode, str] = Field(
        default="",
        description="Python expression to evaluate. Variables are available as locals.",
    )
    variables: dict[str, Any] | GraphNode | tuple[GraphNode, str] = Field(
        default={}, description="Variables available to the expression"
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.code.EvaluateExpression"


class ExecutePython(GraphNode):
    """
    Executes Python code with safety restrictions.
    python, code, execute

    Use cases:
    - Run custom data transformations
    - Prototype node functionality
    - Debug and testing workflows

    IMPORTANT: Only enabled in non-production environments
    """

    code: str | GraphNode | tuple[GraphNode, str] = Field(
        default="",
        description="Python code to execute. Dynamic properties and inputs are available as locals. Assign the desired output to the 'result' variable.",
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.code.ExecutePython"
