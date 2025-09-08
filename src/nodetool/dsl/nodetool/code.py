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


import nodetool.nodes.nodetool.code


class ExecuteBash(GraphNode):
    """
    Executes Bash script with safety restrictions.
    bash, shell, code, execute

    IMPORTANT: Only enabled in non-production environments
    """

    BashImage: typing.ClassVar[type] = (
        nodetool.nodes.nodetool.code.ExecuteBash.BashImage
    )
    code: str | GraphNode | tuple[GraphNode, str] = Field(
        default="",
        description="Bash script to execute as-is. Dynamic inputs are provided as env vars. Stdout lines are emitted on 'stdout'; stderr lines on 'stderr'.",
    )
    image: nodetool.nodes.nodetool.code.ExecuteBash.BashImage = Field(
        default=nodetool.nodes.nodetool.code.ExecuteBash.BashImage.UBUNTU_22_04,
        description="Docker image to use for execution",
    )
    stdin: str | GraphNode | tuple[GraphNode, str] = Field(
        default="",
        description="String to write to process stdin before any streaming input. Use newlines to separate lines.",
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.code.ExecuteBash"


import nodetool.nodes.nodetool.code


class ExecuteCommand(GraphNode):
    """
    Executes a single shell command inside a Docker container.
    command, execute, shell, bash, sh

    IMPORTANT: Only enabled in non-production environments
    """

    CommandImage: typing.ClassVar[type] = (
        nodetool.nodes.nodetool.code.ExecuteCommand.CommandImage
    )
    command: str | GraphNode | tuple[GraphNode, str] = Field(
        default="",
        description="Single command to run via the selected shell. Stdout lines are emitted on 'stdout'; stderr lines on 'stderr'.",
    )
    image: nodetool.nodes.nodetool.code.ExecuteCommand.CommandImage = Field(
        default=nodetool.nodes.nodetool.code.ExecuteCommand.CommandImage.BASH_5_2,
        description="Docker image to use for execution",
    )
    stdin: str | GraphNode | tuple[GraphNode, str] = Field(
        default="",
        description="String to write to process stdin before any streaming input. Use newlines to separate lines.",
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.code.ExecuteCommand"


import nodetool.nodes.nodetool.code


class ExecuteJavaScript(GraphNode):
    """
    Executes JavaScript (Node.js) code with safety restrictions.
    javascript, nodejs, code, execute

    IMPORTANT: Only enabled in non-production environments
    """

    JavaScriptImage: typing.ClassVar[type] = (
        nodetool.nodes.nodetool.code.ExecuteJavaScript.JavaScriptImage
    )
    code: str | GraphNode | tuple[GraphNode, str] = Field(
        default="",
        description="JavaScript code to execute as-is under Node.js. Dynamic inputs are provided as env vars. Stdout lines are emitted on 'stdout'; stderr lines on 'stderr'.",
    )
    image: nodetool.nodes.nodetool.code.ExecuteJavaScript.JavaScriptImage = Field(
        default=nodetool.nodes.nodetool.code.ExecuteJavaScript.JavaScriptImage.NODE_22_ALPINE,
        description="Docker image to use for execution",
    )
    stdin: str | GraphNode | tuple[GraphNode, str] = Field(
        default="",
        description="String to write to process stdin before any streaming input. Use newlines to separate lines.",
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.code.ExecuteJavaScript"


import nodetool.nodes.nodetool.code


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

    PythonImage: typing.ClassVar[type] = (
        nodetool.nodes.nodetool.code.ExecutePython.PythonImage
    )
    code: str | GraphNode | tuple[GraphNode, str] = Field(
        default="",
        description="Python code to execute as-is. Dynamic inputs are provided as env vars. Stdout lines are emitted on 'stdout'; stderr lines on 'stderr'.",
    )
    image: nodetool.nodes.nodetool.code.ExecutePython.PythonImage = Field(
        default=nodetool.nodes.nodetool.code.ExecutePython.PythonImage.PYTHON_3_11_SLIM,
        description="Docker image to use for execution",
    )
    stdin: str | GraphNode | tuple[GraphNode, str] = Field(
        default="",
        description="String to write to process stdin before any streaming input. Use newlines to separate lines.",
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.code.ExecutePython"


import nodetool.nodes.nodetool.code


class ExecuteRuby(GraphNode):
    """
    Executes Ruby code with safety restrictions.
    ruby, code, execute

    IMPORTANT: Only enabled in non-production environments
    """

    RubyImage: typing.ClassVar[type] = (
        nodetool.nodes.nodetool.code.ExecuteRuby.RubyImage
    )
    code: str | GraphNode | tuple[GraphNode, str] = Field(
        default="",
        description="Ruby code to execute as-is. Dynamic inputs are provided as env vars. Stdout lines are emitted on 'stdout'; stderr lines on 'stderr'.",
    )
    image: nodetool.nodes.nodetool.code.ExecuteRuby.RubyImage = Field(
        default=nodetool.nodes.nodetool.code.ExecuteRuby.RubyImage.RUBY_3_3_ALPINE,
        description="Docker image to use for execution",
    )
    stdin: str | GraphNode | tuple[GraphNode, str] = Field(
        default="",
        description="String to write to process stdin before any streaming input. Use newlines to separate lines.",
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.code.ExecuteRuby"
