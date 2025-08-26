import ast
import asyncio
import base64
import json
import textwrap
from typing import Any
from enum import Enum
from nodetool.workflows.types import NodeUpdate
from pydantic import Field
from nodetool.common.environment import Environment
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from .python_runner import PythonDockerRunner


class ExecutePython(BaseNode):
    """
    Executes Python code with safety restrictions.
    python, code, execute

    Use cases:
    - Run custom data transformations
    - Prototype node functionality
    - Debug and testing workflows

    IMPORTANT: Only enabled in non-production environments
    """

    _is_dynamic = True
    _supports_dynamic_outputs = True

    class PythonImage(Enum):
        PYTHON_3_12_SLIM = "python:3.12-slim"
        PYTHON_3_11_SLIM = "python:3.11-slim"
        PYTHON_3_10_SLIM = "python:3.10-slim"
        JUPYTER_DATASCIENCE_NOTEBOOK = "jupyter/datascience-notebook:latest"
        JUPYTER_SCIPY_NOTEBOOK = "jupyter/scipy-notebook:latest"
        PYTORCH_CPU = "pytorch/pytorch:latest"

    code: str = Field(
        default="",
        description="Python code to execute. Dynamic inputs are available as locals. Send output using the `yield` keyword, e.g. `yield 'output', 'Hello, world!'` or `yield 'dynmamic_output', {'key': 'value'}`",
    )

    image: PythonImage = Field(
        default=PythonImage.PYTHON_3_11_SLIM,
        description="Docker image to use for execution",
    )

    @classmethod
    def return_type(cls):
        return {"error": str, "output": Any}

    async def gen_process(self, context: ProcessingContext):
        if Environment.is_production():
            raise RuntimeError("Python code execution is disabled in production")

        if not self.code.strip():
            raise RuntimeError("Code is required")

        runner = PythonDockerRunner(image=self.image.value)
        async for slot, value in runner.stream(
            user_code=self.code,
            env_locals=self._dynamic_properties,
            context=context,
            node=self,
            allow_dynamic_outputs=self.supports_dynamic_outputs(),
        ):
            if value is not None:
                yield slot, value


class EvaluateExpression(BaseNode):
    """
    Evaluates a Python expression with safety restrictions.
    python, expression, evaluate

    Use cases:
    - Calculate values dynamically
    - Transform data with simple expressions
    - Quick data validation

    IMPORTANT: Only enabled in non-production environments
    """

    expression: str = Field(
        default="",
        description="Python expression to evaluate. Variables are available as locals.",
    )

    variables: dict[str, Any] = Field(
        default={}, description="Variables available to the expression"
    )

    async def process(self, context: ProcessingContext) -> Any:
        if Environment.is_production():
            raise RuntimeError("Python expression evaluation is disabled in production")

        if not self.expression.strip():
            return None

        # Basic static analysis with AST whitelisting
        tree = ast.parse(self.expression, mode="eval")

        allowed_call_names = {
            "abs",
            "all",
            "any",
            "bool",
            "float",
            "int",
            "len",
            "max",
            "min",
            "round",
            "str",
            "sum",
        }

        allowed_unary_ops = (ast.UAdd, ast.USub, ast.Not, ast.Invert)
        allowed_bin_ops = (
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.Mod,
            ast.Pow,
            ast.FloorDiv,
            ast.MatMult,
        )
        allowed_bool_ops = (ast.And, ast.Or)
        allowed_cmp_ops = (
            ast.Eq,
            ast.NotEq,
            ast.Lt,
            ast.LtE,
            ast.Gt,
            ast.GtE,
            ast.Is,
            ast.IsNot,
            ast.In,
            ast.NotIn,
        )

        def _validate(node: ast.AST) -> None:
            if isinstance(node, ast.Expression):
                _validate(node.body)
                return
            if isinstance(node, ast.Constant):
                return
            if isinstance(node, ast.Name):
                # Variables or allowed globals by name (usage without calling is fine)
                return
            if isinstance(node, ast.Tuple):
                for elt in node.elts:
                    _validate(elt)
                return
            if isinstance(node, ast.List):
                for elt in node.elts:
                    _validate(elt)
                return
            if isinstance(node, ast.Dict):
                for k, v in zip(node.keys, node.values):
                    if k is not None:
                        _validate(k)
                    _validate(v)
                return
            if isinstance(node, ast.Set):
                for elt in node.elts:
                    _validate(elt)
                return
            if isinstance(node, ast.UnaryOp):
                if not isinstance(node.op, allowed_unary_ops):
                    raise ValueError("Operator not allowed")
                _validate(node.operand)
                return
            if isinstance(node, ast.BinOp):
                if not isinstance(node.op, allowed_bin_ops):
                    raise ValueError("Operator not allowed")
                _validate(node.left)
                _validate(node.right)
                return
            if isinstance(node, ast.BoolOp):
                if not isinstance(node.op, allowed_bool_ops):
                    raise ValueError("Boolean operator not allowed")
                for v in node.values:
                    _validate(v)
                return
            if isinstance(node, ast.Compare):
                for op in node.ops:
                    if not isinstance(op, allowed_cmp_ops):
                        raise ValueError("Comparison operator not allowed")
                _validate(node.left)
                for cmp in node.comparators:
                    _validate(cmp)
                return
            if isinstance(node, ast.IfExp):
                _validate(node.test)
                _validate(node.body)
                _validate(node.orelse)
                return
            if isinstance(node, ast.Subscript):
                _validate(node.value)
                _validate(node.slice)
                return
            if isinstance(node, ast.Slice):
                if node.lower:
                    _validate(node.lower)
                if node.upper:
                    _validate(node.upper)
                if node.step:
                    _validate(node.step)
                return
            if isinstance(node, ast.Call):
                # Only allow calling whitelisted simple names
                if (
                    not isinstance(node.func, ast.Name)
                    or node.func.id not in allowed_call_names
                ):
                    raise ValueError("Only calls to whitelisted functions are allowed")
                for arg in node.args:
                    _validate(arg)
                for kw in node.keywords:
                    if kw.value is not None:
                        _validate(kw.value)
                return
            # Disallow everything else: attributes, lambdas, comprehensions, assigns, etc.
            raise ValueError("Unsupported expression construct")

        _validate(tree)

        # Create restricted environment
        restricted_globals = {
            "__builtins__": {
                "abs": abs,
                "all": all,
                "any": any,
                "bool": bool,
                "float": float,
                "int": int,
                "len": len,
                "max": max,
                "min": min,
                "round": round,
                "str": str,
                "sum": sum,
            }
        }

        try:
            return eval(self.expression, restricted_globals, self.variables)
        except Exception as e:
            raise RuntimeError(f"Error evaluating expression: {str(e)}")
