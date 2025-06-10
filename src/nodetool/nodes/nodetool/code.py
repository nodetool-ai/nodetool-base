import ast
from typing import Any
from pydantic import Field
from nodetool.common.environment import Environment
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext


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

    code: str = Field(
        default="",
        description="Python code to execute. Input variables are available as locals. Assign the desired output to the 'result' variable.",
    )

    inputs: dict[str, Any] = Field(
        default={},
        description="Input variables available to the code as locals.",
    )

    async def process(self, context: ProcessingContext) -> Any:
        if Environment.is_production():
            raise RuntimeError("Python code execution is disabled in production")

        if not self.code.strip():
            return None

        # Basic static analysis for dangerous operations
        tree = ast.parse(self.code)
        for node in ast.walk(tree):
            # Block imports
            if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                raise ValueError("Import statements are not allowed")

            # Block file operations
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ["open", "eval", "exec"]:
                        raise ValueError(f"Function {node.func.id}() is not allowed")

        # Create restricted globals
        restricted_globals = {
            "__builtins__": {
                # Math & numeric utilities
                "abs": abs,               # Absolute value
                "divmod": divmod,         # Clean int division and remainder
                "float": float,           # Convert to a decimal number
                "int": int,               # Convert to a whole number
                "max": max,               # Maximum of iterable
                "min": min,               # Minimum of iterable
                "pow": pow,               # Exponentiation
                "round": round,           # Round a number
                "sum": sum,               # Sum iterable of numbers

                # Iterable and transformation utilities
                "all": all,               # True if all elements are true
                "any": any,               # True if any element is true
                "enumerate": enumerate,   # Get index-value pairs from iterable
                "filter": filter,         # Functional programming
                "len": len,               # Length of iterable
                "map": map,               # Functional programming
                "range": range,           # Generate integer ranges
                "reversed": reversed,     # Reverse list or string
                "sorted": sorted,         # Sort iterable
                "zip": zip,               # Combine multiple iterables

                # Type checks and logic
                "bool": bool,             # Convert value to True or False
                "isinstance": isinstance, # Safe type check
                "type": type,             # Get type of object

                # Built-in data types
                "dict": dict,             # Create a dictionary
                "frozenset": frozenset,   # Create an immutable set
                "list": list,             # Create a list
                "set": set,               # Create a set
                "str": str,               # Convert to string
                "tuple": tuple,           # Create a tuple

                # Debugging & Introspection
                "hash": hash,             # Get hash value of an object
                "repr": repr,             # Get string representation of an object

                # Local variable access
                "locals": lambda: self.inputs,  # Use values from Dictionary input
            }
        }


        # Execute in restricted environment
        try:
            exec(self.code, restricted_globals, self.inputs)
            return self.inputs.get("result", None)
        except Exception as e:
            raise RuntimeError(f"Error executing Python code: {str(e)}")


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

        # Basic static analysis
        tree = ast.parse(self.expression, mode="eval")
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                raise ValueError("Imports are not allowed in expressions")
            if isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Name) or node.func.id not in {
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
                }:
                    raise ValueError(
                        "Only safe built-in function calls are allowed in expressions"
                    )

        # Create restricted environment
        restricted_globals = {
            "__builtins__": {
                "abs": abs,               # Absolute value
                "all": all,               # True if all elements are true
                "any": any,               # True if any element is true
                "bool": bool,             # Convert value to True or False
                "float": float,           # Convert to a decimal number
                "int": int,               # Convert to a whole number
                "len": len,               # Length of an object
                "max": max,               # Maximum of an iterable
                "min": min,               # Minimum of an iterable
                "round": round,           # Round a number
                "str": str,               # Convert to string
                "sum": sum,               # Sum an iterable
            }
        }

        try:
            return eval(self.expression, restricted_globals, self.variables)
        except Exception as e:
            raise RuntimeError(f"Error evaluating expression: {str(e)}")
