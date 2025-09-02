import ast
from typing import Any
from enum import Enum
from pydantic import Field
from nodetool.common.environment import Environment
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import LogUpdate
from nodetool.code_runners.python_runner import PythonDockerRunner
from nodetool.code_runners.javascript_runner import JavaScriptDockerRunner
from nodetool.code_runners.bash_runner import BashDockerRunner
from nodetool.code_runners.ruby_runner import RubyDockerRunner
from nodetool.code_runners.command_runner import CommandDockerRunner
from nodetool.workflows.io import NodeInputs, NodeOutputs
from nodetool.code_runners.runtime_base import StreamRunnerBase


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
    _runner: StreamRunnerBase | None = None

    class PythonImage(Enum):
        PYTHON_3_11_SLIM = "python:3.11-slim"
        JUPYTER_SCIPY_NOTEBOOK = "jupyter/scipy-notebook:latest"

    code: str = Field(
        default="",
        description=(
            "Python code to execute as-is. Dynamic inputs are provided as env vars. "
            "Stdout lines are emitted on 'stdout'; stderr lines on 'stderr'."
        ),
    )

    image: PythonImage = Field(
        default=PythonImage.PYTHON_3_11_SLIM,
        description="Docker image to use for execution",
    )

    stdin: str = Field(
        default="",
        description=(
            "String to write to process stdin before any streaming input. "
            "Use newlines to separate lines."
        ),
    )

    @classmethod
    def is_streaming_input(cls):
        return True

    @classmethod
    def return_type(cls):
        return {"stdout": str, "stderr": str}

    @classmethod
    def is_streaming_output(cls):
        return True

    async def run(self, context: ProcessingContext, inputs: NodeInputs, outputs: NodeOutputs) -> None:  # type: ignore[override]
        if not self.code.strip():
            raise RuntimeError("Code is required")

        async def create_stdin_stream():
            if self.stdin:
                yield self.stdin
            async for value in inputs.stream("stdin"):
                yield str(value) if value is not None else ""

        # Consider stdin "connected" if there are buffered items or an open upstream
        use_stdin = (
            bool(self.stdin)
            or inputs.has_buffered("stdin")
            or inputs.has_stream("stdin")
        )
        stdin_stream = create_stdin_stream() if use_stdin else None

        runner = PythonDockerRunner(image=self.image.value)
        self._runner = runner
        async for slot, value in runner.stream(
            user_code=self.code,
            env_locals=self._dynamic_properties,
            context=context,
            node=self,
            allow_dynamic_outputs=self.supports_dynamic_outputs(),
            stdin_stream=stdin_stream,
        ):
            if value is None:
                continue
            text_value = value if isinstance(value, str) else str(value)
            # Send log updates for stdout/stderr
            if slot == "stdout":
                context.post_message(
                    LogUpdate(
                        node_id=self.id,
                        node_name=self.get_title(),
                        content=str(value).rstrip("\n"),
                        severity="info",
                    )
                )
            elif slot == "stderr":
                context.post_message(
                    LogUpdate(
                        node_id=self.id,
                        node_name=self.get_title(),
                        content=str(value).rstrip("\n"),
                        severity="error",
                    )
                )
            await outputs.emit(slot, text_value)

    async def finalize(self, context: ProcessingContext):  # type: ignore[override]
        """Stop any running Docker container for this node.

        Args:
            context: Processing context (unused).

        Returns:
            None
        """
        try:
            if self._runner:
                self._runner.stop()
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug(f"ExecutePython finalize: {e}")


class ExecuteJavaScript(BaseNode):
    """
    Executes JavaScript (Node.js) code with safety restrictions.
    javascript, nodejs, code, execute

    IMPORTANT: Only enabled in non-production environments
    """

    _is_dynamic = True
    _supports_dynamic_outputs = True
    _runner: StreamRunnerBase | None = None

    class JavaScriptImage(Enum):
        NODE_22_ALPINE = "node:22-alpine"

    code: str = Field(
        default="",
        description=(
            "JavaScript code to execute as-is under Node.js. Dynamic inputs are provided as env vars. "
            "Stdout lines are emitted on 'stdout'; stderr lines on 'stderr'."
        ),
    )

    image: JavaScriptImage = Field(
        default=JavaScriptImage.NODE_22_ALPINE,
        description="Docker image to use for execution",
    )

    stdin: str = Field(
        default="",
        description=(
            "String to write to process stdin before any streaming input. "
            "Use newlines to separate lines."
        ),
    )

    @classmethod
    def is_streaming_input(cls):
        return True

    @classmethod
    def return_type(cls):
        return {"stdout": str, "stderr": str}

    # async def gen_process(self, context: ProcessingContext):
    #     if not self.code.strip():
    #         raise RuntimeError("Code is required")

    #     async def create_stdin_stream():
    #         if self.stdin:
    #             yield self.stdin
    #         async for handle, value in self.iter_any_input():
    #             yield str(value) if value is not None else ""

    #     if self.stdin or self.has_streaming_inputs():
    #         stdin_stream = create_stdin_stream()
    #     else:
    #         stdin_stream = None

    #     runner = JavaScriptDockerRunner(image=self.image.value)
    #     async for slot, value in runner.stream(
    #         user_code=self.code,
    #         env_locals=self._dynamic_properties,
    #         context=context,
    #         node=self,
    #         allow_dynamic_outputs=self.supports_dynamic_outputs(),
    #         stdin_stream=stdin_stream,
    #     ):
    #         if value is None:
    #             continue
    #         text_value = value if isinstance(value, str) else str(value)
    #         yield slot, text_value

    async def run(self, context: ProcessingContext, inputs: NodeInputs, outputs: NodeOutputs) -> None:  # type: ignore[override]
        if not self.code.strip():
            raise RuntimeError("Code is required")

        async def create_stdin_stream():
            if self.stdin:
                yield self.stdin
            async for value in inputs.stream("stdin"):
                yield str(value) if value is not None else ""

        use_stdin = (
            bool(self.stdin)
            or inputs.has_buffered("stdin")
            or inputs.has_stream("stdin")
        )
        stdin_stream = create_stdin_stream() if use_stdin else None

        runner = JavaScriptDockerRunner(image=self.image.value)
        self._runner = runner
        async for slot, value in runner.stream(
            user_code=self.code,
            env_locals=self._dynamic_properties,
            context=context,
            node=self,
            allow_dynamic_outputs=self.supports_dynamic_outputs(),
            stdin_stream=stdin_stream,
        ):
            if value is None:
                continue
            text_value = value if isinstance(value, str) else str(value)
            # Send log updates for stdout/stderr
            if slot == "stdout":
                context.post_message(
                    LogUpdate(
                        node_id=self.id,
                        node_name=self.get_title(),
                        content=str(value).rstrip("\n"),
                        severity="info",
                    )
                )
            elif slot == "stderr":
                context.post_message(
                    LogUpdate(
                        node_id=self.id,
                        node_name=self.get_title(),
                        content=str(value).rstrip("\n"),
                        severity="error",
                    )
                )
            await outputs.emit(slot, text_value)

    async def finalize(self, context: ProcessingContext):  # type: ignore[override]
        """Stop any running Docker container for this node.

        Args:
            context: Processing context (unused).

        Returns:
            None
        """
        try:
            if self._runner:
                self._runner.stop()
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug(f"ExecuteJavaScript finalize: {e}")


class ExecuteBash(BaseNode):
    """
    Executes Bash script with safety restrictions.
    bash, shell, code, execute

    IMPORTANT: Only enabled in non-production environments
    """

    _is_dynamic = True
    _supports_dynamic_outputs = True
    _runner: StreamRunnerBase | None = None

    class BashImage(Enum):
        BASH_5_2 = "bash:5.2"
        DEBIAN_12 = "debian:12"
        UBUNTU_22_04 = "ubuntu:22.04"
        UBUNTU_24_04 = "ubuntu:24.04"
        JUPYTER_SCIPY_NOTEBOOK = "jupyter/scipy-notebook:latest"

    code: str = Field(
        default="",
        description=(
            "Bash script to execute as-is. Dynamic inputs are provided as env vars. "
            "Stdout lines are emitted on 'stdout'; stderr lines on 'stderr'."
        ),
    )

    image: BashImage = Field(
        default=BashImage.UBUNTU_22_04,
        description="Docker image to use for execution",
    )

    stdin: str = Field(
        default="",
        description=(
            "String to write to process stdin before any streaming input. "
            "Use newlines to separate lines."
        ),
    )

    @classmethod
    def is_streaming_input(cls):
        return True

    @classmethod
    def is_streaming_output(cls):
        return True

    @classmethod
    def return_type(cls):
        return {"stdout": str, "stderr": str}

    # async def gen_process(self, context: ProcessingContext):
    #     if not self.code.strip():
    #         raise RuntimeError("Code is required")

    #     async def create_stdin_stream():
    #         if self.stdin:
    #             yield self.stdin
    #         async for handle, value in self.iter_any_input():
    #             yield str(value) if value is not None else ""

    #     if self.stdin or self.has_streaming_inputs():
    #         stdin_stream = create_stdin_stream()
    #     else:
    #         stdin_stream = None

    #     runner = BashDockerRunner(image=self.image.value)
    #     async for slot, value in runner.stream(
    #         user_code=self.code,
    #         env_locals=self._dynamic_properties,
    #         context=context,
    #         node=self,
    #         allow_dynamic_outputs=self.supports_dynamic_outputs(),
    #         stdin_stream=stdin_stream,
    #     ):
    #         if value is None:
    #             continue
    #         text_value = value if isinstance(value, str) else str(value)
    #         yield slot, text_value

    async def run(self, context: ProcessingContext, inputs: NodeInputs, outputs: NodeOutputs) -> None:  # type: ignore[override]
        if not self.code.strip():
            raise RuntimeError("Code is required")

        async def create_stdin_stream():
            if self.stdin:
                yield self.stdin
            async for value in inputs.stream("stdin"):
                yield str(value) if value is not None else ""

        use_stdin = (
            bool(self.stdin)
            or inputs.has_buffered("stdin")
            or inputs.has_stream("stdin")
        )
        stdin_stream = create_stdin_stream() if use_stdin else None

        runner = BashDockerRunner(image=self.image.value)
        self._runner = runner
        async for slot, value in runner.stream(
            user_code=self.code,
            env_locals=self._dynamic_properties,
            context=context,
            node=self,
            allow_dynamic_outputs=self.supports_dynamic_outputs(),
            stdin_stream=stdin_stream,
        ):
            if value is None:
                continue
            text_value = value if isinstance(value, str) else str(value)
            # Send log updates for stdout/stderr
            print(f"slot: {slot}, value: {value}")
            if slot == "stdout":
                context.post_message(
                    LogUpdate(
                        node_id=self.id,
                        node_name=self.get_title(),
                        content=str(value).rstrip("\n"),
                        severity="info",
                    )
                )
            elif slot == "stderr":
                context.post_message(
                    LogUpdate(
                        node_id=self.id,
                        node_name=self.get_title(),
                        content=str(value).rstrip("\n"),
                        severity="error",
                    )
                )
            await outputs.emit(slot, text_value)

    async def finalize(self, context: ProcessingContext):  # type: ignore[override]
        """Stop any running Docker container for this node.

        Args:
            context: Processing context (unused).

        Returns:
            None
        """
        try:
            if self._runner:
                self._runner.stop()
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug(f"ExecuteBash finalize: {e}")


class ExecuteRuby(BaseNode):
    """
    Executes Ruby code with safety restrictions.
    ruby, code, execute

    IMPORTANT: Only enabled in non-production environments
    """

    _is_dynamic = True
    _supports_dynamic_outputs = True
    _runner: StreamRunnerBase | None = None

    class RubyImage(Enum):
        RUBY_3_3_ALPINE = "ruby:3.3-alpine"

    code: str = Field(
        default="",
        description=(
            "Ruby code to execute as-is. Dynamic inputs are provided as env vars. "
            "Stdout lines are emitted on 'stdout'; stderr lines on 'stderr'."
        ),
    )

    image: RubyImage = Field(
        default=RubyImage.RUBY_3_3_ALPINE,
        description="Docker image to use for execution",
    )

    stdin: str = Field(
        default="",
        description=(
            "String to write to process stdin before any streaming input. "
            "Use newlines to separate lines."
        ),
    )

    @classmethod
    def is_streaming_input(cls):
        return True

    @classmethod
    def is_streaming_output(cls):
        return True

    @classmethod
    def return_type(cls):
        return {"stdout": str, "stderr": str}

    # async def gen_process(self, context: ProcessingContext):
    #     if not self.code.strip():
    #         raise RuntimeError("Code is required")

    #     async def create_stdin_stream():
    #         if self.stdin:
    #             yield self.stdin
    #         async for handle, value in self.iter_any_input():
    #             yield str(value) if value is not None else ""

    #     if self.stdin or self.has_streaming_inputs():
    #         stdin_stream = create_stdin_stream()
    #     else:
    #         stdin_stream = None

    #     runner = RubyDockerRunner(image=self.image.value)
    #     async for slot, value in runner.stream(
    #         user_code=self.code,
    #         env_locals=self._dynamic_properties,
    #         context=context,
    #         node=self,
    #         allow_dynamic_outputs=self.supports_dynamic_outputs(),
    #         stdin_stream=stdin_stream,
    #     ):
    #         if value is None:
    #             continue
    #         text_value = value if isinstance(value, str) else str(value)
    #         yield slot, text_value

    async def run(self, context: ProcessingContext, inputs: NodeInputs, outputs: NodeOutputs) -> None:  # type: ignore[override]
        if not self.code.strip():
            raise RuntimeError("Code is required")

        async def create_stdin_stream():
            if self.stdin:
                yield self.stdin
            async for value in inputs.stream("stdin"):
                yield str(value) if value is not None else ""

        use_stdin = (
            bool(self.stdin)
            or inputs.has_buffered("stdin")
            or inputs.has_stream("stdin")
        )
        stdin_stream = create_stdin_stream() if use_stdin else None

        runner = RubyDockerRunner(image=self.image.value)
        self._runner = runner
        async for slot, value in runner.stream(
            user_code=self.code,
            env_locals=self._dynamic_properties,
            context=context,
            node=self,
            allow_dynamic_outputs=self.supports_dynamic_outputs(),
            stdin_stream=stdin_stream,
        ):
            if value is None:
                continue
            text_value = value if isinstance(value, str) else str(value)
            # Send log updates for stdout/stderr
            if slot == "stdout":
                context.post_message(
                    LogUpdate(
                        node_id=self.id,
                        node_name=self.get_title(),
                        content=str(value).rstrip("\n"),
                        severity="info",
                    )
                )
            elif slot == "stderr":
                context.post_message(
                    LogUpdate(
                        node_id=self.id,
                        node_name=self.get_title(),
                        content=str(value).rstrip("\n"),
                        severity="error",
                    )
                )
            await outputs.emit(slot, text_value)

    async def finalize(self, context: ProcessingContext):  # type: ignore[override]
        """Stop any running Docker container for this node.

        Args:
            context: Processing context (unused).

        Returns:
            None
        """
        try:
            if self._runner:
                self._runner.stop()
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug(f"ExecuteRuby finalize: {e}")


class ExecuteCommand(BaseNode):
    """
    Executes a single shell command inside a Docker container.
    command, execute, shell, bash, sh

    IMPORTANT: Only enabled in non-production environments
    """

    _is_dynamic = True
    _supports_dynamic_outputs = True
    _runner: StreamRunnerBase | None = None

    class CommandImage(Enum):
        BASH_5_2 = "bash:5.2"
        ALPINE_3 = "alpine:3"
        UBUNTU_22_04 = "ubuntu:22.04"
        UBUNTU_24_04 = "ubuntu:24.04"

    command: str = Field(
        default="",
        description=(
            "Single command to run via the selected shell. "
            "Stdout lines are emitted on 'stdout'; stderr lines on 'stderr'."
        ),
    )

    image: CommandImage = Field(
        default=CommandImage.BASH_5_2,
        description="Docker image to use for execution",
    )

    stdin: str = Field(
        default="",
        description=(
            "String to write to process stdin before any streaming input. "
            "Use newlines to separate lines."
        ),
    )

    @classmethod
    def is_streaming_input(cls):
        return True

    @classmethod
    def is_streaming_output(cls):
        return True

    @classmethod
    def return_type(cls):
        return {"stdout": str, "stderr": str}

    # async def gen_process(self, context: ProcessingContext):
    #     if not self.command.strip():
    #         raise RuntimeError("Command is required")

    #     async def create_stdin_stream():
    #         if self.stdin:
    #             yield self.stdin
    #         async for handle, value in self.iter_any_input():
    #             yield str(value) if value is not None else ""

    #     if self.stdin or self.has_streaming_inputs():
    #         stdin_stream = create_stdin_stream()
    #     else:
    #         stdin_stream = None

    #     runner = CommandDockerRunner(image=self.image.value)
    #     async for slot, value in runner.stream(
    #         user_code=self.command,
    #         env_locals=self._dynamic_properties,
    #         context=context,
    #         node=self,
    #         allow_dynamic_outputs=self.supports_dynamic_outputs(),
    #         stdin_stream=stdin_stream,
    #     ):
    #         if value is None:
    #             continue
    #         text_value = value if isinstance(value, str) else str(value)
    #         yield slot, text_value

    async def run(self, context: ProcessingContext, inputs: NodeInputs, outputs: NodeOutputs) -> None:  # type: ignore[override]
        if not self.command.strip():
            raise RuntimeError("Command is required")

        async def create_stdin_stream():
            if self.stdin:
                yield self.stdin
            async for value in inputs.stream("stdin"):
                yield str(value) if value is not None else ""

        use_stdin = (
            bool(self.stdin)
            or inputs.has_buffered("stdin")
            or inputs.has_stream("stdin")
        )
        stdin_stream = create_stdin_stream() if use_stdin else None

        runner = CommandDockerRunner(image=self.image.value)
        self._runner = runner
        async for slot, value in runner.stream(
            user_code=self.command,
            env_locals=self._dynamic_properties,
            context=context,
            node=self,
            allow_dynamic_outputs=self.supports_dynamic_outputs(),
            stdin_stream=stdin_stream,
        ):
            if value is None:
                continue
            text_value = value if isinstance(value, str) else str(value)
            # Send log updates for stdout/stderr
            if slot == "stdout":
                context.post_message(
                    LogUpdate(
                        node_id=self.id,
                        node_name=self.get_title(),
                        content=str(value).rstrip("\n"),
                        severity="info",
                    )
                )
            elif slot == "stderr":
                context.post_message(
                    LogUpdate(
                        node_id=self.id,
                        node_name=self.get_title(),
                        content=str(value).rstrip("\n"),
                        severity="error",
                    )
                )
            await outputs.emit(slot, text_value)

    async def finalize(self, context: ProcessingContext):  # type: ignore[override]
        """Stop any running Docker container for this node.

        Args:
            context: Processing context (unused).

        Returns:
            None
        """
        try:
            if self._runner:
                self._runner.stop()
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug(f"ExecuteCommand finalize: {e}")


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
