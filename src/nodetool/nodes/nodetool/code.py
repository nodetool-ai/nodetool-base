from typing import ClassVar, TypedDict
from enum import Enum
from nodetool.config.logging_config import get_logger
from pydantic import Field
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import LogUpdate
from nodetool.code_runners.python_runner import PythonDockerRunner
from nodetool.code_runners.javascript_runner import JavaScriptDockerRunner
from nodetool.code_runners.bash_runner import BashDockerRunner
from nodetool.code_runners.ruby_runner import RubyDockerRunner
from nodetool.code_runners.command_runner import CommandDockerRunner
from nodetool.code_runners.lua_runner import LuaRunner, LuaSubprocessRunner
from nodetool.workflows.io import NodeInputs, NodeOutputs
from nodetool.code_runners.runtime_base import StreamRunnerBase

log = get_logger(__name__)


class ExecutionMode(Enum):
    DOCKER = "docker"
    SUBPROCESS = "subprocess"


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

    _is_dynamic: ClassVar[bool] = True
    _supports_dynamic_outputs: ClassVar[bool] = True
    _runner: StreamRunnerBase | None = None

    class PythonImage(Enum):
        PYTHON_3_11_SLIM = "python:3.11-slim"
        JUPYTER_SCIPY_NOTEBOOK = "jupyter/scipy-notebook:latest"

    code: str = Field(
        default="",
        description=(
            "Python code to execute as-is. Dynamic inputs are provided as local vars. "
            "Stdout lines are emitted on 'stdout'; stderr lines on 'stderr'."
        ),
    )

    image: PythonImage = Field(
        default=PythonImage.PYTHON_3_11_SLIM,
        description="Docker image to use for execution",
    )

    execution_mode: ExecutionMode = Field(
        default=ExecutionMode.DOCKER,
        description="Execution mode: 'docker' or 'subprocess'",
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

    class OutputType(TypedDict):
        stdout: str
        stderr: str

    @classmethod
    def return_type(cls):
        return cls.OutputType

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

        runner = PythonDockerRunner(
            image=self.image.value,
            mode=self.execution_mode.value,
        )
        self._runner = runner
        async for slot, value in runner.stream(
            user_code=self.code,
            env_locals=self._dynamic_properties,
            context=context,
            node=self,
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
            log.debug(f"ExecutePython finalize: {e}")


class ExecuteJavaScript(BaseNode):
    """
    Executes JavaScript (Node.js) code with safety restrictions.
    javascript, nodejs, code, execute
    """

    _is_dynamic: ClassVar[bool] = True
    _supports_dynamic_outputs: ClassVar[bool] = True
    _runner: StreamRunnerBase | None = None

    class JavaScriptImage(Enum):
        NODE_22_ALPINE = "node:22-alpine"

    code: str = Field(
        default="",
        description=(
            "JavaScript code to execute as-is under Node.js. Dynamic inputs are provided as local vars. "
            "Stdout lines are emitted on 'stdout'; stderr lines on 'stderr'."
        ),
    )

    image: JavaScriptImage = Field(
        default=JavaScriptImage.NODE_22_ALPINE,
        description="Docker image to use for execution",
    )

    execution_mode: ExecutionMode = Field(
        default=ExecutionMode.DOCKER,
        description="Execution mode: 'docker' or 'subprocess'",
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

    class OutputType(TypedDict):
        stdout: str
        stderr: str

    @classmethod
    def return_type(cls):
        return cls.OutputType

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

        runner = JavaScriptDockerRunner(
            image=self.image.value,
            mode=self.execution_mode.value,
        )
        self._runner = runner
        async for slot, value in runner.stream(
            user_code=self.code,
            env_locals=self._dynamic_properties,
            context=context,
            node=self,
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
            log.debug(f"ExecuteJavaScript finalize: {e}")


class ExecuteBash(BaseNode):
    """
    Executes Bash script with safety restrictions.
    bash, shell, code, execute
    """

    _is_dynamic: ClassVar[bool] = True
    _supports_dynamic_outputs: ClassVar[bool] = True
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

    execution_mode: ExecutionMode = Field(
        default=ExecutionMode.DOCKER,
        description="Execution mode: 'docker' or 'subprocess'",
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

    class OutputType(TypedDict):
        stdout: str
        stderr: str

    @classmethod
    def return_type(cls):
        return cls.OutputType

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

        runner = BashDockerRunner(
            image=self.image.value,
            mode=self.execution_mode.value,
        )
        self._runner = runner
        async for slot, value in runner.stream(
            user_code=self.code,
            env_locals=self._dynamic_properties,
            context=context,
            node=self,
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
            log.debug(f"ExecuteBash finalize: {e}")


class ExecuteRuby(BaseNode):
    """
    Executes Ruby code with safety restrictions.
    ruby, code, execute
    """

    _is_dynamic: ClassVar[bool] = True
    _supports_dynamic_outputs: ClassVar[bool] = True
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

    execution_mode: ExecutionMode = Field(
        default=ExecutionMode.DOCKER,
        description="Execution mode: 'docker' or 'subprocess'",
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

    class OutputType(TypedDict):
        stdout: str
        stderr: str

    @classmethod
    def return_type(cls):
        return cls.OutputType

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

        runner = RubyDockerRunner(
            image=self.image.value,
            mode=self.execution_mode.value,
        )
        self._runner = runner
        async for slot, value in runner.stream(
            user_code=self.code,
            env_locals=self._dynamic_properties,
            context=context,
            node=self,
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
            log.debug(f"ExecuteRuby finalize: {e}")


class ExecuteLua(BaseNode):
    """
    Executes Lua code with a local sandbox (no Docker).
    lua, code, execute, sandbox
    """

    _is_dynamic: ClassVar[bool] = True
    _supports_dynamic_outputs: ClassVar[bool] = True
    _runner: StreamRunnerBase | None = None

    class LuaExecutable(Enum):
        LUA = "lua"
        LUAJIT = "luajit"

    code: str = Field(
        default="",
        description=(
            "Lua code to execute as-is in a restricted environment. Dynamic inputs are provided as variables. "
            "Stdout lines are emitted on 'stdout'; stderr lines on 'stderr'."
        ),
    )

    executable: LuaExecutable = Field(
        default=LuaExecutable.LUA, description="Lua executable to use"
    )

    execution_mode: ExecutionMode = Field(
        default=ExecutionMode.SUBPROCESS,
        description="Execution mode: 'docker' or 'subprocess'",
    )

    timeout_seconds: int = Field(
        default=10, description="Max seconds to allow execution before forced stop"
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

    class OutputType(TypedDict):
        stdout: str
        stderr: str

    @classmethod
    def return_type(cls):
        return cls.OutputType

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

        if self.execution_mode == ExecutionMode.SUBPROCESS:
            runner = LuaSubprocessRunner(
                executable=self.executable.value,
                timeout_seconds=int(self.timeout_seconds),
            )
        else:
            runner = LuaRunner(
                image="nickblah/lua:5.2.4-luarocks-ubuntu",
                timeout_seconds=int(self.timeout_seconds),
                mode=self.execution_mode.value,
            )
        # type: ignore[assignment]
        self._runner = runner  # StreamRunnerBase-compatible API
        async for slot, value in runner.stream(
            user_code=self.code,
            env_locals=self._dynamic_properties,
            context=context,
            node=self,
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
        try:
            if self._runner and hasattr(self._runner, "stop"):
                self._runner.stop()
        except Exception as e:
            log.debug(f"ExecuteLua finalize: {e}")


class ExecuteCommand(BaseNode):
    """
    Executes a single shell command inside a Docker container.
    command, execute, shell, bash, sh

    IMPORTANT: Only enabled in non-production environments
    """

    _is_dynamic: ClassVar[bool] = True
    _supports_dynamic_outputs: ClassVar[bool] = True
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

    execution_mode: ExecutionMode = Field(
        default=ExecutionMode.DOCKER,
        description="Execution mode: 'docker' or 'subprocess'",
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

    class OutputType(TypedDict):
        stdout: str
        stderr: str

    @classmethod
    def return_type(cls):
        return cls.OutputType

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

        runner = CommandDockerRunner(
            image=self.image.value,
            mode=self.execution_mode.value,
        )
        self._runner = runner
        async for slot, value in runner.stream(
            user_code=self.command,
            env_locals=self._dynamic_properties,
            context=context,
            node=self,
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
            log.debug(f"ExecuteCommand finalize: {e}")


class PythonRunner(BaseNode):
    """
    Runs a Python REPL that executes code from a stream of commands.
    python, code, execute, repl, stream, runner

    Use cases:
    - Execute multiple Python commands sequentially
    - Create interactive Python workflows
    - Process a stream of Python code snippets

    Each input from the 'commands' stream is executed as Python code.
    IMPORTANT: Only enabled in non-production environments
    """

    _is_dynamic: ClassVar[bool] = True
    _supports_dynamic_outputs: ClassVar[bool] = True
    _runner: StreamRunnerBase | None = None

    class PythonImage(Enum):
        PYTHON_3_11_SLIM = "python:3.11-slim"
        JUPYTER_SCIPY_NOTEBOOK = "jupyter/scipy-notebook:latest"

    image: PythonImage = Field(
        default=PythonImage.PYTHON_3_11_SLIM,
        description="Docker image to use for execution",
    )

    execution_mode: ExecutionMode = Field(
        default=ExecutionMode.DOCKER,
        description="Execution mode: 'docker' or 'subprocess'",
    )

    commands: str = Field(
        default="",
        description=(
            "Stream of Python commands to execute. Each command is executed separately. "
            "Connect this to a streaming source for continuous execution."
        ),
    )

    @classmethod
    def is_streaming_input(cls):
        return True

    class OutputType(TypedDict):
        stdout: str
        stderr: str

    @classmethod
    def return_type(cls):
        return cls.OutputType

    @classmethod
    def is_streaming_output(cls):
        return True

    async def run(self, context: ProcessingContext, inputs: NodeInputs, outputs: NodeOutputs) -> None:  # type: ignore[override]
        # Create runner once
        runner = PythonDockerRunner(
            image=self.image.value,
            mode=self.execution_mode.value,
        )
        self._runner = runner

        # Execute static command if provided
        if self.commands.strip():
            async for slot, value in runner.stream(
                user_code=self.commands,
                env_locals=self._dynamic_properties,
                context=context,
                node=self,
                stdin_stream=None,
            ):
                if value is None:
                    continue
                text_value = value if isinstance(value, str) else str(value)
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

        # Execute streaming commands one by one as they arrive
        async for value in inputs.stream("commands"):
            if value is None or not str(value).strip():
                continue
            command = str(value)
            async for slot, val in runner.stream(
                user_code=command,
                env_locals=self._dynamic_properties,
                context=context,
                node=self,
                stdin_stream=None,
            ):
                if val is None:
                    continue
                text_value = val if isinstance(val, str) else str(val)
                if slot == "stdout":
                    context.post_message(
                        LogUpdate(
                            node_id=self.id,
                            node_name=self.get_title(),
                            content=str(val).rstrip("\n"),
                            severity="info",
                        )
                    )
                elif slot == "stderr":
                    context.post_message(
                        LogUpdate(
                            node_id=self.id,
                            node_name=self.get_title(),
                            content=str(val).rstrip("\n"),
                            severity="error",
                        )
                    )
                await outputs.emit(slot, text_value)

    async def finalize(self, context: ProcessingContext):  # type: ignore[override]
        try:
            if self._runner:
                self._runner.stop()
        except Exception as e:
            log.debug(f"PythonRunner finalize: {e}")


class JavaScriptRunner(BaseNode):
    """
    Runs a JavaScript REPL that executes code from a stream of commands.
    javascript, nodejs, code, execute, repl, stream, runner

    Use cases:
    - Execute multiple JavaScript commands sequentially
    - Create interactive JavaScript workflows
    - Process a stream of JavaScript code snippets

    Each input from the 'commands' stream is executed as JavaScript code.
    """

    _is_dynamic: ClassVar[bool] = True
    _supports_dynamic_outputs: ClassVar[bool] = True
    _runner: StreamRunnerBase | None = None

    class JavaScriptImage(Enum):
        NODE_22_ALPINE = "node:22-alpine"

    image: JavaScriptImage = Field(
        default=JavaScriptImage.NODE_22_ALPINE,
        description="Docker image to use for execution",
    )

    execution_mode: ExecutionMode = Field(
        default=ExecutionMode.DOCKER,
        description="Execution mode: 'docker' or 'subprocess'",
    )

    commands: str = Field(
        default="",
        description=(
            "Stream of JavaScript commands to execute. Each command is executed separately. "
            "Connect this to a streaming source for continuous execution."
        ),
    )

    @classmethod
    def is_streaming_input(cls):
        return True

    class OutputType(TypedDict):
        stdout: str
        stderr: str

    @classmethod
    def return_type(cls):
        return cls.OutputType

    @classmethod
    def is_streaming_output(cls):
        return True

    async def run(self, context: ProcessingContext, inputs: NodeInputs, outputs: NodeOutputs) -> None:  # type: ignore[override]
        # Create runner once
        runner = JavaScriptDockerRunner(
            image=self.image.value,
            mode=self.execution_mode.value,
        )
        self._runner = runner

        # Execute static command if provided
        if self.commands.strip():
            async for slot, value in runner.stream(
                user_code=self.commands,
                env_locals=self._dynamic_properties,
                context=context,
                node=self,
                stdin_stream=None,
            ):
                if value is None:
                    continue
                text_value = value if isinstance(value, str) else str(value)
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

        # Execute streaming commands one by one as they arrive
        async for value in inputs.stream("commands"):
            if value is None or not str(value).strip():
                continue
            command = str(value)
            async for slot, val in runner.stream(
                user_code=command,
                env_locals=self._dynamic_properties,
                context=context,
                node=self,
                stdin_stream=None,
            ):
                if val is None:
                    continue
                text_value = val if isinstance(val, str) else str(val)
                if slot == "stdout":
                    context.post_message(
                        LogUpdate(
                            node_id=self.id,
                            node_name=self.get_title(),
                            content=str(val).rstrip("\n"),
                            severity="info",
                        )
                    )
                elif slot == "stderr":
                    context.post_message(
                        LogUpdate(
                            node_id=self.id,
                            node_name=self.get_title(),
                            content=str(val).rstrip("\n"),
                            severity="error",
                        )
                    )
                await outputs.emit(slot, text_value)

    async def finalize(self, context: ProcessingContext):  # type: ignore[override]
        try:
            if self._runner:
                self._runner.stop()
        except Exception as e:
            log.debug(f"JavaScriptRunner finalize: {e}")


class BashRunner(BaseNode):
    """
    Runs a Bash REPL that executes commands from a stream.
    bash, shell, code, execute, repl, stream, runner

    Use cases:
    - Execute multiple shell commands sequentially
    - Create interactive shell workflows
    - Process a stream of Bash commands

    Each input from the 'commands' stream is executed as a Bash command.
    """

    _is_dynamic: ClassVar[bool] = True
    _supports_dynamic_outputs: ClassVar[bool] = True
    _runner: StreamRunnerBase | None = None

    class BashImage(Enum):
        BASH_5_2 = "bash:5.2"
        DEBIAN_12 = "debian:12"
        UBUNTU_22_04 = "ubuntu:22.04"
        UBUNTU_24_04 = "ubuntu:24.04"
        JUPYTER_SCIPY_NOTEBOOK = "jupyter/scipy-notebook:latest"

    image: BashImage = Field(
        default=BashImage.UBUNTU_22_04,
        description="Docker image to use for execution",
    )

    execution_mode: ExecutionMode = Field(
        default=ExecutionMode.DOCKER,
        description="Execution mode: 'docker' or 'subprocess'",
    )

    commands: str = Field(
        default="",
        description=(
            "Stream of Bash commands to execute. Each command is executed separately. "
            "Connect this to a streaming source for continuous execution."
        ),
    )

    @classmethod
    def is_streaming_input(cls):
        return True

    @classmethod
    def is_streaming_output(cls):
        return True

    class OutputType(TypedDict):
        stdout: str
        stderr: str

    @classmethod
    def return_type(cls):
        return cls.OutputType

    async def run(self, context: ProcessingContext, inputs: NodeInputs, outputs: NodeOutputs) -> None:  # type: ignore[override]
        # Create runner once
        runner = BashDockerRunner(
            image=self.image.value,
            mode=self.execution_mode.value,
        )
        self._runner = runner

        # Execute static command if provided
        if self.commands.strip():
            async for slot, value in runner.stream(
                user_code=self.commands,
                env_locals=self._dynamic_properties,
                context=context,
                node=self,
                stdin_stream=None,
            ):
                if value is None:
                    continue
                text_value = value if isinstance(value, str) else str(value)
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

        # Execute streaming commands one by one as they arrive
        async for value in inputs.stream("commands"):
            if value is None or not str(value).strip():
                continue
            command = str(value)
            async for slot, val in runner.stream(
                user_code=command,
                env_locals=self._dynamic_properties,
                context=context,
                node=self,
                stdin_stream=None,
            ):
                if val is None:
                    continue
                text_value = val if isinstance(val, str) else str(val)
                if slot == "stdout":
                    context.post_message(
                        LogUpdate(
                            node_id=self.id,
                            node_name=self.get_title(),
                            content=str(val).rstrip("\n"),
                            severity="info",
                        )
                    )
                elif slot == "stderr":
                    context.post_message(
                        LogUpdate(
                            node_id=self.id,
                            node_name=self.get_title(),
                            content=str(val).rstrip("\n"),
                            severity="error",
                        )
                    )
                await outputs.emit(slot, text_value)

    async def finalize(self, context: ProcessingContext):  # type: ignore[override]
        try:
            if self._runner:
                self._runner.stop()
        except Exception as e:
            log.debug(f"BashRunner finalize: {e}")


class RubyRunner(BaseNode):
    """
    Runs a Ruby REPL that executes code from a stream of commands.
    ruby, code, execute, repl, stream, runner

    Use cases:
    - Execute multiple Ruby commands sequentially
    - Create interactive Ruby workflows
    - Process a stream of Ruby code snippets

    Each input from the 'commands' stream is executed as Ruby code.
    """

    _is_dynamic: ClassVar[bool] = True
    _supports_dynamic_outputs: ClassVar[bool] = True
    _runner: StreamRunnerBase | None = None

    class RubyImage(Enum):
        RUBY_3_3_ALPINE = "ruby:3.3-alpine"

    image: RubyImage = Field(
        default=RubyImage.RUBY_3_3_ALPINE,
        description="Docker image to use for execution",
    )

    execution_mode: ExecutionMode = Field(
        default=ExecutionMode.DOCKER,
        description="Execution mode: 'docker' or 'subprocess'",
    )

    commands: str = Field(
        default="",
        description=(
            "Stream of Ruby commands to execute. Each command is executed separately. "
            "Connect this to a streaming source for continuous execution."
        ),
    )

    @classmethod
    def is_streaming_input(cls):
        return True

    @classmethod
    def is_streaming_output(cls):
        return True

    class OutputType(TypedDict):
        stdout: str
        stderr: str

    @classmethod
    def return_type(cls):
        return cls.OutputType

    async def run(self, context: ProcessingContext, inputs: NodeInputs, outputs: NodeOutputs) -> None:  # type: ignore[override]
        # Create runner once
        runner = RubyDockerRunner(
            image=self.image.value,
            mode=self.execution_mode.value,
        )
        self._runner = runner

        # Execute static command if provided
        if self.commands.strip():
            async for slot, value in runner.stream(
                user_code=self.commands,
                env_locals=self._dynamic_properties,
                context=context,
                node=self,
                stdin_stream=None,
            ):
                if value is None:
                    continue
                text_value = value if isinstance(value, str) else str(value)
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

        # Execute streaming commands one by one as they arrive
        async for value in inputs.stream("commands"):
            if value is None or not str(value).strip():
                continue
            command = str(value)
            async for slot, val in runner.stream(
                user_code=command,
                env_locals=self._dynamic_properties,
                context=context,
                node=self,
                stdin_stream=None,
            ):
                if val is None:
                    continue
                text_value = val if isinstance(val, str) else str(val)
                if slot == "stdout":
                    context.post_message(
                        LogUpdate(
                            node_id=self.id,
                            node_name=self.get_title(),
                            content=str(val).rstrip("\n"),
                            severity="info",
                        )
                    )
                elif slot == "stderr":
                    context.post_message(
                        LogUpdate(
                            node_id=self.id,
                            node_name=self.get_title(),
                            content=str(val).rstrip("\n"),
                            severity="error",
                        )
                    )
                await outputs.emit(slot, text_value)

    async def finalize(self, context: ProcessingContext):  # type: ignore[override]
        try:
            if self._runner:
                self._runner.stop()
        except Exception as e:
            log.debug(f"RubyRunner finalize: {e}")


class LuaRunnerNode(BaseNode):
    """
    Runs a Lua REPL that executes code from a stream of commands.
    lua, code, execute, repl, stream, runner, sandbox

    Use cases:
    - Execute multiple Lua commands sequentially
    - Create interactive Lua workflows
    - Process a stream of Lua code snippets

    Each input from the 'commands' stream is executed as Lua code.
    """

    _is_dynamic: ClassVar[bool] = True
    _supports_dynamic_outputs: ClassVar[bool] = True
    _runner: StreamRunnerBase | None = None

    class LuaExecutable(Enum):
        LUA = "lua"
        LUAJIT = "luajit"

    executable: LuaExecutable = Field(
        default=LuaExecutable.LUA, description="Lua executable to use"
    )

    execution_mode: ExecutionMode = Field(
        default=ExecutionMode.SUBPROCESS,
        description="Execution mode: 'docker' or 'subprocess'",
    )

    timeout_seconds: int = Field(
        default=10, description="Max seconds to allow execution before forced stop"
    )

    commands: str = Field(
        default="",
        description=(
            "Stream of Lua commands to execute. Each command is executed separately. "
            "Connect this to a streaming source for continuous execution."
        ),
    )

    @classmethod
    def is_streaming_input(cls):
        return True

    @classmethod
    def is_streaming_output(cls):
        return True

    class OutputType(TypedDict):
        stdout: str
        stderr: str

    @classmethod
    def return_type(cls):
        return cls.OutputType

    async def run(self, context: ProcessingContext, inputs: NodeInputs, outputs: NodeOutputs) -> None:  # type: ignore[override]
        # Create runner once (reused for all commands)
        if self.execution_mode == ExecutionMode.SUBPROCESS:
            runner = LuaSubprocessRunner(
                executable=self.executable.value,
                timeout_seconds=int(self.timeout_seconds),
            )
        else:
            runner = LuaRunner(
                image="nickblah/lua:5.2.4-luarocks-ubuntu",
                timeout_seconds=int(self.timeout_seconds),
                mode=self.execution_mode.value,
            )
        self._runner = runner

        # Execute static command if provided
        if self.commands.strip():
            async for slot, value in runner.stream(
                user_code=self.commands,
                env_locals=self._dynamic_properties,
                context=context,
                node=self,
                stdin_stream=None,
            ):
                if value is None:
                    continue
                text_value = value if isinstance(value, str) else str(value)
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

        # Execute streaming commands one by one as they arrive
        async for value in inputs.stream("commands"):
            if value is None or not str(value).strip():
                continue
            command = str(value)
            async for slot, val in runner.stream(
                user_code=command,
                env_locals=self._dynamic_properties,
                context=context,
                node=self,
                stdin_stream=None,
            ):
                if val is None:
                    continue
                text_value = val if isinstance(val, str) else str(val)
                if slot == "stdout":
                    context.post_message(
                        LogUpdate(
                            node_id=self.id,
                            node_name=self.get_title(),
                            content=str(val).rstrip("\n"),
                            severity="info",
                        )
                    )
                elif slot == "stderr":
                    context.post_message(
                        LogUpdate(
                            node_id=self.id,
                            node_name=self.get_title(),
                            content=str(val).rstrip("\n"),
                            severity="error",
                        )
                    )
                await outputs.emit(slot, text_value)

    async def finalize(self, context: ProcessingContext):  # type: ignore[override]
        try:
            if self._runner and hasattr(self._runner, "stop"):
                self._runner.stop()
        except Exception as e:
            log.debug(f"LuaRunnerNode finalize: {e}")


class ShellRunner(BaseNode):
    """
    Runs a shell command runner that executes commands from a stream.
    command, execute, shell, bash, sh, stream, runner

    Use cases:
    - Execute multiple shell commands sequentially
    - Create interactive command workflows
    - Process a stream of shell commands

    Each input from the 'commands' stream is executed as a shell command.
    IMPORTANT: Only enabled in non-production environments
    """

    _is_dynamic: ClassVar[bool] = True
    _supports_dynamic_outputs: ClassVar[bool] = True
    _runner: StreamRunnerBase | None = None

    class CommandImage(Enum):
        BASH_5_2 = "bash:5.2"
        ALPINE_3 = "alpine:3"
        UBUNTU_22_04 = "ubuntu:22.04"
        UBUNTU_24_04 = "ubuntu:24.04"

    image: CommandImage = Field(
        default=CommandImage.BASH_5_2,
        description="Docker image to use for execution",
    )

    execution_mode: ExecutionMode = Field(
        default=ExecutionMode.DOCKER,
        description="Execution mode: 'docker' or 'subprocess'",
    )

    commands: str = Field(
        default="",
        description=(
            "Stream of shell commands to execute. Each command is executed separately. "
            "Connect this to a streaming source for continuous execution."
        ),
    )

    @classmethod
    def is_streaming_input(cls):
        return True

    @classmethod
    def is_streaming_output(cls):
        return True

    class OutputType(TypedDict):
        stdout: str
        stderr: str

    @classmethod
    def return_type(cls):
        return cls.OutputType

    async def run(self, context: ProcessingContext, inputs: NodeInputs, outputs: NodeOutputs) -> None:  # type: ignore[override]
        # Create runner once
        runner = CommandDockerRunner(
            image=self.image.value,
            mode=self.execution_mode.value,
        )
        self._runner = runner

        # Execute static command if provided
        if self.commands.strip():
            async for slot, value in runner.stream(
                user_code=self.commands,
                env_locals=self._dynamic_properties,
                context=context,
                node=self,
                stdin_stream=None,
            ):
                if value is None:
                    continue
                text_value = value if isinstance(value, str) else str(value)
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

        # Execute streaming commands one by one as they arrive
        async for value in inputs.stream("commands"):
            if value is None or not str(value).strip():
                continue
            command = str(value)
            async for slot, val in runner.stream(
                user_code=command,
                env_locals=self._dynamic_properties,
                context=context,
                node=self,
                stdin_stream=None,
            ):
                if val is None:
                    continue
                text_value = val if isinstance(val, str) else str(val)
                if slot == "stdout":
                    context.post_message(
                        LogUpdate(
                            node_id=self.id,
                            node_name=self.get_title(),
                            content=str(val).rstrip("\n"),
                            severity="info",
                        )
                    )
                elif slot == "stderr":
                    context.post_message(
                        LogUpdate(
                            node_id=self.id,
                            node_name=self.get_title(),
                            content=str(val).rstrip("\n"),
                            severity="error",
                        )
                    )
                await outputs.emit(slot, text_value)

    async def finalize(self, context: ProcessingContext):  # type: ignore[override]
        try:
            if self._runner:
                self._runner.stop()
        except Exception as e:
            log.debug(f"ShellRunner finalize: {e}")
