"""
Python Docker Runner (streaming-only)
====================================

This module provides a sandboxed, streaming-only Python execution environment
implemented via Docker. User-supplied Python code is executed inside an
ephemeral container and results are streamed back to the workflow engine as
they are produced.

Key characteristics
-------------------
- Streaming-only API: execution always yields incremental results; there is no
  blocking, non-streaming call. Use `PythonDockerRunner.stream(...)`.
- Stdout capture: every non-control line written to stdout by the user's code
  is forwarded as a streamed item on slot "output".
- Control protocol: the injected wrapper uses a single control prefix
  `data:` with JSON payloads. Supported payloads:
  - Yield item: `{ "slot": string, "value": any }` — streamed immediately.
  There is no final control message; the stream ends when the container exits.
- Container isolation: network is disabled; memory/CPU limits are configurable.
- Image management: the runner pulls `image` if it is not present locally.
- Requirements: Docker daemon must be running; the host must have the
  `docker` Python SDK installed.

Usage
-----
Create an instance and iterate the async generator:

    runner = PythonDockerRunner()
    async for slot, value in runner.stream(user_code, env_locals, context, node):
        ...

Author code contract
--------------------
- Your code is wrapped as:

    def main(**env):
        ...  # your code

  The mapping `env` is constructed from `env_locals` and passed to `main`.
- If `main` returns an iterator/generator, each item is streamed. You may
  either yield raw values (sent on slot "output") or `(slot, value)` tuples to
  target a specific output slot.
- You may also print to stdout; each line is forwarded on slot "output".
- To finish with a value, simply `return value` from `main`; the wrapper emits
  a `data:` line on your behalf.

Security notes
--------------
This is a best-effort sandbox intended for non-production execution. While the
container has no network and is resource-limited, user code still runs as
Python and may be able to exhaust resources or exploit interpreter internals.
Do not enable in production environments.
"""

import asyncio
import json
import textwrap
from typing import Any, AsyncGenerator

from nodetool.workflows.types import NodeProgress, NodeUpdate
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext


class PythonDockerRunner:
    """Execute Python code inside Docker and stream results.

    Parameters
    ----------
    image:
        Docker image to use (must contain Python). Defaults to
        "python:3.11-slim".
    mem_limit:
        Docker memory limit (e.g., "256m").
    nano_cpus:
        CPU quota in nanoseconds (e.g., 1_000_000_000 for ~1 core).
    timeout_seconds:
        Maximum allowed runtime for a job. Currently configured at the object
        level; enforcement is expected to be handled by the caller or
        integrated here in the future.
    """

    def __init__(
        self,
        image: str = "python:3.11-slim",
        mem_limit: str = "256m",
        nano_cpus: int = 1_000_000_000,
        timeout_seconds: int = 10,
    ):
        self.image = image
        self.mem_limit = mem_limit
        self.nano_cpus = nano_cpus
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def _to_python_literal(value: Any) -> str:
        """Convert a Python value to a safe literal for code injection.

        This is used to serialize `env_locals` into the container script in a
        way that preserves types for common JSON-friendly structures and simple
        Python objects.

        Supported: None, bool, int/float, str, list, tuple, dict (string keys).
        Raises for unsupported types.
        """
        if value is None:
            return "None"
        if isinstance(value, bool):
            return "True" if value else "False"
        if isinstance(value, (int, float)):
            return repr(value)
        if isinstance(value, str):
            return repr(value)
        if isinstance(value, list):
            return (
                "["
                + ", ".join(PythonDockerRunner._to_python_literal(v) for v in value)
                + "]"
            )
        if isinstance(value, tuple):
            inner = ", ".join(PythonDockerRunner._to_python_literal(v) for v in value)
            if len(value) == 1:
                inner += ","
            return "(" + inner + ")"
        if isinstance(value, dict):
            for k in value.keys():
                if not isinstance(k, str):
                    raise RuntimeError("Only string keys are supported in locals")
            items = ", ".join(
                f"{repr(k)}: {PythonDockerRunner._to_python_literal(v)}"
                for k, v in value.items()
            )
            return "{" + items + "}"
        raise RuntimeError(f"Unsupported type in locals: {type(value).__name__}")

    @staticmethod
    def _build_script(user_code: str, locals_literal: str) -> str:
        """Generate the container Python script with the control protocol.

        The generated script:
        - Defines a `main(**env)` function that wraps user code
        - Executes `main` and streams results using the `data:` prefix
        - There is no mandatory final message; the container exit ends the stream
        - Exceptions are forwarded as `data:` lines with an error payload
        """
        indented_user_code = textwrap.indent(user_code, "    ")
        main_code = "def main(**env):\n" + indented_user_code + "\n"

        return (
            (
                "import json\n"
                "import inspect\n"
                "PREFIX='data:'\n"
                f"env = {locals_literal}\n"
                "\n"
            )
            + main_code
            + "\n"
            + (
                "\ntry:\n"
                "    res = main(**env)\n"
                "    if inspect.isgenerator(res) or hasattr(res, '__iter__') and not isinstance(res, (str, bytes, dict)):\n"
                "        for _item in res:\n"
                "            if isinstance(_item, tuple) and len(_item) == 2 and isinstance(_item[0], str):\n"
                "                _slot, _val = _item\n"
                "            else:\n"
                "                _slot, _val = 'output', _item\n"
                "            print(PREFIX + json.dumps({'slot': _slot, 'value': _val}, default=str))\n"
                "    else:\n"
                "        print(PREFIX + json.dumps({'slot': 'output', 'value': res}, default=str))\n"
                "except Exception as e:\n"
                "    print(PREFIX + json.dumps({'slot': 'error', 'value': str(e)}, default=str))\n"
            )
        )

    async def stream(
        self,
        user_code: str,
        env_locals: dict[str, Any],
        context: ProcessingContext,
        node: BaseNode,
        allow_dynamic_outputs: bool = True,
    ) -> AsyncGenerator[tuple[str, Any], None]:
        """Stream results from user code running inside Docker.

        Parameters
        ----------
        user_code:
            Python source to run inside `def main(**env): ...`.
        env_locals:
            Mapping made available to `main` as keyword arguments.
        context:
            Processing context used for posting `NodeUpdate` messages and other
            workflow concerns.
        node:
            The workflow node initiating this execution.
        allow_dynamic_outputs:
            If True, new output slots are added on-the-fly when unseen slots are
            yielded by the user code.

        Yields
        ------
        (slot, value):
            - For control yields: the slot specified by the user or "output" by
              default.
            - For stdout lines: slot is "output", value is the raw line.

        Notes
        -----
        There is no required final control message. The stream ends when the
        container exits. Exceptions inside the container are emitted as
        `data:` lines on slot "error".
        """
        try:
            locals_literal = PythonDockerRunner._to_python_literal(env_locals)
        except Exception as e:
            raise RuntimeError(f"Failed to convert inputs to Python literals: {str(e)}")

        container_script = self._build_script(user_code, locals_literal)

        import asyncio as _asyncio

        queue: _asyncio.Queue[dict[str, Any]] = _asyncio.Queue()
        loop = _asyncio.get_running_loop()

        def _sync_run_stream():
            try:
                import docker

                client = docker.from_env()
                try:
                    client.ping()
                except Exception:
                    raise RuntimeError(
                        "Docker daemon is not available. Please start Docker and try again."
                    )

                command = ["python", "-c", container_script]

                # Ensure image is present
                try:
                    client.images.get(self.image)
                except Exception:
                    context.post_message(
                        NodeProgress(
                            node_id=node.id,
                            progress=0,
                            total=100,
                            chunk=f"Pulling image: {self.image}",
                        )
                    )
                    client.images.pull(self.image)

                container = None
                try:
                    container = client.containers.create(
                        image=self.image,
                        command=command,
                        network_disabled=True,
                        mem_limit=self.mem_limit,
                        nano_cpus=self.nano_cpus,
                        stdin_open=False,
                        tty=False,
                        detach=True,
                    )
                    container.start()

                    for raw in container.logs(
                        stream=True, follow=True, stdout=True, stderr=True
                    ):
                        try:
                            line = raw.decode("utf-8", errors="ignore").strip()
                        except Exception:
                            line = ""
                        if not line:
                            continue

                        if line.startswith("data:"):
                            payload = line.split(":", 1)[1].strip()
                            try:
                                data = json.loads(payload)
                                if isinstance(data, dict) and "slot" in data:
                                    _asyncio.run_coroutine_threadsafe(
                                        queue.put({"type": "yield", **data}), loop
                                    )
                                else:
                                    # Unrecognized control payload, forward raw line as output
                                    _asyncio.run_coroutine_threadsafe(
                                        queue.put(
                                            {
                                                "type": "yield",
                                                "slot": "output",
                                                "value": line,
                                            }
                                        ),
                                        loop,
                                    )
                            except Exception:
                                # Invalid JSON in control line, forward raw as output
                                _asyncio.run_coroutine_threadsafe(
                                    queue.put(
                                        {
                                            "type": "yield",
                                            "slot": "output",
                                            "value": line,
                                        }
                                    ),
                                    loop,
                                )
                        else:
                            # Forward any non-control stdout as output yields
                            _asyncio.run_coroutine_threadsafe(
                                queue.put(
                                    {"type": "yield", "slot": "output", "value": line}
                                ),
                                loop,
                            )

                    # Stream completes on container exit; enqueue a terminal message
                    _asyncio.run_coroutine_threadsafe(
                        queue.put({"type": "final", "ok": True}),
                        loop,
                    )
                finally:
                    try:
                        if container is not None:
                            container.remove(force=True)
                    except Exception:
                        pass
            except Exception as e:
                _asyncio.run_coroutine_threadsafe(
                    queue.put({"type": "final", "ok": False, "error": str(e)}), loop
                )

        _ = _asyncio.create_task(_asyncio.to_thread(_sync_run_stream))

        while True:
            msg = await queue.get()
            if not isinstance(msg, dict):
                continue
            if msg.get("type") == "yield":
                slot = msg.get("slot", "output")
                value = msg.get("value")
                if allow_dynamic_outputs:
                    try:
                        node.add_output(
                            slot, type(value) if value is not None else None
                        )
                    except Exception:
                        pass
                yield slot, value
            elif msg.get("type") == "final":
                if not msg.get("ok", False):
                    raise RuntimeError(
                        f"Error executing Python code: {msg.get('error', 'Unknown error')}"
                    )
                break
