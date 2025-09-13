import base64
import asyncio
from enum import Enum
import io
import json

from nodetool.agents.tools.workflow_tool import GraphTool
from nodetool.nodes.nodetool.agents import serialize_tool_result
from nodetool.workflows.graph_utils import find_node, get_downstream_subgraph

from openai.resources.beta.realtime.realtime import AsyncRealtimeConnection
from openai.types.beta.realtime import (
    ConversationItemParam,
    ResponseAudioDoneEvent,
    ResponseDoneEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
    Session,
)
from openai.types.beta.realtime.session import (
    Tool as OpenAITool,
    TurnDetection,
)
from openai.types.beta.realtime.session_update_event_param import Session
from openai.types.beta.realtime.response_create_event_param import Response
from openai.types.beta.realtime.error_event import ErrorEvent
from openai.types.beta.realtime.response_audio_delta_event import (
    ResponseAudioDeltaEvent,
)
from openai.types.beta.realtime.response_audio_transcript_delta_event import (
    ResponseAudioTranscriptDeltaEvent,
)
from pydantic import Field

from nodetool.agents.tools.base import Tool
from nodetool.config.environment import Environment

from nodetool.workflows.types import (
    ToolCallUpdate,
)

from nodetool.metadata.types import (
    AudioRef,
    LanguageModel,
)
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.io import NodeInputs, NodeOutputs
from nodetool.chat.providers import Chunk
from nodetool.config.logging_config import get_logger

log = get_logger(__name__)

DEFAULT_SYSTEM_PROMPT = """
You are an AI assistant interacting in real-time. Follow these rules unless explicitly overridden by the user:

1. Respond promptly — minimize delay. If you do not yet have a complete answer, acknowledge the question and indicate what you are doing to find the answer.
2. Maintain correctness. Always aim for accuracy; if you’re uncertain, say so and optionally offer to verify.
3. Be concise but clear. Prioritize key information first, then supporting details if helpful.
4. Ask clarifying questions when needed. If the user’s request is ambiguous, request clarification rather than guessing.
5. Be consistent in terminology and definitions. Once you adopt a term or abbreviation, use it consistently in this conversation.
6. Respect politeness and neutrality. Do not use emotive language unless the conversation tone demands it.
7. Stay within safe and ethical bounds. Avoid disallowed content; follow OpenAI policies.
8. Adapt to the user’s style and level. If the user seems technical, use technical detail; if non-technical, explain with simpler language.
---
You are now active. Await the user’s request.
"""


class RealtimeAgent(BaseNode):
    """
    Stream responses using the official OpenAI Realtime client. Supports optional audio input and streams text chunks.
    realtime, streaming, openai, audio-input, text-output

    Uses `AsyncOpenAI().beta.realtime.connect(...)` with the events API:
    - Sends session settings via `session.update`
    - Adds user input via `conversation.item.create`
    - Streams back `response.text.delta` events until `response.done`
    """

    class Model(str, Enum):
        GPT_4O_REaltime = "gpt-4o-realtime-preview"
        GPT_4O_MINI_REaltime = "gpt-4o-mini-realtime-preview"

    class Voice(str, Enum):
        NONE = "none"
        ASH = "ash"
        ALLOY = "alloy"
        BALLAD = "ballad"
        CORAL = "coral"
        ECHO = "echo"
        FABLE = "fable"
        ONYX = "onyx"
        NOVA = "nova"
        SHIMMER = "shimmer"
        SAGE = "sage"
        VERSE = "verse"

    model: Model = Field(title="Model", default=Model.GPT_4O_MINI_REaltime)

    system: str = Field(
        title="System",
        default=DEFAULT_SYSTEM_PROMPT,
        description="System instructions for the realtime session",
    )
    chunk: Chunk = Field(
        title="Chunk",
        default=Chunk(),
        description="The audio chunk to use as input.",
    )
    voice: Voice = Field(
        title="Voice",
        default=Voice.ALLOY,
        description="The voice for the audio output",
    )
    speed: float = Field(
        title="Speed",
        ge=0.25,
        le=1.5,
        default=1.0,
        description="The speed of the model's spoken response",
    )
    temperature: float = Field(
        title="Temperature",
        ge=0.6,
        le=1.2,
        default=0.8,
        description="The temperature for the response",
    )

    _supports_dynamic_outputs = True

    def should_route_output(self, output_name: str) -> bool:
        """
        Do not route dynamic outputs; they represent tool entry points.
        Still route declared outputs like 'text', 'chunk', 'audio'.
        """
        return output_name not in self._dynamic_outputs

    @classmethod
    def is_streaming_output(cls) -> bool:
        return True

    @classmethod
    def is_cacheable(cls) -> bool:
        return False

    @classmethod
    def is_streaming_input(cls) -> bool:
        return True

    @classmethod
    def return_type(cls):
        return {
            "chunk": Chunk,
            "audio": AudioRef,
            "text": str,
        }

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["model", "prompt", "chunk", "speed"]

    def _resolve_tools(self, context: ProcessingContext) -> list[Tool]:
        tools: list[Tool] = []
        for name, type_meta in self._dynamic_outputs.items():
            initial_edges, graph = get_downstream_subgraph(context.graph, self.id, name)
            initial_nodes = [find_node(graph, edge.target) for edge in initial_edges]
            nodes = graph.nodes
            if len(nodes) == 0:
                log.debug(f"Skipping tool {name}: no downstream nodes")
                continue
            tool = GraphTool(graph, name, "", initial_edges, initial_nodes)
            tools.append(tool)
        log.info(f"Resolved {len(tools)} tools: {[t.name for t in tools]}")
        return tools

    def _format_tools_for_realtime(self, tools: list[Tool]) -> list[OpenAITool]:
        """Format tools for inclusion in `Response(tools=...)` for Realtime API.

        Args:
            tools (list[Tool]): Resolved tools.

        Returns:
            list[dict]: Realtime response tool descriptors.
        """
        return [
            OpenAITool(
                name=tool.name,
                description=tool.description,
                parameters=tool.input_schema,
            )
            for tool in tools
        ]

    def _format_session_tools_for_realtime(self, tools: list[Tool]) -> list[OpenAITool]:
        """Format tools for inclusion in session.update tools.

        Args:
            tools (list[Tool]): Resolved tools.

        Returns:
            list[dict]: Session-level tool descriptors.
        """
        return [
            OpenAITool(
                type="function",
                name=tool.name,
                description=tool.description,
                parameters=tool.input_schema,
            )
            for tool in tools
        ]

    async def _trigger_response(
        self, connection: AsyncRealtimeConnection, shared: dict
    ) -> None:
        """Request a new response from the Realtime API.

        Args:
            connection: OpenAI realtime connection resource.
            tool_defs (list[dict]): Response-level tool definitions.
            shared (dict): Shared state dict tracking 'pending' responses.
        """
        await connection.response.create(response=Response())
        shared["pending"] = shared.get("pending", 0) + 1

    async def _send_user_text(
        self, connection: AsyncRealtimeConnection, text: str
    ) -> None:
        """Send a user text message to the live conversation.

        Args:
            connection: OpenAI realtime connection resource.
            text (str): Text content to send.
        """
        await connection.conversation.item.create(
            item={
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": text}],
            }
        )

    async def _append_audio_chunks_b64(
        self, connection: AsyncRealtimeConnection, chunks: list[str]
    ) -> None:
        """Append base64 PCM16 audio chunks to the input buffer.

        Args:
            connection: OpenAI realtime connection resource.
            chunks (list[str]): Base64-encoded PCM16 audio chunks.
        """
        for ch in chunks:
            await connection.input_audio_buffer.append(audio=ch)

    async def _handle_function_call_completion(
        self,
        connection: AsyncRealtimeConnection,
        context: ProcessingContext,
        tools_instances: list[Tool],
        call_id: str,
        tool_name: str | None,
        args_str: str,
    ) -> None:
        """Execute a completed function call and send its output to the model.

        Args:
            connection: OpenAI realtime connection resource.
            context (ProcessingContext): Workflow execution context.
            tools_instances (list[Tool]): Resolved tools.
            call_id (str): Function call identifier from the model.
            tool_name (str | None): Tool/function name requested by the model.
            args_str (str | dict): Arguments payload (JSON string or dict).
        """
        log.info(
            f"Tool call initiated: call_id={call_id}, tool_name={tool_name} args={args_str}"
        )

        parsed_args = json.loads(args_str)

        matching: Tool | None = None
        for t in tools_instances:
            if t and t.name == tool_name:
                matching = t
                break

        if matching is None:
            log.warning(f"Tool not found: {tool_name}")
            return

        log.info(f"Tool matched: {tool_name}, executing with args: {parsed_args}")

        context.post_message(
            ToolCallUpdate(
                node_id=self.id,
                name=tool_name or "",
                args=parsed_args,
                message=matching.user_message(parsed_args),
            )
        )

        try:
            log.debug(f"Starting tool execution: {tool_name}")
            tool_result = await matching.process(context, parsed_args)
            tool_result_json = json.dumps(serialize_tool_result(tool_result))
            log.info(f"Tool execution completed successfully: {tool_name}")
        except Exception as e:
            log.error(f"Tool execution failed: {tool_name}, error: {str(e)}")
            tool_result_json = json.dumps({"error": f"Tool error: {str(e)}"})

        log.debug(f"Sending tool result back to model: call_id={call_id}")
        await connection.conversation.item.create(
            item=ConversationItemParam(
                type="function_call_output",
                call_id=call_id,
                output=tool_result_json,
            )
        )
        log.info(f"Tool result sent back to model: call_id={call_id}")

    async def _producer_loop(
        self,
        connection: AsyncRealtimeConnection,
        inputs: NodeInputs,
        context: ProcessingContext,
        shared: dict,
    ) -> None:
        """Continuously read streaming inputs and forward them to the session.

        Args:
            connection: OpenAI realtime connection resource.
            inputs (NodeInputs): Streaming workflow inputs.
            context (ProcessingContext): Workflow execution context.
            tool_defs (list[dict]): Response-level tool definitions.
            shared (dict): Shared state dict with flags/counters.
        """
        log.debug("Producer loop started")
        chunk_text_buf: list[str] = []
        async for handle, item in inputs.any():
            if handle == "chunk":
                assert isinstance(item, Chunk)
                if item.content_type == "audio":
                    if item.content:
                        log.debug(f"Received audio chunk: {len(item.content)} bytes")
                        await connection.input_audio_buffer.append(audio=item.content)
                    if getattr(item, "done", False):
                        log.debug("Audio chunk done, triggering response")
                        await self._trigger_response(connection, shared)
                else:
                    if item.content:
                        chunk_text_buf.append(item.content)
                    if item.done:
                        text_val = "".join(chunk_text_buf)
                        chunk_text_buf = []
                        if text_val.strip():
                            log.debug(f"Sending text input: {len(text_val)} characters")
                            await self._send_user_text(connection, text_val)
                            await self._trigger_response(connection, shared)
            else:
                log.error(f"Unknown handle in producer loop: {handle}")
                raise ValueError(f"Unknown handle: {handle}")

        log.debug("Producer loop completed")
        shared["done_producing"] = True

    async def _consumer_loop(
        self,
        connection: AsyncRealtimeConnection,
        context: ProcessingContext,
        outputs: NodeOutputs,
        tools_instances: list[Tool],
        shared: dict,
    ) -> None:
        """Consume realtime events, stream deltas, and handle function calls.

        Args:
            connection: OpenAI realtime connection resource.
            context (ProcessingContext): Workflow execution context.
            outputs (NodeOutputs): Output emitter for streaming data.
            tools_instances (list[Tool]): Resolved tools.
            shared (dict): Shared state dict with flags/counters.
        """
        log.debug("Consumer loop started")
        full_text_parts: list[str] = []
        audio_accum = bytearray()
        current_response_id: str | None = None

        async for event in connection:
            event_type = getattr(event, "type", None)
            if event_type is None:
                try:
                    event_type = event.model_dump().get("type")  # type: ignore[attr-defined]
                except Exception:
                    event_type = None
            log.debug(f"Received event: {event_type or type(event).__name__}")

            if isinstance(event, ResponseTextDeltaEvent):
                current_response_id = event.response_id
                if event.delta:
                    full_text_parts.append(event.delta or "")
                    log.debug(f"Text delta: {len(event.delta)} characters")
                    await outputs.emit(
                        "chunk", Chunk(content=event.delta or "", done=False)
                    )
            elif isinstance(event, ResponseAudioTranscriptDeltaEvent):
                current_response_id = event.response_id
                if event.delta:
                    full_text_parts.append(event.delta or "")
                    log.debug(f"Audio transcript delta: {len(event.delta)} characters")
                    await outputs.emit(
                        "chunk", Chunk(content=event.delta or "", done=False)
                    )
            elif isinstance(event, ResponseAudioDeltaEvent):
                current_response_id = event.response_id
                if event.delta:
                    audio_accum.extend(base64.b64decode(event.delta))
                    log.debug(f"Audio delta: {len(event.delta)} bytes")
                    await outputs.emit(
                        "chunk",
                        Chunk(
                            content=event.delta or "", done=False, content_type="audio"
                        ),
                    )
            elif isinstance(event_type, str) and event_type in (
                "response.output_item.added",
                "response.output_item.done",
            ):
                data = event.model_dump()  # type: ignore[attr-defined]
                item = data.get("item", {})
                if item.get("type") == "function_call":
                    log.info(
                        f"Function call event: {item.get('name')} (call_id: {item.get('call_id')})"
                    )
                    await self._handle_function_call_completion(
                        connection,
                        context,
                        tools_instances,
                        item.get("call_id"),
                        item.get("name"),
                        item.get("arguments", "{}"),
                    )
            elif isinstance(event, ResponseTextDoneEvent):
                log.debug("Text response completed")
                await outputs.emit("chunk", Chunk(content="", done=True))
            elif isinstance(event, ResponseAudioDoneEvent):
                log.debug("Audio response completed")
                await outputs.emit(
                    "chunk", Chunk(content="", done=True, content_type="audio")
                )
            elif isinstance(event, ResponseDoneEvent):
                try:
                    current_response_id = getattr(event.response, "id", None)
                except Exception:
                    current_response_id = None
                log.info(
                    f"Response completed: status={event.response.status}, id={current_response_id}"
                )
                if event.response.status == "cancelled":
                    # Treat turn detection as an interrupt, not a fatal error
                    reason = None
                    try:
                        status_details = getattr(event.response, "status_details", None)
                        if isinstance(status_details, dict):
                            reason = status_details.get("reason")
                        else:
                            reason = getattr(status_details, "reason", None)
                    except Exception:
                        reason = None

                    if reason == "turn_detected":
                        log.debug("Turn detected, flushing accumulated content")
                        # Flush any accumulated content and finalize this turn
                        if len(audio_accum) > 0:
                            await outputs.emit(
                                "audio", AudioRef(data=bytes(audio_accum))
                            )
                        final_text = "".join(full_text_parts)
                        if final_text.strip():
                            await outputs.emit("text", final_text)
                        full_text_parts = []
                        audio_accum = bytearray()
                        shared["pending"] = shared.get("pending", 0) - 1
                        if (
                            shared.get("done_producing")
                            and shared.get("pending", 0) <= 0
                        ):
                            log.debug("Consumer loop finished (turn detected)")
                            break
                        # Continue consuming further events without raising
                        continue
                    else:
                        log.error(f"Realtime response cancelled: {reason or 'unknown'}")
                        raise RuntimeError(
                            f"Realtime response cancelled: {reason or 'unknown'}"
                        )
                if len(audio_accum) > 0:
                    await outputs.emit("audio", AudioRef(data=bytes(audio_accum)))
                final_text = "".join(full_text_parts)
                if final_text.strip():
                    await outputs.emit("text", final_text)
                full_text_parts = []
                audio_accum = bytearray()
                shared["pending"] = shared.get("pending", 0) - 1
                if shared.get("done_producing") and shared.get("pending", 0) <= 0:
                    log.debug("Consumer loop finished (normal completion)")
                    break
            elif isinstance(event, ErrorEvent):
                msg = event.error or str(event)
                log.error(f"Realtime error: {msg}")
                raise RuntimeError(f"Realtime error: {msg}")

    async def _encode_audio_pcm16_chunks(
        self, context: ProcessingContext, audio: AudioRef
    ) -> list[str]:
        """Encode an audio asset to base64 PCM16 chunk(s).

        Args:
            context (ProcessingContext): Workflow execution context.
            audio (AudioRef): Audio asset to encode.

        Returns:
            list[str]: Base64-encoded PCM16 chunk(s).
        """
        if not audio or audio.is_empty():
            return []
        audio_segment = await context.audio_to_audio_segment(audio)
        with io.BytesIO() as buf:
            audio_segment.export(buf, format="s16le")
            b = buf.getvalue()
        return [base64.b64encode(b).decode("utf-8")]

    async def run(
        self,
        context: ProcessingContext,
        inputs: NodeInputs,
        outputs: NodeOutputs,
    ) -> None:
        """Run the realtime agent with streaming input/output and tools.

        Args:
            context (ProcessingContext): Workflow execution context.
            inputs (NodeInputs): Streaming inputs (text/audio/chunk).
            outputs (NodeOutputs): Output emitter for streaming chunks and artifacts.
        """
        log.info(f"Starting RealtimeAgent with model: {self.model.value}")
        from openai import AsyncOpenAI  # Official SDK v1

        env = Environment.get_environment()
        api_key = env.get("OPENAI_API_KEY")
        if not api_key:
            log.error("OPENAI_API_KEY is not set in environment/secrets")
            raise ValueError("OPENAI_API_KEY is not set in environment/secrets")

        client = AsyncOpenAI(api_key=api_key)
        log.debug("OpenAI client initialized")

        async with client.beta.realtime.connect(model="gpt-realtime") as connection:
            log.info("Connected to OpenAI Realtime API")
            # Resolve and format tools for the realtime session
            tools_instances: list[Tool] = self._resolve_tools(context)
            log.debug("Updating session with tools and configuration")
            # Use dict to include tools; cast to Any to appease strict SDK types
            await connection.session.update(
                session=Session(
                    input_audio_format="pcm16",
                    output_audio_format="pcm16",
                    turn_detection=TurnDetection(
                        type="semantic_vad",
                        create_response=True,
                        eagerness="auto",
                        interrupt_response=True,
                    ),  # pyright: ignore[reportArgumentType]
                    instructions=self.system or "",
                    model=self.model.value,
                    modalities=["text", "audio"],
                    tools=self._format_session_tools_for_realtime(
                        tools_instances
                    ),  # pyright: ignore[reportArgumentType]
                    tool_choice="auto",
                    speed=self.speed,
                    temperature=self.temperature,
                )
            )
            log.info("Session updated successfully")

            # Refactored execution using helper loops
            shared = {"pending": 0, "done_producing": False}
            log.info("Starting producer and consumer loops")
            await asyncio.gather(
                self._producer_loop(connection, inputs, context, shared),
                self._consumer_loop(
                    connection,
                    context,
                    outputs,
                    tools_instances,
                    shared,
                ),
            )
            log.info("RealtimeAgent execution completed")
            return


class RealtimeTranscription(BaseNode):
    """
    Stream microphone or audio input to OpenAI Realtime and emit transcription.

    Emits:
      - `chunk` Chunk(content=..., done=False) for transcript deltas
      - `chunk` Chunk(content="", done=True) to mark segment end
      - `text` final aggregated transcript when input ends
    """

    @classmethod
    def is_cacheable(cls) -> bool:
        return False

    @classmethod
    def return_type(cls):
        return {
            "text": str,
            "chunk": Chunk,
        }

    @classmethod
    def is_streaming_output(cls) -> bool:
        return True

    @classmethod
    def is_streaming_input(cls) -> bool:
        return True

    model: LanguageModel = Field(default=LanguageModel(), description="Model to use")
    system: str = Field(default="", description="System instructions (optional)")
    temperature: float = Field(default=0.8, description="Decoding temperature")

    async def _trigger_response(
        self, connection: AsyncRealtimeConnection, shared: dict
    ) -> None:
        await connection.response.create(response=Response())
        shared["pending"] = shared.get("pending", 0) + 1

    async def _producer_loop(
        self,
        connection: AsyncRealtimeConnection,
        inputs: NodeInputs,
        context: ProcessingContext,
        shared: dict,
    ) -> None:
        async for handle, item in inputs.any():
            if handle != "chunk":
                raise ValueError(f"Unknown handle: {handle}")
            assert isinstance(item, Chunk)
            if item.content_type == "audio":
                if item.content:
                    await connection.input_audio_buffer.append(audio=item.content)
                if getattr(item, "done", False):
                    await self._trigger_response(connection, shared)
            else:
                # Ignore non-audio chunks for transcription-only
                if item.done:
                    await self._trigger_response(connection, shared)
        shared["done_producing"] = True

    async def _consumer_loop(
        self,
        connection: AsyncRealtimeConnection,
        context: ProcessingContext,
        outputs: NodeOutputs,
        shared: dict,
    ) -> None:
        log.debug("Transcription consumer loop started")
        full_text_parts: list[str] = []
        aggregated: list[str] = []
        async for event in connection:
            event_type = getattr(event, "type", "unknown")
            log.debug(f"Transcription event: {event_type}")

            if hasattr(event, "type") and event.type in (
                "response.audio_transcript.delta",
            ):
                delta = getattr(event, "delta", None)
                if not delta and hasattr(event, "model_dump"):
                    try:
                        delta = event.model_dump().get("delta")  # type: ignore[attr-defined]
                    except Exception:
                        delta = None
                if delta:
                    full_text_parts.append(delta)
                    log.debug(f"Transcript delta: {len(delta)} characters")
                    await outputs.emit("chunk", Chunk(content=delta, done=False))
            elif hasattr(event, "type") and event.type in (
                "response.audio_transcript.done",
                "response.text.done",
            ):
                # End of a segment; emit a done marker for streaming consumers
                log.debug("Transcript segment completed")
                await outputs.emit("chunk", Chunk(content="", done=True))
            elif hasattr(event, "type") and event.type == "response.done":
                # Handle turn detection cancellations gracefully
                try:
                    status = getattr(event.response, "status", None)
                except Exception:
                    status = None
                if status == "cancelled":
                    reason = None
                    try:
                        details = getattr(event.response, "status_details", None)
                        if isinstance(details, dict):
                            reason = details.get("reason")
                        else:
                            reason = getattr(details, "reason", None)
                    except Exception:
                        reason = None
                    if reason == "turn_detected":
                        segment = "".join(full_text_parts).strip()
                        if segment:
                            aggregated.append(segment)
                        full_text_parts = []
                        shared["pending"] = shared.get("pending", 0) - 1
                        if (
                            shared.get("done_producing")
                            and shared.get("pending", 0) <= 0
                        ):
                            break
                        continue
                    else:
                        raise RuntimeError(
                            f"Realtime response cancelled: {reason or 'unknown'}"
                        )

                # Normal completion: finalize segment
                segment = "".join(full_text_parts).strip()
                if segment:
                    aggregated.append(segment)
                full_text_parts = []
                shared["pending"] = shared.get("pending", 0) - 1
                if shared.get("done_producing") and shared.get("pending", 0) <= 0:
                    break
            elif hasattr(event, "type") and event.type == "error":
                from openai.types.beta.realtime.error_event import ErrorEvent as _Err

                if isinstance(event, _Err):
                    raise RuntimeError(f"Realtime error: {event.error}")
                raise RuntimeError("Realtime error")

        # Emit final aggregated transcript
        final_text = " ".join(aggregated).strip()
        if final_text:
            log.info(f"Emitting final transcript: {len(final_text)} characters")
            await outputs.emit("text", final_text)
        log.debug("Transcription consumer loop completed")

    async def run(
        self,
        context: ProcessingContext,
        inputs: NodeInputs,
        outputs: NodeOutputs,
    ) -> None:
        log.info("Starting RealtimeTranscription")
        from openai import AsyncOpenAI

        env = Environment.get_environment()
        api_key = env.get("OPENAI_API_KEY")
        if not api_key:
            log.error("OPENAI_API_KEY is not set in environment/secrets")
            raise ValueError("OPENAI_API_KEY is not set in environment/secrets")

        client = AsyncOpenAI(api_key=api_key)
        log.debug("OpenAI client initialized for transcription")

        async with client.beta.realtime.connect(model="gpt-realtime") as connection:
            log.info("Connected to OpenAI Realtime API for transcription")
            log.debug("Updating transcription session")
            await connection.session.update(
                session=Session(
                    input_audio_format="pcm16",
                    output_audio_format="pcm16",
                    turn_detection=TurnDetection(
                        type="server_vad",
                        create_response=True,
                        eagerness="low",
                        interrupt_response=True,
                    ),  # pyright: ignore[reportArgumentType]
                    instructions=self.system or "",
                    model=(
                        self.model.id or "gpt-4o-mini-realtime-preview"
                    ),  # pyright: ignore[reportArgumentType]
                    modalities=["audio", "text"],
                    temperature=self.temperature,
                )
            )
            log.info("Transcription session updated successfully")

            shared = {"pending": 0, "done_producing": False}
            log.info("Starting transcription producer and consumer loops")
            await asyncio.gather(
                self._producer_loop(connection, inputs, context, shared),
                self._consumer_loop(connection, context, outputs, shared),
            )
            log.info("RealtimeTranscription execution completed")
            return
