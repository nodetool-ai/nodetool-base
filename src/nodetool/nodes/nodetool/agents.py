import base64
import asyncio
import json
import re
from typing import Any, AsyncGenerator, cast, ClassVar, TypedDict

from nodetool.agents.tools.workflow_tool import GraphTool
from nodetool.workflows.graph_utils import find_node, get_downstream_subgraph
from pydantic import Field

from nodetool.agents.tools.base import Tool
from nodetool.chat.providers import get_provider

from nodetool.workflows.types import (
    ToolCallUpdate,
)

from nodetool.metadata.types import (
    LanguageModel,
    ToolName,
    ImageRef,
    AudioRef,
    Message,
    MessageTextContent,
    MessageImageContent,
    MessageAudioContent,
    MessageContent,
    ImageRef,
    ToolName,
    ToolCall,
    AudioRef,
    LanguageModel,
)
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import ToolCallUpdate, EdgeUpdate
from nodetool.workflows.io import NodeInputs, NodeOutputs
from nodetool.chat.providers import Chunk
from nodetool.metadata.types import Provider
from nodetool.chat.providers import get_provider
from nodetool.config.logging_config import get_logger

log = get_logger(__name__)
# Log level is controlled by env (DEBUG/NODETOOL_LOG_LEVEL)


class Summarizer(BaseNode):
    """
    Generate concise summaries of text content using LLM providers with streaming output.
    text, summarization, nlp, content, streaming

    Specialized for creating high-quality summaries with real-time streaming:
    - Condensing long documents into key points
    - Creating executive summaries with live output
    - Extracting main ideas from text as they're generated
    - Maintaining factual accuracy while reducing length
    """

    @classmethod
    def is_cacheable(cls) -> bool:
        return False

    class OutputType(TypedDict):
        text: str | None
        chunk: Chunk | None

    system_prompt: str = Field(
        default="""
        You are an expert summarizer. Your task is to create clear, accurate, and concise summaries using Markdown for structuring. 
        Follow these guidelines:
        1. Identify and include only the most important information.
        2. Maintain factual accuracy - do not add or modify information.
        3. Use clear, direct language.
        4. Aim for approximately {self.max_tokens} tokens.
        """,
        description="The system prompt for the summarizer",
    )

    model: LanguageModel = Field(
        default=LanguageModel(),
        description="Model to use for summarization",
    )
    text: str = Field(default="", description="The text to summarize")
    image: ImageRef = Field(
        default=ImageRef(),
        description="Optional image to condition the summary",
    )
    audio: AudioRef = Field(
        default=AudioRef(),
        description="Optional audio to condition the summary",
    )
    max_tokens: int = Field(
        default=200,
        description="Target maximum number of tokens for the summary",
        ge=50,
        le=16384,
    )
    context_window: int = Field(
        title="Context Window (Ollama)", default=4096, ge=1, le=65536
    )

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["text", "max_tokens", "model", "image", "audio"]

    async def gen_process(
        self, context: ProcessingContext
    ) -> AsyncGenerator[OutputType, None]:
        if self.model.provider == Provider.Empty:
            raise ValueError("Select a model")

        content: list[MessageContent] = []
        content.append(MessageTextContent(text=self.text))
        if self.image.is_set():
            content.append(MessageImageContent(image=self.image))
        if self.audio.is_set():
            content.append(MessageAudioContent(audio=self.audio))

        messages = [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content=content),
        ]

        text = ""

        provider = get_provider(self.model.provider)
        async for chunk in provider.generate_messages(
            messages=messages,
            model=self.model.id,
            max_tokens=self.max_tokens,
            context_window=self.context_window,
        ):
            if isinstance(chunk, Chunk):
                if chunk.content_type == "text" or chunk.content_type is None:
                    yield {"chunk": chunk, "text": None}
                text += chunk.content

        yield {"text": text, "chunk": None}


DEFAULT_EXTRACTOR_SYSTEM_PROMPT = """
You are a precise structured data extractor.

Goal
- Extract exactly the fields described in <JSON_SCHEMA> from the content in <TEXT> (and any attached media).

Output format (MANDATORY)
- Output exactly ONE fenced code block labeled json containing ONLY the JSON object:

  ```json
  { ...single JSON object matching <JSON_SCHEMA>... }
  ```

- No additional prose before or after the block.

Extraction rules
- Use only information found in <TEXT> or attached media. Do not invent facts.
- Preserve source values; normalize internal whitespace and trim leading/trailing spaces.
- If a required field is missing or not explicitly stated, return the closest reasonable default consistent with its type:
  - string: ""
  - number: 0
  - boolean: false
  - array/object: empty value of that type (only if allowed by the schema)
- Dates/times: prefer ISO 8601 when the schema type is string and the value represents a date/time.
- If multiple candidates exist, choose the most precise and unambiguous one.

Validation
- Ensure the final JSON validates against <JSON_SCHEMA> exactly.
"""


class Extractor(BaseNode):
    """
    Extract structured data from text content using LLM providers.
    data-extraction, structured-data, nlp, parsing

    Specialized for extracting structured information:
    - Converting unstructured text into structured data
    - Identifying and extracting specific fields from documents
    - Parsing text according to predefined schemas
    - Creating structured records from natural language content
    """

    _supports_dynamic_outputs: ClassVar[bool] = True

    @classmethod
    def get_title(cls) -> str:
        return "Extractor"

    system_prompt: str = Field(
        default=DEFAULT_EXTRACTOR_SYSTEM_PROMPT,
        description="The system prompt for the data extractor",
    )

    model: LanguageModel = Field(
        default=LanguageModel(),
        description="Model to use for data extraction",
    )
    text: str = Field(default="", description="The text to extract data from")
    image: ImageRef = Field(
        default=ImageRef(),
        description="Optional image to assist extraction",
    )
    audio: AudioRef = Field(
        default=AudioRef(),
        description="Optional audio to assist extraction",
    )
    max_tokens: int = Field(
        default=4096,
        ge=1,
        le=16384,
        description="The maximum number of tokens to generate.",
    )
    context_window: int = Field(
        title="Context Window (Ollama)", default=4096, ge=1, le=65536
    )

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["text", "model", "image", "audio"]

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        import json

        if self.model.provider == Provider.Empty:
            raise ValueError("Select a model")

        provider = get_provider(self.model.provider)

        # Build JSON schema from instance dynamic outputs (default each to string)
        output_slots = self.get_dynamic_output_slots()

        if len(output_slots) == 0:
            raise ValueError("Declare outputs for the fields you want to extract")

        properties: dict[str, Any] = {
            slot.name: slot.type.get_json_schema() for slot in output_slots
        }
        required: list[str] = [slot.name for slot in output_slots]

        schema = {
            "type": "object",
            "title": "Extraction Results",
            "additionalProperties": False,
            "properties": properties,
            "required": required,
        }

        additional_instructions = (
            "\n<JSON_SCHEMA>\n" + json.dumps(schema, indent=2) + "\n</JSON_SCHEMA>\n"
        )
        if self.image.is_set():
            additional_instructions += (
                "Use the attached image to assist with the extraction.\n"
            )
        if self.audio.is_set():
            additional_instructions += (
                "Use the attached audio to assist with the extraction.\n"
            )

        user_content: list[MessageContent] = [
            MessageTextContent(text=self.text),
        ]

        if self.image.is_set():
            user_content.append(MessageImageContent(image=self.image))
        if self.audio.is_set():
            user_content.append(MessageAudioContent(audio=self.audio))

        messages = [
            Message(
                role="system", content=self.system_prompt + additional_instructions
            ),
            Message(role="user", content=user_content),
        ]

        # Prefer model to emit a fenced ```json block; fall back to curly-brace extraction
        assistant_message = await provider.generate_message(
            model=self.model.id,
            messages=messages,
            max_tokens=self.max_tokens,
            context_window=self.context_window,
        )

        raw = str(assistant_message.content or "").strip()
        # 1) Try to extract content from a ```json ... ``` fenced block
        fenced_match = None
        try:
            fenced_match = re.search(r"```json\s*([\s\S]*?)\s*```", raw, re.IGNORECASE)
        except Exception:
            fenced_match = None
        if fenced_match:
            candidate = fenced_match.group(1).strip()
            try:
                result_obj = json.loads(candidate)
            except Exception:
                # fall through to brace extraction below
                result_obj = None  # type: ignore[assignment]
        else:
            result_obj = None  # type: ignore[assignment]

        # 2) Fallback: extract minimal substring between first { and last }
        if result_obj is None:
            try:
                start = raw.find("{")
                end = raw.rfind("}")
                if 0 <= start < end:
                    snippet = raw[start : end + 1]
                    result_obj = json.loads(snippet)
            except Exception:
                result_obj = None  # type: ignore[assignment]

        if not isinstance(result_obj, dict):
            raise ValueError("Extractor did not return a dictionary")
        return result_obj


DEFAULT_CLASSIFY_SYSTEM_PROMPT = """
You are a precise classifier.

Goal
- Select exactly one category from the list provided by the user.

Output format (MANDATORY)
- Return ONLY a single JSON object with this exact schema and nothing else:
  {"category": "<one-of-the-allowed-categories>"}
- No prose, no Markdown, no code fences, no explanations, no extra keys.

Selection criteria
- Choose the single best category that captures the main intent of the text.
- If multiple categories seem plausible, pick the most probable one; do not return multiple.
- If none fit perfectly, choose the closest allowed category. If the list includes "Other" or "Unknown", prefer it when appropriate.
- Be robust to casing, punctuation, emojis, and minor typos. Handle negation correctly (e.g., "not spam" ≠ spam).
- Never invent categories that are not in the provided list.

Behavior
- Be deterministic for the same input.
- Do not ask clarifying questions; make the best choice with what's given.
"""


class Classifier(BaseNode):
    """
    Classify text into predefined or dynamic categories using LLM.
    classification, nlp, categorization

    Use cases:
    - Sentiment analysis
    - Topic classification
    - Intent detection
    - Content categorization
    """

    @classmethod
    def get_title(cls) -> str:
        return "Classifier"

    system_prompt: str = Field(
        default=DEFAULT_CLASSIFY_SYSTEM_PROMPT,
        description="The system prompt for the classifier",
    )

    model: LanguageModel = Field(
        default=LanguageModel(),
        description="Model to use for classification",
    )
    text: str = Field(default="", description="Text to classify")
    image: ImageRef = Field(
        default=ImageRef(),
        description="Optional image to classify in context",
    )
    audio: AudioRef = Field(
        default=AudioRef(),
        description="Optional audio to classify in context",
    )
    categories: list[str] = Field(
        default=[],
        description="List of possible categories. If empty, LLM will determine categories.",
    )
    max_tokens: int = Field(
        default=1024,
        ge=1,
        le=16384,
        description="The maximum number of tokens to generate.",
    )
    context_window: int = Field(
        title="Context Window (Ollama)", default=4096, ge=1, le=65536
    )

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["text", "categories", "model", "image", "audio"]

    async def process(self, context: ProcessingContext) -> str:
        if self.model.provider == Provider.Empty:
            raise ValueError("Select a model")

        if len(self.categories) < 2:
            raise ValueError("At least 2 categories are required")

        # Build messages instructing the model to pick a category and emit strict JSON
        user_text = (
            "Classify the given input into exactly one category from the list.\n"
            f"Allowed categories: {', '.join(self.categories)}\n\n"
            f"Text: {self.text}"
        )
        user_content: list[MessageContent] = [MessageTextContent(text=user_text)]
        if self.image.is_set():
            user_content.append(MessageImageContent(image=self.image))
        if self.audio.is_set():
            user_content.append(MessageAudioContent(audio=self.audio))

        messages = [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content=user_content),
        ]

        # Response schema forcing a single enum field
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["category"],
            "properties": {
                "category": {
                    "type": "string",
                    "enum": self.categories,
                    "description": "One of the allowed categories",
                }
            },
        }

        provider = get_provider(self.model.provider)

        try:
            assistant_message = await provider.generate_message(
                model=self.model.id,
                messages=messages,
                context_window=self.context_window,
                max_tokens=self.max_tokens,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "classification_result",
                        "schema": schema,
                        "strict": True,
                    },
                },
            )
            raw = str(assistant_message.content)
        except Exception as e:
            # Fall back to a best-effort mapping using the model without response_format
            log.debug(f"Classifier: response_format failed ({e}); retrying without it")
            assistant_message = await provider.generate_message(
                model=self.model.id,
                messages=messages,
                context_window=self.context_window,
            )
            raw = str(assistant_message.content)

        # Parse robustly, then validate and coerce to an allowed category
        category = _parse_or_infer_category(raw, self.categories)
        return category


def _parse_or_infer_category(raw: str, categories: list[str]) -> str:
    """Parse JSON {"category": ...} or infer from free text, always returning a valid category.

    Strategy:
    1) Try strict JSON parse for key "category"
    2) Try to extract minimal JSON object substring and parse again
    3) Try to find a direct category mention in the text (case-insensitive)
    4) Fallback to similarity match using difflib
    5) Final fallback: return "Other"/"Unknown" if present; otherwise first category
    """
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            value = obj.get("category")
            if isinstance(value, str):
                for c in categories:
                    if value.strip().lower() == c.strip().lower():
                        return c
    except Exception:
        pass

    # Attempt to extract a JSON object substring
    try:
        start = raw.find("{")
        end = raw.rfind("}")
        if 0 <= start < end:
            snippet = raw[start : end + 1]
            obj = json.loads(snippet)
            if isinstance(obj, dict):
                value = obj.get("category")
                if isinstance(value, str):
                    for c in categories:
                        if value.strip().lower() == c.strip().lower():
                            return c
    except Exception:
        pass

    # Direct mention match
    lowered = raw.lower()
    for c in categories:
        if c.lower() in lowered:
            return c

    # Similarity match
    try:
        import difflib

        best = None
        best_score = 0.0
        for c in categories:
            score = difflib.SequenceMatcher(
                None, lowered.strip(), c.lower().strip()
            ).ratio()
            if score > best_score:
                best = c
                best_score = score
        if best is not None and best_score >= 0.5:
            return best
    except Exception:
        pass

    # Final fallback
    for fallback in ("other", "unknown"):
        for c in categories:
            if c.strip().lower() == fallback:
                return c
    return categories[0]


DEFAULT_SYSTEM_PROMPT = """You are a an AI agent. 

Behavior
- Understand the user's intent and the context of the task.
- Break down the task into smaller steps.
- Be precise, concise, and actionable.
- Use tools to accomplish your goal. 

Tool preambles
- Outline the next step(s) you will perform.
- After acting, summarize the outcome.

Rendering
- Use Markdown to display media assets.
- Display images, audio, and video assets using the appropriate Markdown.

File handling
- Inputs and outputs are files in the /workspace directory.
- Write outputs of code execution to the /workspace directory.
"""


def serialize_tool_result(tool_result: Any) -> Any:
    """
    Serialize a tool result to a JSON-serializable object.
    """
    try:
        if isinstance(tool_result, dict):
            return {k: serialize_tool_result(v) for k, v in tool_result.items()}
        if isinstance(tool_result, list):
            return [serialize_tool_result(v) for v in tool_result]
        if isinstance(tool_result, (bytes, bytearray)):
            import base64

            return {
                "__type__": "bytes",
                "base64": base64.b64encode(tool_result).decode("utf-8"),
            }
        # Pydantic/BaseModel or BaseType
        if getattr(tool_result, "model_dump", None) is not None:
            return tool_result.model_dump()
        # Handle set/tuple
        if isinstance(tool_result, (set, tuple)):
            return [serialize_tool_result(v) for v in tool_result]
        # Numpy types
        try:
            import numpy as np  # type: ignore

            if isinstance(tool_result, np.ndarray):
                return tool_result.tolist()
            # numpy scalar types
            if isinstance(tool_result, (np.integer,)):
                return int(tool_result)
            if isinstance(tool_result, (np.floating,)):
                return float(tool_result)
            if isinstance(tool_result, (np.bool_,)):
                return bool(tool_result)
            # generic fallback, including np.generic subclasses
            if isinstance(tool_result, np.generic):
                try:
                    return tool_result.item()
                except Exception:
                    pass
            # datetime/timedelta
            if isinstance(tool_result, np.datetime64):
                try:
                    return np.datetime_as_string(tool_result, timezone="naive")
                except Exception:
                    return str(tool_result)
            if isinstance(tool_result, np.timedelta64):
                return str(tool_result)
        except Exception:
            pass
        # Pandas types
        try:
            import pandas as pd  # type: ignore

            if isinstance(tool_result, pd.DataFrame):
                return tool_result.to_dict(orient="records")
            if isinstance(tool_result, pd.Series):
                return tool_result.tolist()
            if isinstance(tool_result, pd.Index):
                return tool_result.tolist()
            if isinstance(tool_result, pd.Timestamp):
                return tool_result.isoformat()
            if isinstance(tool_result, pd.Timedelta):
                return str(tool_result)
            # pandas NA scalar
            if tool_result is pd.NA:  # type: ignore[attr-defined]
                return None
        except Exception:
            pass
        # Fallback: make it a string
        return tool_result
    except Exception:
        # Absolute fallback to string to avoid breaking the agent loop
        return str(tool_result)


class Agent(BaseNode):
    """
    Generate natural language responses using LLM providers and streams output.
    llm, text-generation, chatbot, question-answering, streaming
    """

    model: LanguageModel = Field(
        default=LanguageModel(),
        description="Model to use for execution",
    )
    system: str = Field(
        title="System",
        default=DEFAULT_SYSTEM_PROMPT,
        description="The system prompt for the LLM",
    )
    prompt: str = Field(
        title="Prompt",
        default="",
        description="The prompt for the LLM",
    )
    image: ImageRef = Field(
        title="Image",
        default=ImageRef(),
        description="The image to analyze",
    )
    audio: AudioRef = Field(
        title="Audio",
        default=AudioRef(),
        description="The audio to analyze",
    )
    history: list[Message] = Field(
        title="Messages", default=[], description="The messages for the LLM"
    )
    max_tokens: int = Field(title="Max Tokens", default=8192, ge=1, le=100000)
    context_window: int = Field(
        title="Context Window (Ollama)", default=4096, ge=1, le=65536
    )

    _supports_dynamic_outputs: ClassVar[bool] = True

    def should_route_output(self, output_name: str) -> bool:
        """
        Do not route dynamic outputs; they represent tool entry points.
        Still route declared outputs like 'text', 'chunk', 'audio'.
        """
        return output_name not in self._dynamic_outputs

    @classmethod
    def is_cacheable(cls) -> bool:
        return False

    class OutputType(TypedDict, total=False):
        text: str
        chunk: Chunk
        audio: AudioRef

    def _resolve_tools(self, context: ProcessingContext) -> list[Tool]:
        tools = []
        for name, type_meta in self._dynamic_outputs.items():
            initial_edges, graph = get_downstream_subgraph(context.graph, self.id, name)
            initial_nodes = [find_node(graph, edge.target) for edge in initial_edges]
            nodes = graph.nodes
            if len(nodes) == 0:
                continue
            tool = GraphTool(graph, name, "", initial_edges, initial_nodes)
            tools.append(tool)
        return tools

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return [
            "prompt",
            "model",
            "image",
            "audio",
        ]

    @classmethod
    def is_streaming_output(cls) -> bool:
        return True

    @classmethod
    def is_streaming_input(cls) -> bool:
        return True

    def _prepare_messages(self) -> list[Message]:
        """Build the message history for a single model interaction.

        Returns:
            list[Message]: Ordered list of messages including system, prior history,
                and the current user content (text/image/audio).
        """
        content = []
        content.append(MessageTextContent(text=self.prompt))
        if self.image.is_set():
            content.append(MessageImageContent(image=self.image))
        if self.audio.is_set():
            content.append(MessageAudioContent(audio=self.audio))

        messages: list[Message] = [
            Message(role="system", content=self.system),
        ]
        for message in self.history:
            messages.append(message)
        messages.append(Message(role="user", content=content))
        log.debug(
            "Agent initial messages prepared: num_messages=%d has_image=%s has_audio=%s prompt_len=%d",
            len(messages),
            self.image.is_set(),
            self.audio.is_set(),
            len(self.prompt or ""),
        )
        return messages

    async def _execute_agent_loop(
        self,
        context: ProcessingContext,
        messages: list[Message],
        tools: list[Tool],
        outputs: NodeOutputs,
    ) -> None:
        """Execute one or more model iterations, streaming chunks and handling tools.

        Args:
            context (ProcessingContext): Workflow execution context.
            messages (list[Message]): Message history to seed the model.
            tools (list[Tool]): Resolved tools available for tool-calling.
            outputs (NodeOutputs): Output emitter for streamed chunks and results.
        """
        tools_called = False
        first_time = True
        iteration = 0
        tool_iterations = 0

        while tools_called or first_time:
            iteration += 1
            log.debug(
                "Agent loop start (per-item): iteration=%d tools_called_prev=%s first_time=%s num_messages=%d",
                iteration,
                tools_called,
                first_time,
                len(messages),
            )
            tools_called = False
            first_time = False
            message_text_content = MessageTextContent(text="")
            assistant_message = Message(
                role="assistant",
                content=[message_text_content],
                tool_calls=[],
            )

            provider = get_provider(self.model.provider)
            pending_tool_tasks: list[asyncio.Task] = []
            async for chunk in provider.generate_messages(
                messages=messages,
                model=self.model.id,
                tools=tools,
                max_tokens=self.max_tokens,
                context_window=self.context_window,
            ):
                if messages[-1] != assistant_message:
                    messages.append(assistant_message)
                if isinstance(chunk, Chunk):
                    if chunk.content_type in ("text", None):
                        message_text_content.text += chunk.content
                        await outputs.emit("chunk", chunk)
                        if chunk.done:
                            log.debug(
                                "Agent chunk done (per-item): iteration=%d text_len=%d",
                                iteration,
                                len(message_text_content.text),
                            )
                            await outputs.emit("text", message_text_content.text)
                    elif chunk.content_type == "audio":
                        audio_bytes = base64.b64decode(chunk.content or "")
                        audio_ref = AudioRef(data=audio_bytes)
                        await outputs.emit("audio", audio_ref)
                    else:
                        log.warning(
                            "Agent received unsupported chunk type %s; ignoring",
                            chunk.content_type,
                        )

                    if chunk.done:
                        await outputs.emit("text", message_text_content.text)

                elif isinstance(chunk, ToolCall):
                    tools_called = True
                    try:
                        args_preview = (
                            (
                                json.dumps(chunk.args)[:500]
                                + ("…" if len(json.dumps(chunk.args)) > 500 else "")
                            )
                            if chunk.args is not None
                            else None
                        )
                    except Exception:
                        args_preview = "<unserializable>"
                    log.debug(
                        "Agent tool call (per-item): iteration=%d id=%s name=%s has_args=%s args_preview=%s",
                        iteration,
                        getattr(chunk, "id", None),
                        getattr(chunk, "name", None),
                        chunk.args is not None,
                        args_preview,
                    )
                    assert assistant_message.tool_calls is not None
                    assistant_message.tool_calls.append(chunk)
                    for tool_instance in tools:
                        if tool_instance and tool_instance.name == chunk.name:
                            context.post_message(
                                ToolCallUpdate(
                                    node_id=self.id,
                                    name=chunk.name,
                                    args=chunk.args,
                                    message=tool_instance.user_message(chunk.args),
                                )
                            )
                            # Mark edges as message_sent after tool call is announced
                            initial_edges, _ = get_downstream_subgraph(
                                context.graph, self.id, chunk.name
                            )
                            for e in initial_edges:
                                context.post_message(
                                    EdgeUpdate(
                                        edge_id=e.id or "",
                                        status="message_sent",
                                    )
                                )

                            async def _run_tool(instance: Tool, call: ToolCall):
                                try:
                                    result = await instance.process(context, call.args)
                                    result_json = json.dumps(
                                        serialize_tool_result(result)
                                    )
                                    log.debug(
                                        "Agent tool result (parallel per-item): iteration=%d id=%s name=%s result_len=%d",
                                        iteration,
                                        getattr(call, "id", None),
                                        getattr(call, "name", None),
                                        len(result_json),
                                    )
                                except Exception as e:
                                    log.error(
                                        f"Tool call {call.id} ({call.name}) failed with exception: {e}"
                                    )
                                    result_json = json.dumps(
                                        {"error": f"Error executing tool: {str(e)}"}
                                    )
                                    log.debug(
                                        "Agent tool result error recorded (parallel per-item): iteration=%d id=%s name=%s",
                                        iteration,
                                        getattr(call, "id", None),
                                        getattr(call, "name", None),
                                    )
                                return call.id, call.name, result_json

                            pending_tool_tasks.append(
                                asyncio.create_task(_run_tool(tool_instance, chunk))
                            )
                            break
            if pending_tool_tasks:
                log.debug(
                    "Agent executing %d tool call(s) in parallel for iteration=%d (per-item)",
                    len(pending_tool_tasks),
                    iteration,
                )
                results = await asyncio.gather(*pending_tool_tasks)
                for tool_call_id, tool_name, tool_result_json in results:
                    initial_edges, _ = get_downstream_subgraph(
                        context.graph, self.id, tool_name
                    )
                    for e in initial_edges:
                        context.post_message(
                            EdgeUpdate(edge_id=e.id or "", status="drained")
                        )
                    messages.append(
                        Message(
                            role="tool",
                            tool_call_id=tool_call_id,
                            name=tool_name,
                            content=tool_result_json,
                        )
                    )
            log.debug(
                "Agent loop end (per-item): iteration=%d will_continue=%s assistant_has_tool_calls=%s assistant_text_len=%d total_messages=%d",
                iteration,
                tools_called,
                assistant_message.tool_calls is not None
                and len(assistant_message.tool_calls) > 0,
                len(message_text_content.text),
                len(messages),
            )
        log.debug(
            "Agent loop complete (per-item): iteration=%d",
            iteration,
        )

    async def run(
        self,
        context: ProcessingContext,
        inputs: NodeInputs,
        outputs: NodeOutputs,
    ) -> None:
        if self.model.provider == Provider.Empty:
            raise ValueError("Select a model")
        # If this Agent has no inbound edges, execute once using configured properties.
        # This enables simple graphs (Agent → Preview) to run without an explicit trigger.
        try:
            has_inbound = any(e.target == self.id for e in context.graph.edges)
        except Exception:
            has_inbound = False
        if not has_inbound:
            messages = self._prepare_messages()
            tools = self._resolve_tools(context)
            tool_names = [t.name for t in tools if t is not None]
            log.debug(
                "Agent setup (fallback no-input): model=%s provider=%s context_window=%s max_tokens=%s tools=%s",
                self.model.id,
                self.model.provider,
                self.context_window,
                self.max_tokens,
                tool_names,
            )
            await self._execute_agent_loop(context, messages, tools, outputs)
            return
        # Accumulators for streamed chunk input
        chunk_text_buf: list[str] = []
        audio_accum = bytearray()

        # Consume streaming input and run one agent execution when a logical unit completes
        async for handle, item in inputs.any():
            # Special handling for streamed chunk input
            if handle == "chunk" and isinstance(item, Chunk):
                if item.content_type == "audio":
                    if item.content:
                        try:
                            audio_accum.extend(base64.b64decode(item.content))
                        except Exception:
                            pass
                    if getattr(item, "done", False):
                        # Set accumulated audio and execute once
                        if len(audio_accum) > 0:
                            self.audio = AudioRef(data=bytes(audio_accum))
                        # reset accumulators
                        chunk_text_buf = []
                        audio_accum = bytearray()
                        messages = self._prepare_messages()
                        tools = self._resolve_tools(context)
                        tool_names = [t.name for t in tools if t is not None]
                        log.debug(
                            "Agent setup (chunk-audio): model=%s provider=%s context_window=%s max_tokens=%s tools=%s",
                            self.model.id,
                            self.model.provider,
                            self.context_window,
                            self.max_tokens,
                            tool_names,
                        )
                        await self._execute_agent_loop(
                            context, messages, tools, outputs
                        )
                else:
                    # Treat as text chunk
                    if item.content:
                        chunk_text_buf.append(item.content)
                    if getattr(item, "done", False):
                        self.prompt = "".join(chunk_text_buf)
                        # reset accumulators
                        chunk_text_buf = []
                        audio_accum = bytearray()
                        messages = self._prepare_messages()
                        tools = self._resolve_tools(context)
                        tool_names = [t.name for t in tools if t is not None]
                        log.debug(
                            "Agent setup (chunk-text): model=%s provider=%s context_window=%s max_tokens=%s tools=%s",
                            self.model.id,
                            self.model.provider,
                            self.context_window,
                            self.max_tokens,
                            tool_names,
                        )
                        await self._execute_agent_loop(
                            context, messages, tools, outputs
                        )
                # For intermediate chunks, wait for done signal
                continue

            # Default behavior: assign property and execute immediately per item
            try:
                self.assign_property(handle, item)
            except Exception:
                pass

            messages = self._prepare_messages()
            tools = self._resolve_tools(context)
            tool_names = [t.name for t in tools if t is not None]

            log.debug(
                "Agent setup (per-item): model=%s provider=%s context_window=%s max_tokens=%s tools=%s",
                self.model.id,
                self.model.provider,
                self.context_window,
                self.max_tokens,
                tool_names,
            )

            await self._execute_agent_loop(context, messages, tools, outputs)
