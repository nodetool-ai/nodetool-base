import json
import re
from typing import Any, AsyncGenerator, ClassVar, TypedDict
from nodetool.config.logging_config import get_logger
from pydantic import Field

from nodetool.chat.dataframes import (
    json_schema_for_dataframe,
)
from nodetool.providers import get_provider
from nodetool.metadata.types import (
    Message,
    MessageContent,
    MessageTextContent,
    MessageAudioContent,
    DataframeRef,
    RecordType,
    LanguageModel,
    PlotlyConfig,
    ImageRef,
    AudioRef,
    SVGElement,
    Provider,
    ToolCall,
)
from nodetool.workflows.types import Chunk
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext

from nodetool.chat.dataframes import GenerateDataTool

logger = get_logger(__name__)


DEFAULT_STRUCTURED_OUTPUT_SYSTEM_PROMPT = """
You are a structured data generator focused on JSON outputs.

Goal
- Produce a high-quality JSON object that matches <JSON_SCHEMA> using the guidance in <INSTRUCTIONS> and any supplemental <CONTEXT>.

Output format (MANDATORY)
- Output exactly ONE fenced code block labeled json containing ONLY the JSON object:

  ```json
  { ...single JSON object matching <JSON_SCHEMA>... }
  ```

- No additional prose before or after the block.

Generation rules
- Invent plausible, internally consistent values when not explicitly provided.
- Honor all constraints from <JSON_SCHEMA> (types, enums, ranges, formats).
- Prefer ISO 8601 for dates/times when applicable.
- Ensure numbers respect reasonable magnitudes and relationships described in <INSTRUCTIONS>.
- Avoid referencing external sources; rely solely on the provided guidance.

Validation
- Ensure the final JSON validates against <JSON_SCHEMA> exactly.
"""


class StructuredOutputGenerator(BaseNode):
    """
    Generate structured JSON objects from instructions using LLM providers.
    data-generation, structured-data, json, synthesis

    Specialized for creating structured information:
    - Generating JSON that follows dynamic schemas
    - Fabricating records from requirements and guidance
    - Simulating sample data for downstream workflows
    - Producing consistent structured outputs for testing
    """

    _supports_dynamic_outputs: ClassVar[bool] = True

    @classmethod
    def get_title(cls) -> str:
        return "Structured Output Generator"

    system_prompt: str = Field(
        default=DEFAULT_STRUCTURED_OUTPUT_SYSTEM_PROMPT,
        description="The system prompt guiding JSON generation.",
    )

    model: LanguageModel = Field(
        default=LanguageModel(),
        description="Model to use for structured generation.",
    )
    instructions: str = Field(
        default="",
        description="Detailed instructions for the structured output.",
    )
    context: str = Field(
        default="",
        description="Optional context to ground the generation.",
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
        return ["instructions", "context", "model"]

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        if self.model.provider == Provider.Empty:
            raise ValueError("Select a model")

        provider = get_provider(self.model.provider)

        output_slots = self.get_dynamic_output_slots()
        if len(output_slots) == 0:
            raise ValueError("Declare outputs for the fields you want to generate")

        properties: dict[str, Any] = {
            slot.name: slot.type.get_json_schema() for slot in output_slots
        }
        required: list[str] = [slot.name for slot in output_slots]

        schema = {
            "type": "object",
            "title": "Structured Output Specification",
            "additionalProperties": False,
            "properties": properties,
            "required": required,
        }

        instructions_sections: list[str] = []
        instructions_sections.append(
            "<JSON_SCHEMA>\n" + json.dumps(schema, indent=2) + "\n</JSON_SCHEMA>"
        )
        if self.instructions.strip():
            instructions_sections.append(
                "<INSTRUCTIONS>\n" + self.instructions.strip() + "\n</INSTRUCTIONS>"
            )
        if self.context.strip():
            instructions_sections.append(
                "<CONTEXT>\n" + self.context.strip() + "\n</CONTEXT>"
            )

        additional_instructions = "\n" + "\n".join(instructions_sections) + "\n"

        user_content: list[MessageContent] = [
            MessageTextContent(
                text="Generate a JSON object that satisfies all requirements."
            )
        ]

        messages = [
            Message(
                role="system",
                content=self.system_prompt + additional_instructions,
            ),
            Message(role="user", content=user_content),
        ]

        assistant_message = await provider.generate_message(
            model=self.model.id,
            messages=messages,
            max_tokens=self.max_tokens,
            context_window=self.context_window,
        )

        raw = str(assistant_message.content or "").strip()

        fenced_match = None
        try:
            fenced_match = re.search(r"```json\s*([\s\S]*?)\s*```", raw, re.IGNORECASE)
        except Exception:
            fenced_match = None

        result_obj: dict[str, Any] | None
        if fenced_match:
            candidate = fenced_match.group(1).strip()
            try:
                result_obj = json.loads(candidate)
            except Exception:
                result_obj = None
        else:
            result_obj = None

        if result_obj is None:
            try:
                start = raw.find("{")
                end = raw.rfind("}")
                if 0 <= start < end:
                    snippet = raw[start : end + 1]
                    result_obj = json.loads(snippet)
            except Exception:
                result_obj = None

        if not isinstance(result_obj, dict):
            raise ValueError("StructuredOutputGenerator did not return a dictionary")

        return result_obj


class DataGenerator(BaseNode):
    """
    LLM Agent to create a dataframe based on a user prompt.
    llm, dataframe creation, data structuring

    Use cases:
    - Generating structured data from natural language descriptions
    - Creating sample datasets for testing or demonstration
    - Converting unstructured text into tabular format
    """

    model: LanguageModel = Field(
        default=LanguageModel(),
        description="The model to use for data generation.",
    )
    prompt: str = Field(
        default="",
        description="The user prompt",
    )
    input_text: str = Field(
        default="",
        description="The input text to be analyzed by the agent.",
    )
    max_tokens: int = Field(
        default=4096,
        ge=1,
        le=100000,
        description="The maximum number of tokens to generate.",
    )
    columns: RecordType = Field(
        default=RecordType(),
        description="The columns to use in the dataframe.",
    )

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["prompt", "model", "columns"]

    class OutputType(TypedDict):
        record: dict | None
        dataframe: DataframeRef | None
        index: int | None

    async def gen_process(
        self, context: ProcessingContext
    ) -> AsyncGenerator[OutputType, None]:
        """
        Streaming generation that yields individual records as they are generated
        and a final dataframe once all records are ready.
        """
        system_message = Message(
            role="system",
            content="You are an assistant with access to tools. Use generate_data to emit each row of the data.",
        )

        user_message = Message(
            role="user",
            content=self.prompt + "\n\n" + self.input_text,
        )
        messages = [system_message, user_message]

        collected_rows: list[dict] = []

        provider = get_provider(self.model.provider)
        index = 0
        async for chunk in provider.generate_messages(
            model=self.model.id,
            messages=messages,
            max_tokens=self.max_tokens,
            tools=[
                GenerateDataTool(
                    description="Generate a record according to the schema",
                    columns=self.columns.columns,
                )
            ],
        ):
            if isinstance(chunk, ToolCall):
                logger.debug("Tool call args: %s", chunk.args)
                collected_rows.append(chunk.args)
                # Yield each generated record immediately
                yield {"record": chunk.args, "index": index, "dataframe": None}
                index += 1

        # After streaming completes, yield the full dataframe once
        data = [
            [
                (row[col.name] if col.name in row else None)
                for col in self.columns.columns
            ]
            for row in collected_rows
        ]
        yield {
            "dataframe": DataframeRef(columns=self.columns.columns, data=data),
            "index": None,
            "record": None,
        }


class ListGenerator(BaseNode):
    """
    LLM Agent to create a stream of strings based on a user prompt.
    llm, text streaming

    Use cases:
    - Generating text from natural language descriptions
    - Streaming responses from an LLM
    """

    model: LanguageModel = Field(
        default=LanguageModel(),
        description="The model to use for string generation.",
    )
    prompt: str = Field(
        default="",
        description="The user prompt",
    )
    input_text: str = Field(
        default="",
        description="The input text to be analyzed by the agent.",
    )
    max_tokens: int = Field(
        default=4096,
        ge=1,
        le=100000,
        description="The maximum number of tokens to generate.",
    )

    class OutputType(TypedDict):
        item: str
        index: int

    async def gen_process(
        self, context: ProcessingContext
    ) -> AsyncGenerator[OutputType, None]:
        system_message = Message(
            role="system",
            content="""You are an assistant that generates lists.
            If the user asks for a specific number of items, generate that many.
            The output should be a numbered list.
            Example:
            User: Generate 5 movie titles
            Assistant: 
            1. The Dark Knight
            2. Inception
            3. The Matrix
            4. Interstellar
            5. The Lord of the Rings
            """,
        )

        user_message = Message(
            role="user",
            content=self.prompt + "\n\n" + self.input_text,
        )
        messages = [system_message, user_message]

        buffer = ""
        current_item = ""
        current_index = 0
        in_item = False
        collected_items: list[str] = []

        provider = get_provider(self.model.provider)
        async for chunk in provider.generate_messages(
            model=self.model.id,
            messages=messages,
            max_tokens=self.max_tokens,
        ):
            if isinstance(chunk, Chunk):
                buffer += chunk.content

                # Process the buffer line by line
                lines = buffer.split("\n")
                buffer = lines.pop()  # Keep the last partial line in the buffer

                for line in lines:
                    line = line.strip()
                    if not line:
                        continue

                    # Check if this line starts a new numbered item
                    import re

                    match = re.match(r"^\s*(\d+)\.\s*(.*)", line)

                    if match:
                        # If we were processing a previous item, yield it
                        if in_item and current_item:
                            collected_items.append(current_item)
                            yield {"item": current_item, "index": current_index}

                        # Start a new item
                        current_index = int(match.group(1))
                        current_item = match.group(2)
                        in_item = True
                    elif in_item:
                        # Continue with the current item
                        current_item += " " + line

        # Process any remaining complete lines in the buffer
        if buffer:
            match = re.match(r"^\s*(\d+)\.\s*(.*)", buffer.strip())
            if match:
                # If we were processing a previous item, yield it
                if in_item and current_item:
                    collected_items.append(current_item)
                    yield {"item": current_item, "index": current_index}

                # Process the final item
                current_index = int(match.group(1))
                current_item = match.group(2)
                collected_items.append(current_item)
                yield {"item": current_item, "index": current_index}
            elif in_item:
                # Add to the current item and yield it
                current_item += " " + buffer.strip()
                collected_items.append(current_item)
                yield {"item": current_item, "index": current_index}

        # Flush any final item if not yet emitted
        if (
            in_item
            and current_item
            and (not collected_items or collected_items[-1] != current_item)
        ):
            collected_items.append(current_item)
            yield {"item": current_item, "index": current_index}

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["prompt", "model"]


PLOTLY_CHART_CONFIG_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Plotly Chart Configuration",
    "description": "Simplified schema for defining a Plotly chart configuration",
    "type": "object",
    "required": ["data"],
    "properties": {
        "data": {
            "type": "array",
            "description": "Array of trace objects",
            "items": {
                "type": "object",
                "required": ["type"],
                "properties": {
                    "type": {
                        "type": "string",
                        "description": "The visualization type",
                        "enum": [
                            "scatter",
                            "bar",
                            "pie",
                            "line",
                            "histogram",
                            "box",
                            "violin",
                            "heatmap",
                        ],
                    },
                    "x": {
                        "type": "array",
                        "description": "X-axis values",
                        "items": {"type": "number"},
                    },
                    "y": {
                        "type": "array",
                        "description": "Y-axis values",
                        "items": {"type": "number"},
                    },
                    "mode": {
                        "type": "string",
                        "description": "Drawing mode for scatter traces",
                        "enum": [
                            "lines",
                            "markers",
                            "text",
                            "lines+markers",
                            "text+markers",
                            "text+lines",
                            "text+lines+markers",
                        ],
                    },
                    "name": {
                        "type": "string",
                        "description": "Trace name shown in legend",
                    },
                    "marker": {
                        "type": "object",
                        "description": "Marker properties",
                        "properties": {
                            "color": {
                                "type": "string",
                                "description": "Marker color or array of colors",
                            },
                            "size": {
                                "type": "number",
                                "description": "Marker size or array of sizes",
                            },
                            "opacity": {
                                "type": "number",
                                "description": "Marker opacity",
                                "minimum": 0,
                                "maximum": 1,
                            },
                        },
                    },
                    "line": {
                        "type": "object",
                        "description": "Line properties",
                        "properties": {
                            "color": {"type": "string", "description": "Line color"},
                            "width": {"type": "number", "description": "Line width"},
                            "dash": {
                                "type": "string",
                                "description": "Line dash style",
                                "enum": [
                                    "solid",
                                    "dot",
                                    "dash",
                                    "longdash",
                                    "dashdot",
                                    "longdashdot",
                                ],
                            },
                        },
                    },
                    "text": {
                        "type": "array",
                        "description": "Text displayed on hover or as annotations",
                        "items": {"type": "string"},
                    },
                    "hovertext": {
                        "type": "array",
                        "description": "Text shown on hover",
                        "items": {"type": "string"},
                    },
                },
            },
        },
        "layout": {
            "type": "object",
            "description": "Chart layout options",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Chart title",
                },
                "xaxis": {
                    "type": "object",
                    "description": "X-axis configuration",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "X-axis title",
                        },
                        "type": {
                            "type": "string",
                            "enum": ["linear", "log", "date", "category"],
                            "description": "Axis type",
                        },
                        "range": {
                            "type": "array",
                            "minItems": 2,
                            "maxItems": 2,
                            "items": {"type": "number"},
                            "description": "Axis range [min, max]",
                        },
                    },
                },
                "yaxis": {
                    "type": "object",
                    "description": "Y-axis configuration",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "Y-axis title",
                        },
                        "type": {
                            "type": "string",
                            "enum": ["linear", "log", "date", "category"],
                            "description": "Axis type",
                        },
                        "range": {
                            "type": "array",
                            "minItems": 2,
                            "maxItems": 2,
                            "items": {"type": "number"},
                            "description": "Axis range [min, max]",
                        },
                    },
                },
                "showlegend": {
                    "type": "boolean",
                    "description": "Whether to show the legend",
                },
                "height": {"type": "number", "description": "Chart height in pixels"},
                "width": {"type": "number", "description": "Chart width in pixels"},
                "margin": {
                    "type": "object",
                    "properties": {
                        "l": {"type": "number"},
                        "r": {"type": "number"},
                        "t": {"type": "number"},
                        "b": {"type": "number"},
                        "pad": {"type": "number"},
                    },
                    "description": "Chart margins",
                },
            },
        },
        "config": {
            "type": "object",
            "description": "Chart configuration options",
            "properties": {
                "responsive": {
                    "type": "boolean",
                    "description": "Whether the chart should resize with the window",
                },
                "displayModeBar": {
                    "type": "boolean",
                    "description": "Controls the display of the mode bar",
                },
                "scrollZoom": {
                    "type": "boolean",
                    "description": "Whether scrolling zooms the chart",
                },
            },
        },
    },
}


class ChartGenerator(BaseNode):
    """
    LLM Agent to create Plotly Express charts based on natural language descriptions.
    llm, data visualization, charts

    Use cases:
    - Generating interactive charts from natural language descriptions
    - Creating data visualizations with minimal configuration
    - Converting data analysis requirements into visual representations
    """

    model: LanguageModel = Field(
        default=LanguageModel(),
        description="The model to use for chart generation.",
    )
    prompt: str = Field(
        default="",
        description="Natural language description of the desired chart",
    )
    data: DataframeRef = Field(
        default=DataframeRef(),
        description="The data to visualize",
    )
    max_tokens: int = Field(
        default=4096,
        ge=1,
        le=100000,
        description="The maximum number of tokens to generate.",
    )

    async def process(self, context: ProcessingContext) -> PlotlyConfig:
        system_message = Message(
            role="system",
            content="""You are an expert data visualization assistant specializing in creating Plotly chart configurations.

Your task is to:
1. Analyze the data schema and user's request to determine the most appropriate visualization.
2. Generate a single, complete, ready-to-use JSON object that defines the Plotly chart. This JSON object must conform to the provided schema.
3. Choose chart types that best represent the relationships in the data.
4. Apply best practices for data visualization (e.g., appropriate colors, labels, legends) within the JSON structure.
5. Explain your chart choices and highlight insights the visualization reveals in a separate thought process (not part of the JSON output).

When creating the JSON chart configuration:
- Select appropriate visual encodings based on data types (categorical vs continuous).
- Use multiple traces when comparative analysis would be valuable.
- Set reasonable figure dimensions, margins, and formatting.
- Include helpful annotations or trendlines when they add value.
- Configure interactive elements effectively (tooltips, hover data).
- Consider color accessibility and choose colorblind-friendly palettes.

Always return a single JSON object as your primary output, conforming to the schema. Do not include any other text, explanations, or Python code outside of this JSON object.
""",
        )
        assert self.data.columns is not None, "Define columns"
        assert self.model.provider != Provider.Empty, "Select a model"

        user_message = Message(
            role="user",
            content=f"""Available columns in the dataset:
{json.dumps([c.model_dump() for c in self.data.columns ], indent=2)}

Input data:
{json.dumps(self.data.data if self.data.data else [], indent=2)}

User request: {self.prompt}

# Instructions
Please create a complete Plotly JSON configuration object that best visualizes this data according to the user's request and the provided schema.

Your response must be a single JSON object with 'data' and 'layout' properties, conforming to the following structure:
- The 'data' property should be an array of trace objects (e.g., scatter, bar, pie).
- The 'layout' property should define chart aesthetics like title, axis labels, and legend.

Focus on these chart types when defining traces in the JSON:
- "scatter": For relationships between variables
- "line": For trends over time or sequences
- "bar": For comparing categories
- "histogram": For distributions
- "box": For statistical summaries
- "violin": For probability distributions
- "heatmap": For correlation matrices
- "pie": For part-to-whole relationships

Remember to include axis labels, titles, and proper formatting within the JSON 'layout' object.
Ensure the generated JSON is valid and strictly adheres to the schema provided for the 'plotly_config' tool.
            """,
        )

        messages = [system_message, user_message]

        provider = get_provider(self.model.provider)
        assistant_message = await provider.generate_message(
            model=self.model.id,
            messages=messages,
            max_tokens=self.max_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "plotly_config",
                    "schema": PLOTLY_CHART_CONFIG_SCHEMA,
                },
            },
        )

        return PlotlyConfig(config=json.loads(str(assistant_message.content)))

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["prompt", "data", "model"]


def _extract_svg_content(text: str) -> str:
    """Extract SVG content from text, handling various formats."""
    # Look for SVG tags
    svg_start = text.find("<svg")
    svg_end = text.rfind("</svg>")

    if svg_start == -1 or svg_end == -1:
        raise ValueError("No valid SVG content found in the response")

    # Extract the SVG content including the closing tag
    svg_content = text[svg_start : svg_end + 6]
    logger.debug("Extracted SVG content: %s", svg_content)
    return svg_content


def parse_svg_content(svg_content: str) -> list[SVGElement]:
    """Parse SVG content into SVGElement objects."""
    # For now, return a single SVG element
    return [SVGElement(content=svg_content)]


class SVGGenerator(BaseNode):
    """
    LLM Agent to create SVG elements based on user prompts.
    svg, generator, vector, graphics

    Use cases:
    - Creating vector graphics from text descriptions
    - Generating scalable illustrations
    - Creating custom icons and diagrams
    """

    model: LanguageModel = Field(
        default=LanguageModel(),
        description="The language model to use for SVG generation.",
    )
    prompt: str = Field(
        default="",
        description="The user prompt for SVG generation",
    )
    image: ImageRef = Field(
        default=ImageRef(),
        description="Image to use for generation",
    )
    audio: AudioRef = Field(
        default=AudioRef(),
        description="Audio to use for generation",
    )
    max_tokens: int = Field(
        default=8192,
        ge=1,
        le=100000,
        description="The maximum number of tokens to generate.",
    )

    async def process(self, context: ProcessingContext) -> list[SVGElement]:
        system_message = Message(
            role="system",
            content="""You are an expert at creating SVG graphics.
ONLY RESPOND WITH THE SVG CONTENT, NO OTHER TEXT.
The SVG should be valid, well-formed XML.
Include width and height attributes in the root SVG element.
Use clear, semantic element IDs and class names if needed.""",
        )

        assert self.model.provider != Provider.Empty, "Select a model"

        # Build the user message content
        content_parts = [self.prompt]

        # Add image description if provided
        if self.image.is_set():
            await context.image_to_pil(self.image)  # Validate image can be loaded
            content_parts.append("[Image provided for reference]")

        # Add audio description if provided
        if self.audio.is_set():
            await context.asset_to_bytes(self.audio)  # Validate audio can be loaded
            content_parts.append("[Audio provided for reference]")

        user_message = Message(
            role="user",
            content="\n".join(content_parts)
            + "\nONLY RESPOND WITH THE SVG CONTENT, NO OTHER TEXT.",
        )

        messages = [system_message, user_message]

        provider = get_provider(self.model.provider)
        assistant_message = await provider.generate_message(
            model=self.model.id,
            messages=messages,
            max_tokens=self.max_tokens,
            response_format={"type": "text"},
        )

        try:
            svg_content = str(assistant_message.content)
            final_svg_content = _extract_svg_content(svg_content)
            return parse_svg_content(final_svg_content)
        except Exception as e:
            raise RuntimeError(f"Failed to generate SVG: {str(e)}") from e

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["prompt", "image", "audio", "model"]
