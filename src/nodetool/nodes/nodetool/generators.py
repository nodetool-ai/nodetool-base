import json
from pydantic import Field

from nodetool.chat.dataframes import (
    json_schema_for_dataframe,
)
from nodetool.metadata.types import (
    Message,
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
from nodetool.metadata.types import (
    Message,
)
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext

from nodetool.chat.dataframes import GenerateDataTool, GenerateStringTool


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
        description="The GPT model to use for data generation.",
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

    async def process(self, context: ProcessingContext) -> DataframeRef:
        system_message = Message(
            role="system",
            content="You are an assistant with access to tools.",
        )

        user_message = Message(
            role="user",
            content=self.prompt + "\n\n" + self.input_text,
        )
        messages = [system_message, user_message]

        assistant_message = await context.generate_message(
            node_id=self.id,
            provider=self.model.provider,
            model=self.model.id,
            messages=messages,
            max_tokens=self.max_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "datatable",
                    "schema": json_schema_for_dataframe(self.columns.columns),
                    "strict": True,
                },
            },
        )
        data = [
            [
                (row[col.name] if col.name in row else None)
                for col in self.columns.columns
            ]
            for row in json.loads(str(assistant_message.content)).get("data", [])
        ]
        return DataframeRef(columns=self.columns.columns, data=data)

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["prompt", "model", "columns"]


class DataStreamer(BaseNode):
    """
    LLM Agent to create a stream of data based on a user prompt.
    llm, data streaming, data structuring

    Use cases:
    - Generating structured data from natural language descriptions
    - Creating sample datasets for testing or demonstration
    """

    model: LanguageModel = Field(
        default=LanguageModel(),
        description="The GPT model to use for data generation.",
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

    async def gen_process(self, context: ProcessingContext):
        system_message = Message(
            role="system",
            content="You are an assistant with access to tools. Use generate_data to emit each row of the data.",
        )

        user_message = Message(
            role="user",
            content=self.prompt + "\n\n" + self.input_text,
        )
        messages = [system_message, user_message]

        async for chunk in context.generate_messages(
            node_id=self.id,
            provider=self.model.provider,
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
                print(chunk.args)
                yield "output", chunk.args

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["prompt", "model", "columns"]


class StringStreamer(BaseNode):
    """
    LLM Agent to create a stream of strings based on a user prompt.
    llm, text streaming

    Use cases:
    - Generating text from natural language descriptions
    - Streaming responses from an LLM
    """

    model: LanguageModel = Field(
        default=LanguageModel(),
        description="The GPT model to use for string generation.",
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

    @classmethod
    def return_type(cls):
        return str

    async def gen_process(self, context: ProcessingContext):
        system_message = Message(
            role="system",
            content="You are an assistant that generates text.",
        )

        user_message = Message(
            role="user",
            content=self.prompt + "\n\n" + self.input_text,
        )
        messages = [system_message, user_message]

        async for chunk in context.generate_messages(
            node_id=self.id,
            provider=self.model.provider,
            model=self.model.id,
            messages=messages,
            max_tokens=self.max_tokens,
            tools=[GenerateStringTool(description="Generate a string")],
        ):
            if isinstance(chunk, ToolCall):
                yield "output", chunk.args["string"]

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
        description="The GPT model to use for chart generation.",
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
            content="""You are an expert data visualization assistant specializing in Plotly Express charts.

Your task is to:
1. Analyze the data schema and user's request to create the most appropriate visualization
2. Generate complete, ready-to-use Plotly Express Python code
3. Choose chart types that best represent the relationships in the data
4. Apply best practices for data visualization (appropriate colors, labels, legends)
5. Explain your chart choices and highlight insights the visualization reveals

When creating charts:
- Select appropriate visual encodings based on data types (categorical vs continuous)
- Use multiple traces when comparative analysis would be valuable
- Set reasonable figure dimensions, margins, and formatting
- Include helpful annotations or trendlines when they add value
- Configure interactive elements effectively (tooltips, hover data)
- Consider color accessibility and choose colorblind-friendly palettes

Always return complete, executable Python code that imports plotly.express as px and any other necessary libraries.
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
Please create complete Plotly Express Python code that best visualizes this data according to the user's request.

Your response should include:
1. A recommendation for the most appropriate chart type(s)
2. Complete Python code using plotly.express
3. A brief explanation of why this visualization works well for the data and request
4. Any insights that can be observed from the visualization

Focus on these chart types as appropriate:
- px.scatter: For relationships between variables
- px.line: For trends over time or sequences
- px.bar: For comparing categories
- px.histogram: For distributions
- px.box: For statistical summaries
- px.violin: For probability distributions
- px.heatmap: For correlation matrices
- px.pie: For part-to-whole relationships

Remember to include axis labels, titles, and proper formatting in your code.
            """,
        )

        messages = [system_message, user_message]

        assistant_message = await context.generate_message(
            node_id=self.id,
            provider=self.model.provider,
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
    print(svg_content)
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
            image = await context.image_to_pil(self.image)
            content_parts.append("[Image provided for reference]")

        # Add audio description if provided
        if self.audio.is_set():
            audio = await context.asset_to_bytes(self.audio)
            content_parts.append("[Audio provided for reference]")

        user_message = Message(
            role="user",
            content="\n".join(content_parts)
            + "\nONLY RESPOND WITH THE SVG CONTENT, NO OTHER TEXT.",
        )

        messages = [system_message, user_message]

        assistant_message = await context.generate_message(
            node_id=self.id,
            provider=self.model.provider,
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
