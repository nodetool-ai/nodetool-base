"""
Thumbnail Prompt Generator

Generates AI image prompts for workflow thumbnails by analyzing workflow JSON files.

This workflow:
1. Takes workflow information as input (name, description, tags, node types)
2. Uses an AI Agent to analyze the workflow's purpose
3. Generates a compelling image prompt suitable for a thumbnail

The generated prompts can be used with image generation models to create
visually appealing thumbnails that represent each workflow.
"""

from nodetool.dsl.graph import Graph, create_graph, run_graph
from nodetool.dsl.nodetool.input import StringInput
from nodetool.dsl.nodetool.agents import Agent
from nodetool.dsl.nodetool.text import FormatText
from nodetool.dsl.nodetool.output import Output
from nodetool.metadata.types import LanguageModel, Provider


# System prompt for the thumbnail generation agent
THUMBNAIL_SYSTEM_PROMPT = """You are a creative designer specialized in creating compelling thumbnail imagery for software workflows.

Your task is to analyze a NodeTool workflow and generate a single image prompt that:
1. Visually represents the workflow's purpose and functionality
2. Is suitable for a 1024x1024 thumbnail image
3. Uses modern, professional design aesthetics
4. Avoids text, UI elements, or screenshots
5. Focuses on abstract or conceptual representations
6. Uses vibrant colors and dynamic compositions

Output Format:
Return ONLY the image generation prompt, nothing else. No explanations, no prefixes.
The prompt should be 1-3 sentences, descriptive and visually focused."""


# Input: Workflow name
workflow_name_input = StringInput(
    name="workflow_name",
    description="The name of the workflow",
    value="Album Cover Creator",
)

# Input: Workflow description
workflow_description_input = StringInput(
    name="workflow_description", 
    description="A description of what the workflow does",
    value="Create an album cover for a given song using AI-powered image generation and text processing.",
)

# Input: Workflow tags
workflow_tags_input = StringInput(
    name="workflow_tags",
    description="Comma-separated tags describing the workflow",
    value="album-cover, art, design, music, ai",
)

# Input: Node types used in the workflow
workflow_node_types_input = StringInput(
    name="workflow_node_types",
    description="Comma-separated list of node types used in the workflow",
    value="nodetool.agents.Agent, nodetool.image.TextToImage, nodetool.text.FormatText",
)

# Format the analysis prompt
analysis_prompt = FormatText(
    template="""Analyze this NodeTool workflow and generate an image prompt for its thumbnail:

Workflow Name: {{ name }}
Description: {{ description }}
Tags: {{ tags }}
Node Types Used: {{ node_types }}

Generate a single, compelling image prompt that visually represents this workflow's purpose.
Focus on abstract, artistic representations - no text, UI elements, or literal screenshots.
The image should be eye-catching and professional.""",
    name=workflow_name_input.output,
    description=workflow_description_input.output,
    tags=workflow_tags_input.output,
    node_types=workflow_node_types_input.output,
)

# Agent to generate the thumbnail prompt
thumbnail_agent = Agent(
    prompt=analysis_prompt.output,
    model=LanguageModel(
        type="language_model",
        id="gemma3:4b",
        provider=Provider.Ollama,
    ),
    system=THUMBNAIL_SYSTEM_PROMPT,
    tools=[],  # No tools needed for this task
    max_tokens=256,
    context_window=4096,
)

# Output the generated prompt
output = Output(
    name="thumbnail_prompt",
    value=thumbnail_agent.out.text,
)

# Create the graph
graph = create_graph(output)


if __name__ == "__main__":
    # Run with default values
    result = run_graph(graph)
    print("Generated Thumbnail Prompt:")
    print("-" * 50)
    print(result.get("thumbnail_prompt", "No prompt generated"))
