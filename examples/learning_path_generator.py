"""
Learning path generator – DSL workflow

Reimplementation of nodetool-core/examples/learning_path_generator.py
as a nodetool DSL graph using ResearchAgent with structured outputs.
"""

from typing import List, Dict, Any

from nodetool.dsl.graph import create_graph, run_graph
from nodetool.dsl.nodetool.input import StringInput
from nodetool.dsl.nodetool.text import FormatText, FormatText as MarkdownFormat
from nodetool.dsl.nodetool.agents import ResearchAgent
from nodetool.dsl.nodetool.output import Output
from nodetool.metadata.types import LanguageModel, Provider

topic = StringInput(
    name="topic",
    description="Topic for the learning path",
    value="Getting Started with Docker",
)

objective = FormatText(
    template="""
Create a comprehensive learning path for the topic "{{ topic }}".

Requirements:
- Structure the path into 3 to 5 modules.
- Each module must have:
  - title
  - description
  - resources: a list of concrete web resources (URLs, docs, tutorials).

Focus:
- Keep the plan focused on the learner's core objective.
- Order modules from beginner friendly to more advanced.

Output format:
Return a JSON object:
{
  "topic": "<topic name>",
  "overview": "<short overview of the full path>",
  "modules": [
    {
      "title": "<module title>",
      "description": "<what the learner will accomplish>",
      "resources": ["<url1>", "<url2>", ...]
    },
    ...
  ]
}
""",
    topic=topic.output,
)

learning_agent = ResearchAgent(
    objective=objective.output,
    model=LanguageModel(
        type="language_model",
        id="openai/gpt-oss-120b",
        provider=Provider.HuggingFaceCerebras,
    ),
    dynamic_outputs={
        "topic": str,
        "overview": str,
        "modules": List[Dict[str, Any]],
    },
)

# Render a readable markdown view of the structured plan
learning_plan_markdown = MarkdownFormat(
    template="""
# Learning path: {{ topic }}

## Overview
{{ overview }}

{% for m in modules %}
### {{ m.title }}
{{ m.description }}

Resources:
{% for r in m.resources %}
- {{ r }}
{% endfor %}

{% endfor %}
""",
    topic=learning_agent.out.topic,
    overview=learning_agent.out.overview,
    modules=learning_agent.out.modules,
)

output = Output(
    name="learning_path_markdown",
    value=learning_plan_markdown.output,
)

graph = create_graph(output)

if __name__ == "__main__":
    result = run_graph(graph)
    print(result["learning_path_markdown"])
