"""
Wikipedia style research and documentation – DSL workflow

Reimplementation of nodetool-core/examples/wikipedia_agent_example.py
as a nodetool DSL graph using ResearchAgent.
"""

from nodetool.dsl.graph import create_graph, run_graph
from nodetool.dsl.nodetool.input import StringInput
from nodetool.dsl.nodetool.text import FormatText
from nodetool.dsl.nodetool.agents import ResearchAgent
from nodetool.dsl.nodetool.output import Output
from nodetool.metadata.types import LanguageModel, Provider

topic_url = StringInput(
    name="seed_url",
    description="Starting Wikipedia page to crawl",
    value="https://en.wikipedia.org/wiki/Fine-tuning_(deep_learning)",
)

topic_label = StringInput(
    name="topic_label",
    description="Label used in the final article",
    value="LLM fine tuning",
)

objective = FormatText(
    template="""
You are a Wikipedia style research and documentation agent.

Task:
1) Start from {{ url }}.
2) Identify and crawl linked pages that are relevant to {{ label }}.
3) Extract and organize information required to explain {{ label }} clearly to a technical audience.

Writing requirements:
- Produce a comprehensive markdown article suitable for a Wikipedia style page.
- Include sections such as Overview, Background, Techniques, Practical considerations,
  Limitations, and References.
- Use neutral, precise language and avoid marketing tone.
- Provide reference style bullet lists for key papers or standards.

Output format:
- Return a JSON object with one key:
  {
    "article_markdown": "<full markdown article>"
  }
""",
    url=topic_url.output,
    label=topic_label.output,
)

wiki_agent = ResearchAgent(
    objective=objective.output,
    model=LanguageModel(
        type="language_model",
        id="gpt-4o-mini",
        provider=Provider.OpenAI,
    ),
    dynamic_outputs={
        "article_markdown": str,
    },
)

output = Output(
    name="wikipedia_style_article",
    value=wiki_agent.out.article_markdown,
)

graph = create_graph(output)

if __name__ == "__main__":
    result = run_graph(graph)
    print(result["wikipedia_style_article"])
