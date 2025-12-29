"""
Hacker News research – DSL workflow

Reimplementation of nodetool-core/examples/test_hackernews_agent.py
as a nodetool DSL graph using ResearchAgent.
"""

from typing import List, Dict, Any

from nodetool.dsl.graph import create_graph, run_graph
from nodetool.dsl.nodetool.text import FormatText
from nodetool.dsl.nodetool.agents import ResearchAgent
from nodetool.dsl.nodetool.output import Output
from nodetool.metadata.types import LanguageModel, Provider

objective = FormatText(
    template="""
Scrape the front page of https://news.ycombinator.com/ and analyze activity.

Tasks:
1) Identify the top 5 posts on the front page.
2) For each post:
   - Record title and url.
   - Fetch the discussion thread with the browser tool.
   - Extract the top 3 comments by score or prominence.
   - For each comment, store text and author.

3) Summarize the overall themes across the selected posts.

Output format (JSON):
{
  "summary": "<high level summary of the 5 posts>",
  "posts": [
    {
      "title": "<title>",
      "url": "<url>",
      "top_comments": [
        {"text": "<comment text>", "author": "<username>"},
        ...
      ]
    },
    ...
  ]
}
""",
)

hn_agent = ResearchAgent(
    objective=objective.output,
    model=LanguageModel(
        type="language_model",
        id="openai/gpt-oss-120b",
        provider=Provider.HuggingFaceCerebras,
    ),
    dynamic_outputs={
        "summary": str,
        "posts": List[Dict[str, Any]],
    },
)

report = FormatText(
    template="""
# Hacker News snapshot

## Overview
{{ summary }}

{% for p in posts %}
### {{ p.title }}
{{ p.url }}

Top comments:
{% for c in p.top_comments %}
- {{ c.author }}: {{ c.text }}
{% endfor %}

{% endfor %}
""",
    summary=hn_agent.out.summary,
    posts=hn_agent.out.posts,
)

output = Output(
    name="hn_report",
    value=report.output,
)

graph = create_graph(output)

if __name__ == "__main__":
    result = run_graph(graph)
    print(result["hn_report"])
