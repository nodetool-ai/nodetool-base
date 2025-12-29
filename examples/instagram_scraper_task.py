"""
Instagram trends analysis – DSL workflow

Reimplementation of nodetool-core/examples/instagram_scraper_task.py
as a nodetool DSL graph using ResearchAgent.
"""

from typing import List, Dict

from nodetool.dsl.graph import create_graph, run_graph
from nodetool.dsl.nodetool.input import StringInput
from nodetool.dsl.nodetool.text import FormatText
from nodetool.dsl.nodetool.agents import ResearchAgent
from nodetool.dsl.nodetool.output import Output
from nodetool.metadata.types import LanguageModel, Provider

focus_topic = StringInput(
    name="focus",
    description="What to analyze trends for",
    value="current Instagram trends for tech and AI content",
)

objective = FormatText(
    template="""
Use google_search and browser tools to analyze Instagram trends.

Tasks:
1) Identify top 5 to 10 trending hashtags that are relevant to:
   {{ focus }}

2) For each hashtag:
   - Use google_search to find at least one example post.
   - Use the browser tool to inspect the post details and any public engagement signals
     such as likes, comments, and caption patterns.

3) Synthesize trends:
   - Describe what content style is winning for each hashtag.
   - Highlight patterns in visuals, captions, and calls to action.

Output format (JSON object):
{
  "trends": [
    {
      "hashtag": "<hashtag string>",
      "description": "<short explanation of why this trend matters>"
    },
    ...
  ]
}
""",
    focus=focus_topic.output,
)

instagram_agent = ResearchAgent(
    objective=objective.output,
    model=LanguageModel(
        type="language_model",
        id="openai/gpt-oss-120b",
        provider=Provider.HuggingFaceCerebras,
    ),
    dynamic_outputs={
        "trends": List[Dict[str, str]],
    },
)

summary_markdown = FormatText(
    template="""
# Instagram trend summary

{% for t in trends %}
## {{ t.hashtag }}
{{ t.description }}

{% endfor %}
""",
    trends=instagram_agent.out.trends,
)

output = Output(
    name="instagram_trends_report",
    value=summary_markdown.output,
)

graph = create_graph(output)

if __name__ == "__main__":
    result = run_graph(graph)
    print(result["instagram_trends_report"])
