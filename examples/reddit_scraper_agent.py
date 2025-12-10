"""
Reddit community research – DSL workflow

Reimplementation of nodetool-core/examples/reddit_scraper_agent.py
as a nodetool DSL graph using ResearchAgent.
"""

from typing import List

from nodetool.dsl.graph import create_graph, run_graph
from nodetool.dsl.nodetool.input import StringInput
from nodetool.dsl.nodetool.text import FormatText
from nodetool.dsl.nodetool.agents import ResearchAgent
from nodetool.dsl.nodetool.output import StringOutput
from nodetool.metadata.types import LanguageModel, Provider

subreddit = StringInput(
    name="subreddit",
    description="Subreddit to analyze",
    value="webdev",
)

focus_area = StringInput(
    name="focus_area",
    description="What kind of issues to look for",
    value="common developer challenges and pain points",
)

objective = FormatText(
    template="""
You are an expert Reddit researcher specializing in customer feedback analysis.

Mission:
- Analyze r/{{ subreddit }} to identify {{ focus_area }}.

Research approach:
1) Use google_search with queries such as:
   - "site:reddit.com/r/{{ subreddit }} problem"
   - "site:reddit.com/r/{{ subreddit }} issue"
   - "site:reddit.com/r/{{ subreddit }} help"

2) For each promising post:
   - Fetch the thread with the browser tool.
   - Extract the core problem, context, and any workaround or solution patterns.

3) Synthesize an overview and actionable insights.

Output format (JSON object):
{
  "summary": "Brief 3 to 5 sentence overview of key findings",
  "posts_analyzed": <integer>,
  "key_issues": ["main problem 1", "main problem 2", ...],
  "recommendations": ["suggested improvement 1", "suggested improvement 2", ...]
}
""",
    subreddit=subreddit.output,
    focus_area=focus_area.output,
)

reddit_agent = ResearchAgent(
    objective=objective.output,
    model=LanguageModel(
        type="language_model",
        id="openai/gpt-oss-120b",
        provider=Provider.HuggingFaceCerebras,
    ),
    dynamic_outputs={
        "summary": str,
        "posts_analyzed": int,
        "key_issues": List[str],
        "recommendations": List[str],
    },
)

report = FormatText(
    template="""
# Reddit analysis for r/{{ subreddit }}

## Executive summary
{{ summary }}

- Posts analyzed: {{ posts }}

## Key issues
{% for issue in key_issues %}
- {{ issue }}
{% endfor %}

## Recommendations
{% for rec in recommendations %}
- {{ rec }}
{% endfor %}
""",
    subreddit=subreddit.output,
    summary=reddit_agent.out.summary,
    posts=reddit_agent.out.posts_analyzed,
    key_issues=reddit_agent.out.key_issues,
    recommendations=reddit_agent.out.recommendations,
)

output = StringOutput(
    name="reddit_research_report",
    value=report.output,
)

graph = create_graph(output)

if __name__ == "__main__":
    result = run_graph(graph)
    print(result["reddit_research_report"])
