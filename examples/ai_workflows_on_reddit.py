"""
AI Workflows on Reddit – DSL workflow

Reimplementation of nodetool-core/examples/ai_workflows_on_reddit.py
as a nodetool DSL graph using ResearchAgent.
"""

from nodetool.dsl.graph import create_graph, run_graph
from nodetool.dsl.nodetool.input import StringInput
from nodetool.dsl.nodetool.text import FormatText
from nodetool.dsl.nodetool.agents import ResearchAgent
from nodetool.dsl.nodetool.output import StringOutput
from nodetool.metadata.types import LanguageModel, Provider

# Optional user input to tune the query
topic = StringInput(
    name="topic",
    description="Topic for AI workflows to search on Reddit",
    value="AI workflows",
)

# Turn the original multi phase objective into a parameterized research objective
research_objective = FormatText(
    template="""
Goal: Find examples of AI workflows on Reddit and compile a markdown report of subreddits, posts, and top comments.

You must internally follow three phases, but do not expose the phase mechanics in the final report.

1) discover_posts (mode="discover")
   - Gather up to 50 recent Reddit posts relevant to "{{ topic }}".
   - Try diverse search queries to find posts.
   - Discover related keywords that help uncover more posts.

2) process_posts (mode="process")
   - For each post, strip any trailing slash, append ".json" and fetch via the browser tool.
   - Example: "https://www.reddit.com/r/AI/comments/1234567890/"
     becomes "https://www.reddit.com/r/AI/comments/1234567890.json".
   - For each post, extract: subreddit, post title, selftext, and a useful subset of comments
     (authors and comment text).

3) aggregate_report (mode="aggregate")
   - Aggregate the posts into structured markdown.
   - Provide:
       - An overall summary of observed AI workflow patterns.
       - Per post sections with:
         - Subreddit
         - Title
         - Link
         - Short summary of discussion and top comments.

Output format:
- Return a single JSON object with one key:
  {
    "report_markdown": "<final markdown report>"
  }
- The markdown should be ready to paste into a knowledge base or doc.
""",
    topic=topic.output,
)

# Research agent that uses google_search + browser tools by default
reddit_research_agent = ResearchAgent(
    objective=research_objective.output,
    model=LanguageModel(
        type="language_model",
        id="openai/gpt-oss-120b",
        provider=Provider.HuggingFaceCerebras,
    ),
    dynamic_outputs={
        "report_markdown": str,
    },
)

# Final output node: just expose the compiled markdown report
output = StringOutput(
    name="reddit_ai_workflows_report",
    value=reddit_research_agent.out.report_markdown,
)

graph = create_graph(output)

if __name__ == "__main__":
    result = run_graph(graph)
    print(result["reddit_ai_workflows_report"])
