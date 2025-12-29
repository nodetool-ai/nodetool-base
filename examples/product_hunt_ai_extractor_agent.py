"""
Product Hunt AI product extractor – DSL workflow

Reimplementation of the Product Hunt AI extractor agent as a nodetool DSL graph.
"""

from nodetool.dsl.graph import create_graph, run_graph
from nodetool.dsl.nodetool.input import StringInput
from nodetool.dsl.nodetool.text import FormatText
from nodetool.dsl.nodetool.agents import ResearchAgent
from nodetool.dsl.nodetool.output import Output
from nodetool.metadata.types import LanguageModel, Provider

archive_url = StringInput(
    name="product_hunt_archive_url",
    description="Monthly Product Hunt leaderboard URL",
    value="https://www.producthunt.com/leaderboard/monthly/2025/4",
)

objective = FormatText(
    template="""
You are an expert data analyst specializing in identifying AI powered products from Product Hunt leaderboards.

Start from the monthly leaderboard:
{{ url }}

Tasks:
1) Use the browser tool to load the leaderboard page and enumerate all product entries.
2) For each product:
   - Visit its Product Hunt page.
   - Extract product name, tagline, main description, and tags.
   - Decide if it is an AI product or heavily uses AI.
   - When the page is sparse or ambiguous, you may perform a targeted google_search for:
     "[Product Name] AI features" or "[Product Name] Product Hunt".

3) For AI products:
   - Record:
     - Product name.
     - Product Hunt URL.
     - A concise "AI focus summary" in 1 to 2 sentences.
     - A short list of AI related keywords or tags.
   - Infer the archive month and year from the source URL and include it as "Archive Source Month/Year".

4) Compile a single markdown report covering all identified AI products.

Output format:
Return a JSON object:
{
  "report_markdown": "<markdown report. One section per AI product in a stable format>"
}

If no AI products are identified after thorough analysis, set report_markdown to:
"No AI products were identified in the [Month Year] Product Hunt archive."
with [Month Year] resolved from the leaderboard URL.
""",
    url=archive_url.output,
)

ai_products_agent = ResearchAgent(
    objective=objective.output,
    model=LanguageModel(
        type="language_model",
        id="openai/gpt-oss-120b",
        provider=Provider.HuggingFaceCerebras,
    ),
    dynamic_outputs={
        "report_markdown": str,
    },
)

output = Output(
    name="product_hunt_ai_report",
    value=ai_products_agent.out.report_markdown,
)

graph = create_graph(output)

if __name__ == "__main__":
    result = run_graph(graph)
    print(result["product_hunt_ai_report"])
