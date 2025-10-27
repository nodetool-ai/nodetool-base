"""
Summarize RSS Feed DSL Example

Fetch and summarize RSS feeds using AI.

Workflow:
1. **Fetch RSS Feed** - Retrieve feed items from a URL
2. **Collect Titles** - Combine all titles with a separator
3. **Summarize** - Use an AI model to create a concise summary
4. **Preview** - Display the final summary
"""

from nodetool.dsl.graph import graph_result
from nodetool.dsl.lib.rss import FetchRSSFeed
from nodetool.dsl.nodetool.text import Collect
from nodetool.dsl.nodetool.agents import Summarizer
from nodetool.dsl.nodetool.output import StringOutput
from nodetool.metadata.types import LanguageModel, Provider


async def example():
    """
    Fetch BBC News RSS feed and summarize it.
    """
    # Fetch RSS feed
    rss_feed = FetchRSSFeed(
        url="https://feeds.bbci.co.uk/news/world/europe/rss.xml"
    )

    # Collect all titles into one text block
    collected = Collect(
        # Connect the title output from rss_feed
        input_item=rss_feed.out.title,
        separator="---",
    )

    # Summarize the collected text
    summary = Summarizer(
        text=collected.out.output,
        model=LanguageModel(
            type="language_model",
            id="gemma3:1b",
            provider=Provider.Ollama,
        ),
        max_tokens=1000,
        context_window=4096,
    )

    # Output the summary
    output = StringOutput(
        name="summary",
        value=summary.out.text,
    )

    result = await graph_result(output)
    return result


if __name__ == "__main__":
    import asyncio

    result = asyncio.run(example())
    print(result["summary"])
