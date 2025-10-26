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
        input_item=rss_feed,  # This will use the default output
        separator="---",
    )

    # Summarize the collected text
    summary = Summarizer(
        text=collected,
        system_prompt="""
        You are an expert summarizer. Your task is to create clear, accurate, and concise summaries using Markdown for structuring.
        Follow these guidelines:
        1. Identify and include only the most important information.
        2. Maintain factual accuracy - do not add or modify information.
        3. Use clear, direct language.
        4. Aim for approximately 1000 tokens.
        """,
        model=LanguageModel(
            type="language_model",
            id="openai/gpt-oss-120b",
            provider=Provider.HuggingFaceCerebras,
        ),
        max_tokens=1000,
        context_window=4096,
    )

    # Output the summary
    output = StringOutput(
        name="summary",
        value=summary,
    )

    result = await graph_result(output)
    return result


if __name__ == "__main__":
    import asyncio

    result = asyncio.run(example())
    print(f"Summary: {result}")
