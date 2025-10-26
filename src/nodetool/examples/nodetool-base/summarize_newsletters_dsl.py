"""
Summarize Newsletters DSL Example

Fetch and summarize newsletters from Gmail using AI.

Workflow:
1. **Gmail Search** - Searches Gmail inbox for emails with "AINews" in the subject from the past week
2. **Email Fields Extraction** - Extracts the body content from the retrieved email
3. **Summarizer** - Processes the email body through a language model to generate a concise summary
4. **Preview** - Display the summary result
"""

from nodetool.dsl.graph import graph_result
from nodetool.dsl.lib.mail import GmailSearch, EmailFields
from nodetool.dsl.nodetool.agents import Summarizer
from nodetool.dsl.nodetool.output import StringOutput
from nodetool.metadata.types import LanguageModel, Provider


async def example():
    """
    Search for AINews newsletters and summarize the latest one.
    """
    # Search Gmail for AINews emails from the past week
    gmail_search = GmailSearch(
        subject="AINews",
        date_filter=GmailSearch.DateFilter.SINCE_ONE_WEEK,
        folder=GmailSearch.GmailFolder.INBOX,
        max_results=1,
    )

    # Extract email body fields
    email_fields = EmailFields(email=gmail_search)

    # Summarize the newsletter content
    summarizer = Summarizer(
        text=(email_fields, "body"),  # Connect the body output
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
            name="gpt-oss-120b",
        ),
        max_tokens=1000,
        context_window=4096,
    )

    # Output the result
    output = StringOutput(
        name="summary",
        value=(summarizer, "chunk"),
    )

    result = await graph_result(output)
    return result



if __name__ == "__main__":
    import asyncio

    result = asyncio.run(example())
    print(f"Summary: {result}")
