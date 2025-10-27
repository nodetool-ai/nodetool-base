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
    email_fields = EmailFields(email=gmail_search.out.email)

    # Summarize the newsletter content
    summarizer = Summarizer(
        text=email_fields.out.body,  # Connect the body output
        model=LanguageModel(
            type="language_model",
            id="gemma3:1b",
            provider=Provider.Ollama,
            name="gemma3:1b",
        ),
        max_tokens=1000,
        context_window=4096,
    )

    # Output the result
    output = StringOutput(
        name="summary",
        value=summarizer.out.text,
    )

    result = await graph_result(output)
    return result



if __name__ == "__main__":
    import asyncio

    result = asyncio.run(example())
    print(result["summary"])
