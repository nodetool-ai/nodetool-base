"""
Categorize Emails DSL Example

Automatically categorize and organize emails using AI.

Workflow:
1. **Gmail Search** - Fetches recent emails using specified filters
2. **Template** - Formats each email into a structured prompt
3. **Classifier** - Uses an LLM to classify the email into categories
4. **Add Label** - Applies the determined label to each email in Gmail
"""

from nodetool.dsl.graph import graph_result, run_graph, graph
from nodetool.dsl.lib.mail import GmailSearch, AddLabel
from nodetool.dsl.nodetool.text import Template
from nodetool.dsl.nodetool.agents import Classifier
from nodetool.dsl.nodetool.output import StringOutput
from nodetool.metadata.types import LanguageModel


async def example():
    """
    Search Gmail for recent emails and classify them into categories.
    """
    # Search Gmail for recent emails
    gmail_search = GmailSearch(
        from_address="",
        to_address="",
        subject="",
        body="",
        date_filter="SINCE_ONE_DAY",
        keywords="",
        folder="INBOX",
        text="",
        max_results=10,
    )

    # Format email data for classification
    email_template = Template(
        string="subject: {{subject}}\nsender: {{sender}}\ncontent: {{body|truncate(100)}}",
        values=gmail_search,  # Pass email data from Gmail search
    )

    # Classify emails into categories
    classifier = Classifier(
        text=email_template,
        system_prompt="""
        You are a precise text classifier. Your task is to analyze the input text and assign confidence scores.
        """,
        model=LanguageModel(
            type="language_model",
            id="qwen3:4b",
            provider="ollama",
        ),
        categories=["newsletter", "work", "family", "friends"],
        multi_label=False,
    )

    # Add the classified label to the email
    add_label = AddLabel(
        message_id=(gmail_search, "message_id"),  # Connect specific output
        label=classifier,  # Use the classification result
    )

    # Output the result
    output = StringOutput(
        name="result",
        value=add_label,
    )

    # Run the workflow
    g = graph(output)
    result = await run_graph(g)
    return result


async def example_streaming():
    """
    Process emails with streaming results (shows more dynamic approach).
    """
    # Search and classify emails
    gmail_search = GmailSearch(
        from_address="",
        to_address="",
        subject="",
        body="",
        date_filter="SINCE_ONE_DAY",
        keywords="",
        folder="INBOX",
        text="",
        max_results=10,
    )

    email_template = Template(
        string="subject: {{subject}}\nsender: {{sender}}\ncontent: {{body|truncate(100)}}",
        values=gmail_search,
    )

    classifier = Classifier(
        text=email_template,
        system_prompt="You are a precise text classifier. Analyze and classify emails.",
        model=LanguageModel(
            type="language_model",
            id="qwen3:4b",
            provider="ollama",
        ),
        categories=["newsletter", "work", "family", "friends"],
        multi_label=False,
    )

    add_label = AddLabel(
        message_id=(gmail_search, "message_id"),
        label=classifier,
    )

    output = StringOutput(
        name="result",
        value=add_label,
    )

    g = graph(output)
    result = await run_graph(g)
    return result


if __name__ == "__main__":
    import asyncio

    result = asyncio.run(example())
    print(f"Labels applied: {result}")
