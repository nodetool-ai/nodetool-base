"""Social Media Sentiment Analysis DSL Example.

This example workflow classifies both the sentiment and the dominant
emotion of a single social media post. The result is formatted as a
short report that can be streamed back to an application or dashboard.

Key steps:
1. Collect user-provided text.
2. Run sentiment classification across positive/negative/neutral labels.
3. Detect the leading emotion with a second classifier node.
4. Render a readable summary that separates sentiment from emotion.

Run this module directly to execute the graph with sample data.
"""

from __future__ import annotations

import asyncio

from nodetool.dsl.graph import create_graph, run_graph
from nodetool.dsl.nodetool.agents import Classifier
from nodetool.dsl.nodetool.input import StringInput
from nodetool.dsl.nodetool.output import StringOutput
from nodetool.dsl.nodetool.text import FormatText
from nodetool.metadata.types import LanguageModel, Provider

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------
SENTIMENT_MODEL = LanguageModel(
    type="language_model",
    provider=Provider.OpenAI,
    id="gpt-4o-mini",
)

EMOTION_MODEL = LanguageModel(
    type="language_model",
    provider=Provider.OpenAI,
    id="gpt-4o-mini",
)


def build_social_media_sentiment_graph():
    """Create a graph that analyzes sentiment and emotion for a text sample."""

    # 1. Collect social media text input.
    social_post = StringInput(
        name="social_post",
        description="Raw social media text or customer feedback snippet",
        value=(
            "Just tried the new update – the UI feels so much smoother and the "
            "new shortcuts are lifesavers!"
        ),
    )

    # 2. Sentiment classification (positive, negative, neutral).
    sentiment_classifier = Classifier(
        text=social_post.output,
        categories=["positive", "negative", "neutral"],
        system_prompt=(
            "Classify the overall sentiment of the provided text as positive, "
            "negative, or neutral."
        ),
        model=SENTIMENT_MODEL,
    )

    # 3. Primary emotion detection using a separate classifier.
    emotion_classifier = Classifier(
        text=social_post.output,
        categories=["joy", "trust", "anticipation", "sadness", "anger", "fear", "disgust", "surprise", "neutral"],
        system_prompt=(
            "Select the single primary emotion conveyed in the text. Choose the "
            "closest match from the provided options."
        ),
        model=EMOTION_MODEL,
    )

    # 4. Format the analysis output for downstream consumers.
    analysis_report = FormatText(
        template=(
            """## Real-Time Sentiment Snapshot\n\n"
            "**Original Post**\n"
            "{{ post }}\n\n"
            "**Sentiment Classification**: {{ sentiment }}\n"
            "**Primary Emotion**: {{ emotion }}\n\n"
            "### Engagement Notes\n"
            "- Use sentiment to gauge response velocity.\n"
            "- Emotion highlights the dominant tone for moderation teams.\n"
        """
        ),
        post=social_post.output,
        sentiment=sentiment_classifier.output,
        emotion=emotion_classifier.output,
    )

    summary_output = StringOutput(
        name="sentiment_report",
        value=analysis_report.output,
        description="Formatted summary ready for dashboards or alerts",
    )

    return create_graph(summary_output)


async def main() -> None:
    """Execute the graph with sample data."""

    graph = build_social_media_sentiment_graph()
    results = await run_graph(graph, user_id="demo-user", auth_token="demo-token")
    print(results["sentiment_report"])


if __name__ == "__main__":
    asyncio.run(main())
