"""
Social Media Sentiment Analysis DSL Example

Analyze sentiment and emotions in social media posts or user-provided text.

Workflow:
1. **Text Input** - User provides text content (tweets, reviews, posts)
2. **Text Classification** - Classify sentiment (positive, negative, neutral)
3. **Named Entity Recognition** - Extract mentioned entities
4. **Emotion Detection** - Detect emotional tone (joy, anger, sadness, etc.)
5. **Data Output** - Aggregate results into structured format
"""

from nodetool.dsl.graph import create_graph, run_graph
from nodetool.dsl.nodetool.input import StringInput
from nodetool.dsl.nodetool.agents import Classifier
from nodetool.dsl.nodetool.text import FormatText
from nodetool.dsl.nodetool.output import StringOutput
from nodetool.metadata.types import LanguageModel, Provider


# User input: social media text
social_text = StringInput(
    name="social_post",
    description="Social media post or text to analyze",
    value="This new AI update is absolutely amazing! I'm so excited about the possibilities #innovation",
)

# Sentiment classification
sentiment_classifier = Classifier(
    text=social_text.output,
    system_prompt="You are a sentiment analysis expert. Classify the sentiment as positive, negative, or neutral.",
    model=LanguageModel(
        type="language_model",
        id="gpt-4o-mini",
        provider=Provider.OpenAI,
    ),
    categories=["positive", "negative", "neutral"],
)

# Emotion detection
emotion_detector = Classifier(
    text=social_text.output,
    system_prompt="Identify the primary emotion expressed in this text.",
    model=LanguageModel(
        type="language_model",
        id="gpt-4o-mini",
        provider=Provider.OpenAI,
    ),
    categories=["joy", "sadness", "anger", "fear", "surprise", "neutral"],
)

# Format analysis results
analysis_summary = FormatText(
    template="""## Sentiment Analysis Report

**Original Text:** {{ text }}

**Sentiment:** {{ sentiment }}
**Primary Emotion:** {{ emotion }}

**Analysis:**
- Post contains {{ sentiment }} sentiment
- Emotional tone: {{ emotion }}
- Suitable for understanding audience reaction""",
    text=social_text.output,
    sentiment=sentiment_classifier.output,
    emotion=emotion_detector.output,
)

# Output formatted analysis
text_output = StringOutput(
    name="analysis",
    value=analysis_summary.output,
)

# Create the graph
graph = create_graph(text_output)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Social media sentiment DSL example")
    parser.add_argument(
        "--gradio",
        action="store_true",
        help="Launch a Gradio UI for this workflow",
    )
    args = parser.parse_args()

    if args.gradio:
        try:
            from nodetool.ui.gradio_auto import build_gradio_app
        except Exception:
            print(
                "Gradio UI requires the optional dependency 'gradio'.\n"
                "Install it with: pip install gradio",
            )
            raise

        app = build_gradio_app(
            graph,
            title="Social Media Sentiment (DSL)",
            description=(
                "Enter a post or message to analyze sentiment and primary emotion."
            ),
        )
        app.launch()
    else:
        result = run_graph(graph)
        print(f"Sentiment Analysis: {result}")
