"""Product Description Generator DSL Example.

This example demonstrates a two-agent workflow for marketing teams. A
copywriter agent drafts benefit-focused messaging while an SEO specialist
produces a keyword-aware alternative. The outputs are combined into a
single packaged response that is ready for multi-channel deployment.

Key steps:
1. Collect product metadata inputs.
2. Generate an engaging marketing description.
3. Produce an SEO-optimized variant using the same context.
4. Assemble a formatted bundle for downstream publishing tools.

Run this module directly to execute the graph with the bundled sample
product values.
"""

from __future__ import annotations

import asyncio

from nodetool.dsl.graph import create_graph, run_graph
from nodetool.dsl.nodetool.agents import Agent
from nodetool.dsl.nodetool.input import StringInput
from nodetool.dsl.nodetool.output import StringOutput
from nodetool.dsl.nodetool.text import FormatText
from nodetool.metadata.types import LanguageModel, Provider

# ---------------------------------------------------------------------------
# Model configuration shared across nodes
# ---------------------------------------------------------------------------
MARKETING_MODEL = LanguageModel(
    type="language_model",
    provider=Provider.OpenAI,
    id="gpt-4o",
)

SEO_MODEL = LanguageModel(
    type="language_model",
    provider=Provider.OpenAI,
    id="gpt-4o-mini",
)


def build_product_description_graph():
    """Create a graph that generates marketing and SEO product copy."""

    # 1. Capture product inputs that downstream nodes reuse.
    product_name = StringInput(
        name="product_name",
        description="Product name and high-level category",
        value="AuroraX Smart Lighting System",
    )

    product_features = StringInput(
        name="product_features",
        description="Primary features and differentiators",
        value=(
            "Adaptive white-to-color lighting, voice assistant integration, "
            "energy usage analytics, fully wireless installation"
        ),
    )

    target_audience = StringInput(
        name="target_audience",
        description="Ideal customer profile or segment",
        value="Design-conscious homeowners upgrading smart home experiences",
    )

    # 2. Marketing-focused copy generation.
    marketing_agent = Agent(
        prompt=FormatText(
            template=(
                """Craft a benefit-led marketing description for the following product.\n\n"
                "Product: {{ product }}\n"
                "Key features: {{ features }}\n"
                "Target audience: {{ audience }}\n\n"
                "Deliver 2-3 punchy sentences optimized for landing pages and ads."""
            ),
            product=product_name.output,
            features=product_features.output,
            audience=target_audience.output,
        ).output,
        model=MARKETING_MODEL,
        system="You are an award-winning marketing copywriter focused on compelling storytelling.",
        max_tokens=400,
    )

    # 3. SEO-tailored variation using a second agent.
    seo_agent = Agent(
        prompt=FormatText(
            template=(
                """Write an SEO-optimized product description.\n\n"
                "Product: {{ product }}\n"
                "Key features: {{ features }}\n"
                "Primary audience: {{ audience }}\n\n"
                "Include a short bullet list of search-friendly benefits."""
            ),
            product=product_name.output,
            features=product_features.output,
            audience=target_audience.output,
        ).output,
        model=SEO_MODEL,
        system="You are an SEO strategist who balances readability with high-value keywords.",
        max_tokens=350,
    )

    # 4. Merge both outputs into a single formatted package.
    packaged_copy = FormatText(
        template=(
            """# Product Launch Copy Deck\n\n"
            "## {{ product }}\n\n"
            "### Primary Marketing Message\n"
            "{{ marketing_copy }}\n\n"
            "### SEO Variant\n"
            "{{ seo_copy }}\n\n"
            "### Feature Highlights\n"
            "{{ features }}\n\n"
            "### Target Audience\n"
            "{{ audience }}\n\n"
            "---\n"
            "Distribute the marketing message across paid social and lifecycle emails,\n"
            "while the SEO variant is optimized for website product pages and blog articles."
        """
        ),
        product=product_name.output,
        marketing_copy=marketing_agent.out.text,
        seo_copy=seo_agent.out.text,
        features=product_features.output,
        audience=target_audience.output,
    )

    output = StringOutput(
        name="product_copy_package",
        value=packaged_copy.output,
        description="Combined marketing and SEO narratives for the product",
    )

    return create_graph(output)


async def main() -> None:
    """Execute the graph with sample inputs."""

    graph = build_product_description_graph()
    results = await run_graph(graph, user_id="demo-user", auth_token="demo-token")
    print(results["product_copy_package"])


if __name__ == "__main__":
    asyncio.run(main())
