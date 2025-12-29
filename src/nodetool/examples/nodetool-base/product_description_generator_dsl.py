"""
Product Description Generator DSL Example

Automatically generate marketing copy and product descriptions from product data.

Workflow:
1. **Product Input** - Provide product details (name, features, price)
2. **Market Research** - Analyze target audience
3. **Copy Generation** - Generate engaging descriptions
4. **A/B Variants** - Create multiple versions for testing
5. **Output** - Save generated descriptions
"""

from nodetool.dsl.graph import create_graph, run_graph
from nodetool.dsl.nodetool.input import StringInput
from nodetool.dsl.nodetool.agents import Agent
from nodetool.dsl.nodetool.text import FormatText
from nodetool.dsl.nodetool.output import Output
from nodetool.metadata.types import LanguageModel, Provider


# Product information
product_name = StringInput(
    name="product",
    description="Product name and category",
    value="Wireless Noise-Cancelling Headphones",
)

product_features = StringInput(
    name="features",
    description="Key product features",
    value="40-hour battery life, active noise cancellation, premium audio quality, comfortable design, fast charging",
)

target_audience = StringInput(
    name="audience",
    description="Target customer segment",
    value="Professional remote workers and frequent travelers",
)

# Generate marketing description
description_generator = Agent(
    prompt=FormatText(
        template="""Create a compelling product description for:
Product: {{ product }}
Features: {{ features }}
Target Audience: {{ audience }}

Generate a 2-3 sentence marketing description that highlights benefits, not just features.""",
        product=product_name.output,
        features=product_features.output,
        audience=target_audience.output,
    ).output,
    model=LanguageModel(
        type="language_model",
        id="gpt-4o",
        provider=Provider.OpenAI,
    ),
    system="You are an expert copywriter. Create compelling product descriptions that sell.",
    max_tokens=500,
)

# Generate SEO-optimized version
seo_generator = Agent(
    prompt=FormatText(
        template="""Create an SEO-optimized product description for:
Product: {{ product }}
Features: {{ features }}

Include relevant keywords naturally. Keep it 100-150 words.""",
        product=product_name.output,
        features=product_features.output,
    ).output,
    model=LanguageModel(
        type="language_model",
        id="gpt-4o-mini",
        provider=Provider.OpenAI,
    ),
    system="You are an SEO specialist. Create keyword-rich product descriptions.",
    max_tokens=300,
)

# Format final output
marketing_copy = FormatText(
    template="""# Product Description Package

## Product: {{ product }}

### Marketing Description (A/B Test Version 1):
{{ marketing_desc }}

### SEO-Optimized Description (A/B Test Version 2):
{{ seo_desc }}

### Key Features to Highlight:
{{ features }}

### Target Audience:
{{ audience }}

---
**Usage Tips:**
- Test both descriptions with your audience
- Use Version 1 for social media and ads
- Use Version 2 for website and search optimization
""",
    product=product_name.output,
    marketing_desc=description_generator.out.text,
    seo_desc=seo_generator.out.text,
    features=product_features.output,
    audience=target_audience.output,
)

# Output generated descriptions
output = Output(
    name="product_descriptions",
    value=marketing_copy.output,
)

# Create the graph
graph = create_graph(output)


if __name__ == "__main__":
    result = run_graph(graph)
    print("Product Description Generated:")
    print(result['product_descriptions'])
