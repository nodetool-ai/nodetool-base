"""
Data Generator DSL Example

Generate structured data using AI agents to create synthetic datasets.

Features:
- Generate synthetic data using AI models
- Use prompts to describe the data structure
- Specify columns and data types for the generated output
"""

from nodetool.dsl.graph import graph_result
from nodetool.dsl.nodetool.generators import DataGenerator
from nodetool.dsl.nodetool.output import DataframeOutput
from nodetool.metadata.types import LanguageModel, RecordType, ColumnDef


async def example():
    """
    Generate a table of vegetables with names and colors.
    """
    # Define the data structure
    columns = RecordType(
        columns=[
            ColumnDef(name="name", data_type="string", description=""),
            ColumnDef(name="color", data_type="string", description=""),
        ]
    )

    # Create data generator node
    data_gen = DataGenerator(
        model=LanguageModel(
            type="language_model",
            id="gemma3:4b",
            provider="ollama",
        ),
        prompt="Generate a table of veggies",
        input_text="",
        max_tokens=4096,
        columns=columns,
    )

    # Output the generated dataframe
    output = DataframeOutput(
        name="dataframe_output",
        value=data_gen,
    )

    result = await graph_result(output)
    return result


if __name__ == "__main__":
    import asyncio

    result = asyncio.run(example())
    print(f"Generated data: {result}")
