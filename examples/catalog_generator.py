"""
Example: Catalog Generator DSL Graph

This example demonstrates how to use the NodeTool DSL to create a product catalog
generator. It shows how to:
1. Create inputs using DSL nodes (StringInput)
2. Define a data schema with RecordType and ColumnDef
3. Use DataGenerator to create structured data from LLM
4. Output the result as a DataFrame

The graph can be executed to generate a product catalog in structured tabular format.
"""

import asyncio
from nodetool.dsl.graph import create_graph, run_graph
from nodetool.dsl.nodetool.input import StringInput
from nodetool.dsl.nodetool.output import Output
from nodetool.dsl.nodetool.generators import DataGenerator
from nodetool.metadata.types import RecordType, ColumnDef, LanguageModel, Provider


# Define the language model to use for generation
LLM = LanguageModel(
    type="language_model",
    provider=Provider.Ollama,
    id="ollama/mistral",
)


def build_catalog_generator():
    """
    Generate full product catalogs with structured metadata.

    This function builds a workflow graph that:
    1. Accepts a text prompt describing the product domain
    2. Uses an LLM (Ollama/Mistral) to generate product data
    3. Returns a structured DataFrame with product information

    Returns:
        Graph: A graph object representing the workflow
    """
    # --- Inputs ---
    # Create a string input node that accepts a catalog description
    prompt_in = StringInput(
        name="catalog_prompt",
        description="Describe the product domain, e.g. 'outdoor gear, hiking equipment'",
        value=""
    )

    # --- Schema definition ---
    # Define the structure of the generated product records
    schema = RecordType(columns=[
        ColumnDef(name="sku", data_type="string"),
        ColumnDef(name="name", data_type="string"),
        ColumnDef(name="category", data_type="string"),
        ColumnDef(name="price", data_type="float"),
        ColumnDef(name="short_description", data_type="string"),
        ColumnDef(name="long_description", data_type="string"),
        ColumnDef(name="image_prompt", data_type="string"),
    ])

    # --- Data generation node ---
    # Create a generator node that will produce product data
    generator = DataGenerator(
        model=LLM,
        prompt=prompt_in.output,
        columns=schema,
    )

    # --- Output dataframe ---
    # Create an output node to surface the generated catalog
    out_df = Output(
        name="catalog_dataframe",
        value=generator.out.dataframe,
        description="Generated product catalog ready for CSV export."
    )

    # Return the graph representation
    return create_graph(out_df)


# Example execution (would typically be run async):
if __name__ == "__main__":
    """
    To run this example:

    1. Ensure Ollama is running locally with the Mistral model available
    2. Use the graph with run_graph():

    import asyncio
    from nodetool.dsl.graph import run_graph

    async def main():
        g = build_catalog_generator()
        result = await run_graph(
            g,
            user_id="example_user",
            auth_token="token"
        )
        print(result)

    asyncio.run(main())
    """

    # Build the catalog generator graph
    catalog_graph = build_catalog_generator()
    print("Catalog generator graph built successfully!")
    print(f"Graph nodes: {len(catalog_graph.nodes)}")
    print(f"Graph edges: {len(catalog_graph.edges)}")

    async def main():
        result = await run_graph(catalog_graph, user_id="example_user", auth_token="token")
        print(result)

    asyncio.run(main())
