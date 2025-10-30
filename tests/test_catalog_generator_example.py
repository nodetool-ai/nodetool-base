"""
Test for the catalog generator DSL example.
"""

import sys
import pytest
from pathlib import Path

# Add examples directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))

from catalog_generator import build_catalog_generator


def test_build_catalog_generator():
    """Test that the catalog generator builds successfully."""
    graph = build_catalog_generator()

    # Verify the graph structure
    assert graph is not None
    assert len(graph.nodes) == 3  # StringInput, DataGenerator, DataframeOutput
    assert len(graph.edges) == 2  # prompt_in -> generator, generator -> out_df

    # Verify node types
    node_types = {node.type for node in graph.nodes}
    expected_types = {
        "nodetool.input.StringInput",
        "nodetool.generators.DataGenerator",
        "nodetool.output.DataframeOutput",
    }
    assert node_types == expected_types


def test_catalog_generator_input_node():
    """Test that the input node is properly configured."""
    graph = build_catalog_generator()

    # Find the StringInput node
    input_nodes = [n for n in graph.nodes if n.type == "nodetool.input.StringInput"]
    assert len(input_nodes) == 1

    input_node = input_nodes[0]
    assert input_node.data["name"] == "catalog_prompt"
    assert input_node.data["description"] == "Describe the product domain, e.g. 'outdoor gear, hiking equipment'"


def test_catalog_generator_output_node():
    """Test that the output node is properly configured."""
    graph = build_catalog_generator()

    # Find the DataframeOutput node
    output_nodes = [n for n in graph.nodes if n.type == "nodetool.output.DataframeOutput"]
    assert len(output_nodes) == 1

    output_node = output_nodes[0]
    assert output_node.data["name"] == "catalog_dataframe"
    assert "Generated product catalog" in output_node.data["description"]


def test_catalog_generator_data_generator_node():
    """Test that the DataGenerator node has correct schema."""
    graph = build_catalog_generator()

    # Find the DataGenerator node
    generator_nodes = [n for n in graph.nodes if n.type == "nodetool.generators.DataGenerator"]
    assert len(generator_nodes) == 1

    generator_node = generator_nodes[0]

    # Verify columns schema
    columns = generator_node.data["columns"]
    assert columns["type"] == "record_type"

    column_names = {col["name"] for col in columns["columns"]}
    expected_columns = {"sku", "name", "category", "price", "short_description", "long_description", "image_prompt"}
    assert column_names == expected_columns

    # Verify model configuration
    model = generator_node.data["model"]
    assert model["type"] == "language_model"
    assert model["id"] == "ollama/mistral"
    # provider is an enum that gets serialized as its value
    assert str(model["provider"]).lower() == "provider.ollama"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
