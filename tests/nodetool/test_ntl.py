"""
Tests for the NTL (NodeTool Language) parser and loader.
"""

import pytest
from pathlib import Path

from nodetool.ntl.parser import (
    parse_ntl,
    NTLParseError,
    NTLReference,
)
from nodetool.ntl.loader import (
    load_workflow_from_ntl,
)


class TestNTLLexer:
    """Tests for the NTL lexer."""

    def test_simple_string(self):
        """Test parsing a simple string value."""
        ast = parse_ntl('@name "Test Workflow"')
        assert ast.metadata.name == "Test Workflow"

    def test_string_with_escapes(self):
        """Test parsing strings with escape sequences."""
        ast = parse_ntl('@description "Line1\\nLine2\\tTabbed"')
        assert ast.metadata.description == "Line1\nLine2\tTabbed"

    def test_number_parsing(self):
        """Test parsing integer and float numbers."""
        source = """
node1: test.Node
  int_val = 42
  float_val = 3.14
  negative = -10
"""
        ast = parse_ntl(source)
        props = {p.name: p.value for p in ast.nodes[0].properties}
        assert props["int_val"] == 42
        assert props["float_val"] == 3.14
        assert props["negative"] == -10

    def test_boolean_parsing(self):
        """Test parsing boolean values."""
        source = """
node1: test.Node
  enabled = true
  disabled = false
"""
        ast = parse_ntl(source)
        props = {p.name: p.value for p in ast.nodes[0].properties}
        assert props["enabled"] is True
        assert props["disabled"] is False


class TestNTLParser:
    """Tests for the NTL parser."""

    def test_metadata_parsing(self):
        """Test parsing workflow metadata."""
        source = """
@name "My Workflow"
@description "This is a test workflow"
@tags image, processing, ai
"""
        ast = parse_ntl(source)
        assert ast.metadata.name == "My Workflow"
        assert ast.metadata.description == "This is a test workflow"
        assert ast.metadata.tags == ["image", "processing", "ai"]

    def test_simple_node(self):
        """Test parsing a simple node definition."""
        source = """
input: nodetool.input.StringInput
  name = "text"
  value = "hello world"
"""
        ast = parse_ntl(source)
        assert len(ast.nodes) == 1
        node = ast.nodes[0]
        assert node.id == "input"
        assert node.type == "nodetool.input.StringInput"
        assert len(node.properties) == 2

    def test_node_with_reference(self):
        """Test parsing a node with a reference to another node."""
        source = """
input: nodetool.input.StringInput
  value = "test"

output: nodetool.output.Output
  value = @input.output
"""
        ast = parse_ntl(source)
        assert len(ast.nodes) == 2
        output_node = ast.nodes[1]
        value_prop = output_node.properties[0]
        assert isinstance(value_prop.value, NTLReference)
        assert value_prop.value.node_id == "input"
        assert value_prop.value.handle == "output"

    def test_node_with_object(self):
        """Test parsing a node with object properties."""
        source = """
node1: test.Node
  config = {
    type: "model",
    id: "gpt-4",
    provider: "openai"
  }
"""
        ast = parse_ntl(source)
        node = ast.nodes[0]
        config = node.properties[0].value
        assert isinstance(config, dict)
        assert config["type"] == "model"
        assert config["id"] == "gpt-4"
        assert config["provider"] == "openai"

    def test_node_with_list(self):
        """Test parsing a node with list properties."""
        source = """
node1: test.Node
  items = [1, 2, 3, 4, 5]
  strings = ["a", "b", "c"]
"""
        ast = parse_ntl(source)
        node = ast.nodes[0]
        props = {p.name: p.value for p in node.properties}
        assert props["items"] == [1, 2, 3, 4, 5]
        assert props["strings"] == ["a", "b", "c"]

    def test_explicit_edge(self):
        """Test parsing explicit edge definitions."""
        source = """
input: nodetool.input.StringInput
output: nodetool.output.Output

input.output -> output.value
"""
        ast = parse_ntl(source)
        assert len(ast.edges) == 1
        edge = ast.edges[0]
        assert edge.source_node == "input"
        assert edge.source_handle == "output"
        assert edge.target_node == "output"
        assert edge.target_handle == "value"

    def test_comments(self):
        """Test that comments are properly ignored."""
        source = """
# This is a comment
@name "Test"  # Inline comment

/* Block comment
   spanning multiple lines */

node1: test.Node
  value = 42  # Another comment
"""
        ast = parse_ntl(source)
        assert ast.metadata.name == "Test"
        assert len(ast.nodes) == 1

    def test_nested_object(self):
        """Test parsing nested objects."""
        source = """
node1: test.Node
  model = {
    type: "language_model",
    config: {
      temperature: 0.7,
      max_tokens: 1000
    }
  }
"""
        ast = parse_ntl(source)
        model = ast.nodes[0].properties[0].value
        assert model["type"] == "language_model"
        assert model["config"]["temperature"] == 0.7
        assert model["config"]["max_tokens"] == 1000


class TestNTLLoader:
    """Tests for the NTL loader."""

    def test_load_simple_workflow(self):
        """Test loading a simple workflow."""
        source = """
@name "Test Workflow"
@description "A test workflow"

input: nodetool.input.StringInput
  name = "text"
  value = "hello"

output: nodetool.output.Output
  name = "result"
  value = @input.output
"""
        workflow = load_workflow_from_ntl(source)

        assert workflow["name"] == "Test Workflow"
        assert workflow["description"] == "A test workflow"
        assert len(workflow["graph"]["nodes"]) == 2
        assert len(workflow["graph"]["edges"]) == 1

    def test_node_structure(self):
        """Test that nodes have correct structure."""
        source = """
input: nodetool.input.StringInput
  name = "text"
"""
        workflow = load_workflow_from_ntl(source)
        node = workflow["graph"]["nodes"][0]

        assert node["id"] == "input"
        assert node["type"] == "nodetool.input.StringInput"
        assert node["data"]["name"] == "text"
        assert "ui_properties" in node
        assert "position" in node["ui_properties"]

    def test_edge_from_reference(self):
        """Test that references create edges."""
        source = """
node1: test.Input
  value = "test"

node2: test.Output
  input = @node1.output
"""
        workflow = load_workflow_from_ntl(source)
        edges = workflow["graph"]["edges"]

        assert len(edges) == 1
        edge = edges[0]
        assert edge["source"] == "node1"
        assert edge["sourceHandle"] == "output"
        assert edge["target"] == "node2"
        assert edge["targetHandle"] == "input"

    def test_explicit_and_implicit_edges(self):
        """Test combining explicit edges and reference edges."""
        source = """
a: test.Node
b: test.Node
  input = @a.output
c: test.Node

b.output -> c.input
"""
        workflow = load_workflow_from_ntl(source)
        edges = workflow["graph"]["edges"]

        assert len(edges) == 2

    def test_object_in_data(self):
        """Test that objects are preserved in node data."""
        source = """
node1: test.Node
  model = {
    type: "model",
    id: "test-model"
  }
"""
        workflow = load_workflow_from_ntl(source)
        node = workflow["graph"]["nodes"][0]

        assert node["data"]["model"]["type"] == "model"
        assert node["data"]["model"]["id"] == "test-model"


class TestNTLErrorHandling:
    """Tests for error handling."""

    def test_unterminated_string(self):
        """Test error on unterminated string."""
        with pytest.raises(NTLParseError) as exc_info:
            parse_ntl('@name "unclosed')
        assert "Unterminated string" in str(exc_info.value)

    def test_unexpected_character(self):
        """Test error on unexpected character."""
        with pytest.raises(NTLParseError) as exc_info:
            parse_ntl("node1: test.Node\n  value = $invalid")
        assert "Unexpected character" in str(exc_info.value)

    def test_missing_colon_in_node(self):
        """Test error when node definition missing colon."""
        with pytest.raises(NTLParseError) as exc_info:
            parse_ntl("node1 test.Node")
        assert "Expected" in str(exc_info.value)


class TestNTLExamples:
    """Test loading the example NTL files."""

    @pytest.fixture
    def examples_dir(self):
        """Get the examples directory."""
        return Path(__file__).parent.parent / "src" / "nodetool" / "ntl" / "examples"

    def test_image_enhance_example(self, examples_dir):
        """Test loading the image enhance example."""
        ntl_file = examples_dir / "image_enhance.ntl"
        if ntl_file.exists():
            workflow = load_workflow_from_ntl(ntl_file)
            assert workflow["name"] == "Image Enhance"
            assert len(workflow["graph"]["nodes"]) == 4
            assert len(workflow["graph"]["edges"]) == 3

    def test_transcribe_audio_example(self, examples_dir):
        """Test loading the transcribe audio example."""
        ntl_file = examples_dir / "transcribe_audio.ntl"
        if ntl_file.exists():
            workflow = load_workflow_from_ntl(ntl_file)
            assert workflow["name"] == "Transcribe Audio"
            assert len(workflow["graph"]["nodes"]) == 3

    def test_simple_chat_example(self, examples_dir):
        """Test loading the simple chat example."""
        ntl_file = examples_dir / "simple_chat.ntl"
        if ntl_file.exists():
            workflow = load_workflow_from_ntl(ntl_file)
            assert workflow["name"] == "Simple Chat"


class TestNTLRoundTrip:
    """Test converting NTL to workflow and validating structure."""

    def test_workflow_has_required_fields(self):
        """Test that workflow has all required fields."""
        source = """
@name "Test"
node1: test.Node
"""
        workflow = load_workflow_from_ntl(source)

        required_fields = [
            "id",
            "name",
            "description",
            "tags",
            "graph",
            "input_schema",
            "output_schema",
        ]
        for field in required_fields:
            assert field in workflow

    def test_graph_has_nodes_and_edges(self):
        """Test that graph has nodes and edges lists."""
        source = """
node1: test.Node
"""
        workflow = load_workflow_from_ntl(source)

        assert "nodes" in workflow["graph"]
        assert "edges" in workflow["graph"]
        assert isinstance(workflow["graph"]["nodes"], list)
        assert isinstance(workflow["graph"]["edges"], list)

    def test_multiple_workflows(self):
        """Test parsing different workflow structures."""
        workflows = [
            """
@name "Empty"
""",
            """
@name "Single Node"
node: test.Node
""",
            """
@name "Chain"
a: test.Node
b: test.Node
  input = @a.output
c: test.Node
  input = @b.output
""",
        ]

        for source in workflows:
            workflow = load_workflow_from_ntl(source)
            assert "name" in workflow
            assert "graph" in workflow
