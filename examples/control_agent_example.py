"""
Control Agent Example - Dynamic Parameter Control via Control Edges

This example demonstrates how to use the ControlAgent to dynamically set parameters
for downstream nodes based on context analysis. The ControlAgent uses an LLM to
analyze input and determine appropriate control parameters.

Control Flow Pattern:
    [Input Context] -> [ControlAgent] ~~control edge~~> [Downstream Node]
                                                          (parameters overridden)

The control agent:
1. Receives context (text, data, or analysis results)
2. Uses an LLM to reason about appropriate parameters
3. Outputs control parameters via __control_output__
4. These parameters are routed via control edges to downstream nodes
5. Control edges override normal data inputs on the target node

Key Concept: Control Edges
- Control edges have edge_type="control"
- They carry parameter overrides from control agents to target nodes
- Parameters set via control edges take precedence over normal inputs
- Enables dynamic, context-aware workflow behavior

Note: This example shows the node structure. To use control edges in practice,
you need to connect the ControlAgent's __control_output__ to target nodes using
control edges (edge_type="control") in the nodetool-core workflow runner.
"""

from nodetool.dsl.graph import create_graph
from nodetool.dsl.nodetool.input import StringInput
from nodetool.dsl.nodetool.agents import ControlAgent
from nodetool.dsl.nodetool.output import Output
from nodetool.metadata.types import LanguageModel, Provider


# --- Example 1: Image Processing Control ---
# Analyze image characteristics and set processing parameters dynamically

image_description = StringInput(
    name="image_description",
    description="Description or analysis of the image",
    value="The image is very dark with low contrast. "
          "It appears to be taken indoors with poor lighting. "
          "The subject is barely visible.",
)

image_control_agent = ControlAgent(
    model=LanguageModel(
        type="language_model",
        provider=Provider.Ollama,
        id="qwen3:4b",
    ),
    system="You are an image processing control agent. Analyze the image description "
           "and determine optimal processing parameters.",
    context=image_description.output,
    schema_description="""
Expected parameters:
- brightness: int (-100 to +100) - brightness adjustment
- contrast: int (-100 to +100) - contrast adjustment
- saturation: int (-100 to +100) - color saturation adjustment
- sharpen: float (0.0 to 2.0) - sharpening amount
    """,
)

# The control parameters output would be connected via a control edge to an
# image processing node like ImageAdjust or ImageEnhance
# Example control output: {"brightness": 40, "contrast": 30, "saturation": 10, "sharpen": 1.2}

output_image_params = Output(
    name="image_control_params",
    value=image_control_agent.out.__control_output__,
    description="Control parameters for image processing",
)


# --- Example 2: Text Generation Control ---
# Analyze content requirements and set generation parameters

content_requirements = StringInput(
    name="content_requirements",
    description="Requirements for text generation",
    value="Create a short, casual social media post about a new product launch. "
          "Target audience is young adults. Keep it friendly and engaging.",
)

text_control_agent = ControlAgent(
    model=LanguageModel(
        type="language_model",
        provider=Provider.Ollama,
        id="qwen3:4b",
    ),
    system="You are a text generation control agent. Analyze the content requirements "
           "and determine optimal generation parameters.",
    context=content_requirements.output,
    schema_description="""
Expected parameters:
- temperature: float (0.0 to 2.0) - creativity level
- max_tokens: int (50 to 1000) - response length
- tone: str - tone of voice (formal, casual, professional, friendly)
- style: str - writing style (concise, detailed, creative)
    """,
)

# The control parameters would be connected via control edge to a text generation node
# Example control output: {"temperature": 1.2, "max_tokens": 150, "tone": "friendly", "style": "concise"}

output_text_params = Output(
    name="text_control_params",
    value=text_control_agent.out.__control_output__,
    description="Control parameters for text generation",
)


# --- Example 3: Workflow Routing Control ---
# Analyze data and determine routing parameters

data_analysis = StringInput(
    name="data_analysis",
    description="Analysis of input data",
    value="The dataset contains 5000 records with 15% missing values. "
          "Data types are mixed (numerical and categorical). "
          "The target variable shows moderate class imbalance.",
)

workflow_control_agent = ControlAgent(
    model=LanguageModel(
        type="language_model",
        provider=Provider.Ollama,
        id="llama3.2:3b",
    ),
    system="You are a workflow control agent. Analyze the data characteristics "
           "and determine optimal processing parameters.",
    context=data_analysis.output,
    schema_description="""
Expected parameters:
- handle_missing: str (drop, impute, forward_fill)
- normalization: bool - whether to normalize numeric features
- encode_categoricals: bool - whether to encode categorical variables
- balance_classes: bool - whether to apply class balancing
- validation_split: float (0.1 to 0.3) - validation set size
    """,
)

# The control parameters would be connected via control edge to data processing nodes
# Example control output: {
#   "handle_missing": "impute",
#   "normalization": true,
#   "encode_categoricals": true,
#   "balance_classes": true,
#   "validation_split": 0.2
# }

output_workflow_params = Output(
    name="workflow_control_params",
    value=workflow_control_agent.out.__control_output__,
    description="Control parameters for data processing workflow",
)


# Build the graph
graph = create_graph([output_image_params, output_text_params, output_workflow_params])


if __name__ == "__main__":
    """
    To run this example:
    
    1. Ensure Ollama is running locally with the required models:
       - qwen3:4b (for control agents)
       - llama3.2:3b (alternative model)
    
    2. Run the script:
    
        python examples/control_agent_example.py
    
    The workflow demonstrates:
    - How ControlAgent analyzes context using an LLM
    - How control parameters are generated based on reasoning
    - Different use cases: image processing, text generation, workflow routing
    
    To use control edges in practice:
    - In nodetool-core, create edges with edge_type="control"
    - Connect ControlAgent's __control_output__ to target node's __control__ handle
    - Control parameters will override normal data inputs
    - Enables dynamic, intelligent workflow behavior
    """
    
    print("Control Agent Example")
    print("=" * 60)
    print(f"Nodes: {len(graph.nodes)}")
    print(f"Edges: {len(graph.edges)}")
    print()
    print("Control Agents:")
    print("  1. Image Processing Control - Analyzes image characteristics")
    print("  2. Text Generation Control - Analyzes content requirements")
    print("  3. Workflow Routing Control - Analyzes data characteristics")
    print()
    print("Each agent outputs control parameters via __control_output__")
    print("These parameters can be routed via control edges to downstream nodes")
    print()
    
    # To actually run the agents and see control parameters, you would need:
    # from nodetool.dsl.graph import run_graph
    # result = run_graph(graph, user_id="example_user", auth_token="token")
    # print(f"Image params: {result['image_control_params']}")
    # print(f"Text params: {result['text_control_params']}")
    # print(f"Workflow params: {result['workflow_control_params']}")
