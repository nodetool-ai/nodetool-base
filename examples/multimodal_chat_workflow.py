"""
Example: Multimodal Chat Workflow

This example demonstrates a multimodal chat workflow that can handle both text
and images using the NodeTool DSL. It extends the simple chat pattern to support
vision capabilities:

    [MessageInput].value -> [MessageDeconstructor]
                                  |-> .text  -> text.[Agent]
                                  |-> .image -> image.[Agent]
                                              -> [Output]

The workflow:
1. Accepts a chat message that may contain text and/or an image
2. Deconstructs the message to extract both text and image content
3. Passes both to a vision-capable Agent for processing
4. Returns the agent's response

This pattern is essential for building multimodal chatbots that can understand
and respond to both text queries and images.
"""

from nodetool.dsl.graph import create_graph
from nodetool.dsl.nodetool.input import MessageInput, MessageDeconstructor
from nodetool.dsl.nodetool.agents import Agent
from nodetool.dsl.nodetool.output import Output
from nodetool.metadata.types import LanguageModel, Provider


# --- Chat message input ---
# This node receives the incoming chat message which may contain text and/or an image
message_input = MessageInput(
    name="user_message",
    description="The incoming chat message from the user (may include text and/or image)",
)

# --- Message deconstructor ---
# Extract both text and image content from the message object
# MessageDeconstructor outputs: id, thread_id, role, text, image, audio, model
message_deconstructor = MessageDeconstructor(
    value=message_input.output,
)

# --- Multimodal Agent ---
# Process both text and image inputs to generate a response
# Using a model that can handle image inputs (when images are provided via MessageDeconstructor)
vision_agent = Agent(
    model=LanguageModel(
        type="language_model",
        provider=Provider.Ollama,
        id="gemma3:4b",
    ),
    system="""You are a helpful multimodal assistant that can understand both text and images.

When an image is provided:
- Describe what you see in the image
- Answer questions about the image content
- Provide relevant insights based on the visual content

When only text is provided:
- Respond helpfully to the user's query

Always be informative and accurate in your responses.""",
    prompt=message_deconstructor.out.text,
    image=message_deconstructor.out.image,
)

# --- Output ---
# Return the agent's response
output = Output(
    name="assistant_response",
    value=vision_agent.out.text,
    description="The assistant's response to the multimodal message",
)

# Build the graph
graph = create_graph(output)


if __name__ == "__main__":
    """
    To run this example:
    
    1. Ensure Ollama is running locally with a vision-capable model (gemma3:4b)
    2. Run the script:
    
        python examples/multimodal_chat_workflow.py
    
    The workflow demonstrates the multimodal pattern:
        MessageInput -> MessageDeconstructor -> (text + image) -> Agent -> Output
    """
    
    print("Multimodal Chat Workflow Graph")
    print("=" * 50)
    print(f"Nodes: {len(graph.nodes)}")
    print(f"Edges: {len(graph.edges)}")
    print()
    print("Workflow pattern:")
    print("  [MessageInput].value")
    print("      -> [MessageDeconstructor]")
    print("          |-> .text  -> text.[Agent]")
    print("          |-> .image -> image.[Agent]")
    print("              -> [Output]")
    print()
    
    # To run the graph, uncomment the following:
    # from nodetool.dsl.graph import run_graph
    # result = run_graph(graph, user_id="example_user", auth_token="token")
    # print(f"Response: {result['assistant_response']}")
