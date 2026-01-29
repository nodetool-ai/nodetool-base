"""
Example: Simple Chat Workflow

This example demonstrates a basic chat workflow pattern using the NodeTool DSL.
It shows the typical pipeline for chat-based applications:

    [MessageInput].value -> [MessageDeconstructor].text -> text.[Agent] -> [Output]

The workflow:
1. Accepts a chat message as input via MessageInput
2. Deconstructs the message to extract the text content
3. Passes the text to an Agent for processing
4. Returns the agent's response as output

This pattern is fundamental for building conversational interfaces and chatbots.
"""

from nodetool.dsl.graph import create_graph
from nodetool.dsl.nodetool.input import MessageInput, MessageDeconstructor
from nodetool.dsl.nodetool.agents import Agent
from nodetool.dsl.nodetool.output import Output
from nodetool.metadata.types import LanguageModel, Provider


# --- Chat message input ---
# This node receives the incoming chat message from the user
message_input = MessageInput(
    name="user_message",
    description="The incoming chat message from the user",
)

# --- Message deconstructor ---
# Extract the text content from the message object
# This separates the actual text from message metadata (role, thread_id, etc.)
message_deconstructor = MessageDeconstructor(
    value=message_input.output,
)

# --- Agent node ---
# Process the extracted text and generate a response
# The agent receives the deconstructed text as its prompt
chat_agent = Agent(
    model=LanguageModel(
        type="language_model",
        provider=Provider.Ollama,
        id="llama3.2:3b",
    ),
    system="You are a helpful assistant. Respond to user messages in a friendly and informative manner.",
    prompt=message_deconstructor.out.text,
)

# --- Output ---
# Return the agent's response
output = Output(
    name="assistant_response",
    value=chat_agent.out.text,
    description="The assistant's response to the user message",
)

# Build the graph
graph = create_graph(output)


if __name__ == "__main__":
    """
    To run this example:
    
    1. Ensure Ollama is running locally with the llama3.2:3b model
    2. Run the script:
    
        python examples/simple_chat_workflow.py
    
    The workflow demonstrates the basic pattern:
        MessageInput -> MessageDeconstructor -> Agent -> Output
    """
    
    print("Simple Chat Workflow Graph")
    print("=" * 50)
    print(f"Nodes: {len(graph.nodes)}")
    print(f"Edges: {len(graph.edges)}")
    print()
    print("Workflow pattern:")
    print("  [MessageInput].value")
    print("      -> [MessageDeconstructor].text")
    print("          -> text.[Agent]")
    print("              -> [Output]")
    print()
    
    # To run the graph, uncomment the following:
    # from nodetool.dsl.graph import run_graph
    # result = run_graph(graph, user_id="example_user", auth_token="token")
    # print(f"Response: {result['assistant_response']}")
