"""
Example: Advanced Chat Workflow with Thread Persistence

This example demonstrates a more complex chat workflow that includes:
- Message deconstruction for text and image extraction
- Thread persistence for conversation history
- Vision capabilities for image understanding
- Customizable system prompt for different behaviors

The workflow pattern:
    [MessageInput].value -> [MessageDeconstructor]
                                  |-> .text     -> text.[Agent]
                                  |-> .image    -> image.[Agent]
                                  |-> .thread_id -> thread_id.[Agent]
                                              -> [Output]

This advanced pattern is suitable for production chat applications that need:
- Persistent conversation history
- Multimodal message support (text + images)
- Conversation context management
"""

from nodetool.dsl.graph import create_graph
from nodetool.dsl.nodetool.input import MessageInput, MessageDeconstructor, StringInput
from nodetool.dsl.nodetool.agents import Agent, CreateThread
from nodetool.dsl.nodetool.output import Output
from nodetool.metadata.types import LanguageModel, Provider


# --- System prompt input ---
# Allow customization of the agent's behavior
system_prompt_input = StringInput(
    name="system_prompt",
    description="System prompt to customize the assistant's behavior",
    value="""You are a helpful, knowledgeable AI assistant.

Capabilities:
- Answer questions on a wide range of topics
- Analyze and describe images when provided
- Maintain context from the conversation history
- Provide thoughtful, accurate responses

Guidelines:
- Be concise but thorough
- If you don't know something, say so
- When analyzing images, describe what you observe
- Reference previous messages when relevant to the conversation"""
)

# --- Chat message input ---
# Receives the incoming message from the user
message_input = MessageInput(
    name="user_message",
    description="The incoming chat message (can include text and/or image)",
)

# --- Message deconstructor ---
# Extract all relevant fields from the message
message_deconstructor = MessageDeconstructor(
    value=message_input.output,
)

# --- Thread creation (optional) ---
# Create or reuse a thread for persistent conversation history
# The thread_id from the message can be used to continue a conversation
thread = CreateThread(
    title="Chat Conversation",
    thread_id=message_deconstructor.out.thread_id,
)

# --- Main Chat Agent ---
# Handles the conversation with support for text, images, and thread history
chat_agent = Agent(
    model=LanguageModel(
        type="language_model",
        provider=Provider.Ollama,
        id="gemma3:4b",
    ),
    system=system_prompt_input.output,
    prompt=message_deconstructor.out.text,
    image=message_deconstructor.out.image,
    thread_id=thread.out.thread_id,
)

# --- Output ---
output = Output(
    name="assistant_response",
    value=chat_agent.out.text,
    description="The assistant's response to the chat message",
)

# Build the graph
graph = create_graph(output)


if __name__ == "__main__":
    """
    To run this example:
    
    1. Ensure Ollama is running locally with gemma3:4b model
    2. Run the script:
    
        python examples/advanced_chat_workflow.py
    
    Advanced workflow pattern:
    - System prompt customization
    - Message deconstruction (text, image, thread_id)
    - Thread persistence for conversation history
    - Multimodal Agent processing
    """
    
    print("Advanced Chat Workflow Graph")
    print("=" * 50)
    print(f"Nodes: {len(graph.nodes)}")
    print(f"Edges: {len(graph.edges)}")
    print()
    print("Workflow components:")
    print("  1. StringInput (system_prompt) - Customizable system behavior")
    print("  2. MessageInput (user_message) - Incoming chat message")
    print("  3. MessageDeconstructor - Extract text, image, thread_id")
    print("  4. CreateThread - Persistent conversation history")
    print("  5. Agent - Process message with multimodal support")
    print("  6. Output - Return assistant response")
    print()
    print("Edges (connections):")
    print("  - MessageInput.output -> MessageDeconstructor.value")
    print("  - MessageDeconstructor.text -> Agent.prompt")
    print("  - MessageDeconstructor.image -> Agent.image")
    print("  - MessageDeconstructor.thread_id -> CreateThread.thread_id")
    print("  - CreateThread.thread_id -> Agent.thread_id")
    print("  - StringInput.output -> Agent.system")
    print("  - Agent.text -> Output.value")
    print()
    
    # To run the graph, uncomment the following:
    # from nodetool.dsl.graph import run_graph
    # result = run_graph(graph, user_id="example_user", auth_token="token")
    # print(f"Response: {result['assistant_response']}")
