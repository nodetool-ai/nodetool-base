"""
Chat with Docs DSL Example

An intelligent document retrieval and question-answering system that leverages vector search
and local LLMs to provide accurate, context-aware responses based on your document collection.

Workflow:
1. **String Input** - User query about documents
2. **Hybrid Search** - Vector search in ChromaDB collection
3. **Format Text** - Construct comprehensive prompt with context
4. **Agent** - AI agent that answers based on retrieved documents
5. **Preview** - Display the answer
"""

from nodetool.dsl.graph import graph_result, run_graph, graph
from nodetool.dsl.nodetool.input import StringInput
from nodetool.dsl.vector.chroma import HybridSearch
from nodetool.dsl.nodetool.text import FormatText
from nodetool.dsl.nodetool.agents import Agent
from nodetool.dsl.nodetool.output import StringOutput
from nodetool.metadata.types import LanguageModel, Collection


async def example():
    """
    Chat with documents using RAG (Retrieval Augmented Generation).
    """
    # User query input
    query_input = StringInput(
        name="Query",
        value="what are transformers?",
        description="",
    )

    # Hybrid search in vector database
    search = HybridSearch(
        text=query_input.output,
        collection=Collection(
            type="collection",
            name="test",
        ),
        n_results=5,
        k_constant=60,
        min_keyword_length=3,
    )

    # Format the prompt with retrieved documents
    prompt = FormatText(
        template="""You are an expert assistant answering a user question based on retrieved documents.
Your response must be accurate, well-structured, and grounded in the provided sources.

### User Question:
{{ text }}

### Relevant Information from Retrieved Documents
These snippets can be incomplete or out of context. Ignore information that is not relevant.
{% for doc in documents %}
=====================
**Document {{ loop.index }}**
{{ doc }}
{% endfor %}

### **Instructions for Answering:**
1. Analyze the user question carefully.
2. Extract key insights from the retrieved documents.
3. If multiple documents provide different perspectives, synthesize the most relevant and accurate information.
4. If the documents lack sufficient information, state what is missing instead of guessing.
5. Present the answer in a clear, concise, and structured manner.""",
        # Dynamic properties for the template variables
        text=query_input.output,
        documents=search.out.documents,
    )

    # AI agent to generate answer
    agent = Agent(
        prompt=prompt.output,
        model=LanguageModel(
            type="language_model",
            id="openai/gpt-oss-120b",
            provider="huggingface_cerebras",
            name="gpt-oss-120b",
        ),
        system="You are an AI agent. Understand the user's intent, break down tasks into steps, and be precise. Use tools to accomplish your goal.",
        max_tokens=8192,
        context_window=4096,
    )

    # Output the answer
    output = StringOutput(
        name="answer",
        value=agent.out.text,
    )

    result = await graph_result(output)
    return result


async def example_streaming():
    """
    Alternative version showing the full workflow with all nodes.
    """
    query_input = StringInput(
        name="Query",
        value="what are transformers?",
        description="",
    )

    search = HybridSearch(
        text=query_input.output,
        collection=Collection(type="collection", name="test"),
        n_results=5,
        k_constant=60,
        min_keyword_length=3,
    )

    prompt = FormatText(
        template="""You are an expert assistant answering questions based on retrieved documents.

### User Question:
{{ text }}

### Relevant Documents:
{% for doc in documents %}
Document {{ loop.index }}: {{ doc }}
{% endfor %}

Answer based only on the provided documents.""",
        text=query_input.output,
        documents=search.out.documents,
    )

    agent = Agent(
        prompt=prompt.output,
        model=LanguageModel(
            type="language_model",
            id="openai/gpt-oss-120b",
            provider="huggingface_cerebras",
            name="gpt-oss-120b",
        ),
        system="You are a helpful AI assistant. Answer questions based on the provided context.",
        max_tokens=8192,
        context_window=4096,
    )

    output = StringOutput(
        name="answer",
        value=agent.out.text,
    )
    g = graph(output)
    result = await run_graph(g)
    return result


if __name__ == "__main__":
    import asyncio

    result = asyncio.run(example())
    print(f"Answer: {result}")
