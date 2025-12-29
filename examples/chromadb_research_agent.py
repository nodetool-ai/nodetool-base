"""
Chroma backed research agent – DSL workflow (query side)

Query an existing Chroma collection of papers using HybridSearch and a
research oriented Agent.
"""

from nodetool.dsl.graph import create_graph, run_graph
from nodetool.dsl.nodetool.input import StringInput
from nodetool.dsl.vector.chroma import HybridSearch
from nodetool.dsl.nodetool.text import FormatText
from nodetool.dsl.nodetool.agents import Agent
from nodetool.dsl.nodetool.output import Output
from nodetool.metadata.types import LanguageModel, Collection, Provider

question = StringInput(
    name="research_question",
    description="Research question about the indexed papers",
    value="How do recent transformer variants improve long context modeling?",
)

search = HybridSearch(
    text=question.output,
    collection=Collection(
        type="collection",
        name="test-papers",
    ),
    n_results=8,
    k_constant=60,
    min_keyword_length=3,
)

prompt = FormatText(
    template="""
You are a research assistant working over a Chroma collection of academic papers.

Question:
{{ question }}

You are given excerpts from relevant documents. Use them as the main ground truth.
If something is not supported by the excerpts, flag it as speculation.

Context snippets:
{% for doc in documents %}
---
{{ doc }}
{% endfor %}

Write:
- A concise explanation that answers the question.
- A bullet list of key findings.
- Short notes on open problems or limitations.

Respond in markdown.
""",
    question=question.output,
    documents=search.out.documents,
)

research_agent = Agent(
    prompt=prompt.output,
    model=LanguageModel(
        type="language_model",
        id="openai/gpt-oss-120b",
        provider=Provider.HuggingFaceCerebras,
    ),
    # default system prompt already encourages tool usage and structured thinking
    max_tokens=4096,
    context_window=4096,
)

output = Output(
    name="chroma_research_answer",
    value=research_agent.out.text,
)

graph = create_graph(output)

if __name__ == "__main__":
    result = run_graph(graph)
    print(result["chroma_research_answer"])
