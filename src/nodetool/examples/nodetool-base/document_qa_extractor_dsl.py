"""
Document Q&A Extractor DSL Example

Extract specific information and answer questions from documents using AI.

Workflow:
1. **Document Input** - Provide document text or URL
2. **Question Input** - Ask specific questions about the document
3. **Extract Relevant Sections** - Find relevant content
4. **Answer Generation** - Generate answers based on document
5. **Citation Extraction** - Include source citations
6. **Output Results** - Save structured Q&A pairs
"""

from nodetool.dsl.graph import create_graph, run_graph
from nodetool.dsl.nodetool.input import StringInput
from nodetool.dsl.nodetool.text import FormatText
from nodetool.dsl.nodetool.agents import Agent
from nodetool.dsl.nodetool.output import StringOutput
from nodetool.metadata.types import LanguageModel, Provider


# Sample document content
document = StringInput(
    name="document",
    description="Document text to extract information from",
    value="""
Machine Learning Fundamentals

Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience.

Key Concepts:
1. Supervised Learning - Learning from labeled data
2. Unsupervised Learning - Finding patterns in unlabeled data
3. Reinforcement Learning - Learning through interaction and rewards

Neural Networks:
Neural networks are computational models inspired by biological neurons. They consist of:
- Input layer
- Hidden layers
- Output layer
- Activation functions for non-linearity

Applications:
ML is used in computer vision, natural language processing, recommendation systems, and autonomous vehicles.
""",
)

# Questions to ask about the document
questions = StringInput(
    name="questions",
    description="Questions to answer from the document",
    value="What are the main types of machine learning? How do neural networks work?",
)

# Extract answers using AI
qa_extractor = Agent(
    prompt=FormatText(
        template="""Document:
{{ document }}

Questions:
{{ questions }}

Please answer each question based on the document content.""",
        document=document.output,
        questions=questions.output,
    ).output,
    model=LanguageModel(
        type="language_model",
        id="gpt-4o",
        provider=Provider.OpenAI,
    ),
    system="You are an expert document analyst. Answer questions accurately based on the provided document. Include specific references to the document.",
    max_tokens=1500,
)

# Format results with citations
formatted_qa = FormatText(
    template="""# Document Q&A Results

**Document Subject:** Machine Learning

## Questions Answered:
{{ questions }}

## Answers:
{{ answers }}

**Note:** All answers are derived from the provided document.""",
    questions=questions.output,
    answers=qa_extractor.out.text,
)

# Output the Q&A results
output = StringOutput(
    name="qa_results",
    value=formatted_qa.output,
)

# Create the graph
graph = create_graph(output)


if __name__ == "__main__":
    result = run_graph(graph)
    print("Document Q&A Results:")
    print(result['qa_results'])
