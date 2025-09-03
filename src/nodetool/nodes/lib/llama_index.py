import os
from llama_index.core.node_parser import (
    SemanticSplitterNodeParser,
    HTMLNodeParser,
    JSONNodeParser,
)
from llama_index.core.schema import Document, TextNode
from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.ollama import OllamaEmbedding
from pydantic import Field
from nodetool.workflows.base_node import BaseNode
from nodetool.metadata.types import DocumentRef, LlamaModel, TextChunk, FolderPath
from nodetool.workflows.processing_context import ProcessingContext


class SemanticSplitter(BaseNode):
    """
    Split text semantically.
    chroma, embedding, collection, RAG, index, text, markdown, semantic
    """

    embed_model: LlamaModel = Field(
        default=LlamaModel(),
        description="Embedding model to use",
    )

    document: DocumentRef = Field(
        default=DocumentRef(),
        description="Document ID to associate with the text content",
    )
    buffer_size: int = Field(
        default=1,
        description="Buffer size for semantic splitting",
        ge=1,
        le=100,
    )
    threshold: int = Field(
        default=95,
        description="Breakpoint percentile threshold for semantic splitting",
        ge=0,
        le=100,
    )

    async def gen_process(self, context: ProcessingContext):
        assert self.embed_model.repo_id, "embed_model is required"

        splitter = SemanticSplitterNodeParser(
            buffer_size=self.buffer_size,
            breakpoint_percentile_threshold=self.threshold,
            embed_model=OllamaEmbedding(model_name=self.embed_model.repo_id),
        )

        documents = [Document(text=self.document.data, doc_id=self.document.uri)]

        # Split documents semantically
        nodes = splitter.build_semantic_nodes_from_documents(documents)

        # Convert nodes to TextChunks
        return [
            TextChunk(
                text=node.get_content(),
                source_id=self.document.uri,
                start_index=i,
            )
            for i, node in enumerate(nodes)
        ]


class HTMLSplitter(BaseNode):
    """
    Split HTML content into semantic chunks based on HTML tags.
    html, text, semantic, tags, parsing
    """

    document: DocumentRef = Field(
        default=DocumentRef(),
        description="Document ID to associate with the HTML content",
    )

    async def gen_process(self, context: ProcessingContext):
        parser = HTMLNodeParser(
            tags=[
                "p",
                "h1",
                "h2",
                "h3",
                "h4",
                "h5",
                "h6",
                "li",
                "b",
                "i",
                "u",
                "section",
            ],
            include_metadata=False,
            include_prev_next_rel=False,
        )

        doc = Document(text=self.document.data, doc_id=self.document.uri)

        nodes = parser.get_nodes_from_node(doc)

        return [
            TextChunk(
                text=node.text,
                source_id=self.document.uri,
                start_index=i,
            )
            for i, node in enumerate(nodes)
        ]


class JSONSplitter(BaseNode):
    """
    Split JSON content into semantic chunks.
    json, parsing, semantic, structured
    """

    document: DocumentRef = Field(
        default=DocumentRef(),
        description="Document ID to associate with the JSON content",
    )
    include_metadata: bool = Field(
        default=True, description="Whether to include metadata in nodes"
    )
    include_prev_next_rel: bool = Field(
        default=True, description="Whether to include prev/next relationships"
    )

    async def process(self, context: ProcessingContext) -> list[TextChunk]:
        parser = JSONNodeParser(
            include_metadata=self.include_metadata,
            include_prev_next_rel=self.include_prev_next_rel,
        )

        doc = Document(text=self.document.data, doc_id=self.document.uri)

        nodes = parser.get_nodes_from_node(doc)

        return [
            TextChunk(
                text=node.text,
                source_id=self.document.uri,
                start_index=i,
            )
            for i, node in enumerate(nodes)
        ]
