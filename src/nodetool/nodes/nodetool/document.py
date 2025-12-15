import datetime
import os
import glob
from typing import TypedDict
from nodetool.providers.base import AsyncGenerator
from nodetool.types.model import UnifiedModel
from pydantic import Field
from nodetool.config.environment import Environment
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext, create_file_uri
from nodetool.metadata.types import DocumentRef, HFTextGeneration, LanguageModel, Provider
from llama_index.core.node_parser import (
    SemanticSplitterNodeParser,
    HTMLNodeParser,
    JSONNodeParser,
)
from llama_index.core.schema import Document
from llama_index.embeddings.ollama import OllamaEmbedding
from pydantic import Field
from nodetool.workflows.base_node import BaseNode
from nodetool.metadata.types import DocumentRef, LlamaModel, TextChunk, FolderPath
from nodetool.workflows.processing_context import ProcessingContext
from pydantic import Field
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.metadata.types import DocumentRef, TextChunk
from nodetool.workflows.base_node import BaseNode


class LoadDocumentFile(BaseNode):
    """
    Read a document from disk.
    files, document, read, input, load, file
    """

    path: str = Field(default="", description="Path to the document to read")

    async def process(self, context: ProcessingContext) -> DocumentRef:
        if Environment.is_production():
            raise ValueError("This node is not available in production")
        if not self.path:
            raise ValueError("path cannot be empty")
        expanded_path = os.path.expanduser(self.path)
        return DocumentRef(uri=create_file_uri(expanded_path))


class SaveDocumentFile(BaseNode):
    """
    Write a document to disk.
    files, document, write, output, save, file

    The filename can include time and date variables:
    %Y - Year, %m - Month, %d - Day
    %H - Hour, %M - Minute, %S - Second
    """

    document: DocumentRef = Field(
        default=DocumentRef(), description="The document to save"
    )
    folder: str = Field(default="", description="Folder where the file will be saved")
    filename: str = Field(
        default="",
        description="Name of the file to save. Supports strftime format codes.",
    )

    async def process(self, context: ProcessingContext):
        if Environment.is_production():
            raise ValueError("This node is not available in production")
        if not self.folder:
            raise ValueError("folder cannot be empty")
        if not self.filename:
            raise ValueError("filename cannot be empty")

        expanded_folder = os.path.expanduser(self.folder)
        if not os.path.exists(expanded_folder):
            raise ValueError(f"Folder does not exist: {expanded_folder}")

        filename = datetime.datetime.now().strftime(self.filename)
        expanded_path = os.path.join(expanded_folder, filename)
        data = await context.asset_to_bytes(self.document)
        with open(expanded_path, "wb") as f:
            f.write(data)


class ListDocuments(BaseNode):
    """
    List documents in a directory.
    files, list, directory
    """

    folder: str = Field(default="~", description="Directory to scan")
    pattern: str = Field(default="*", description="File pattern to match (e.g. *.txt)")
    recursive: bool = Field(default=False, description="Search subdirectories")

    class OutputType(TypedDict):
        document: DocumentRef

    async def gen_process(
        self, context: ProcessingContext
    ) -> AsyncGenerator[OutputType, None]:
        if Environment.is_production():
            raise ValueError("This node is not available in production")
        if not self.folder:
            raise ValueError("directory cannot be empty")
        expanded_directory = os.path.expanduser(self.folder)

        if self.recursive:
            pattern = os.path.join(expanded_directory, "**", self.pattern)
            paths = glob.glob(pattern, recursive=True)
        else:
            pattern = os.path.join(expanded_directory, self.pattern)
            paths = glob.glob(pattern)

        for p in paths:
            yield {"document": DocumentRef(uri=create_file_uri(p))}


class SplitDocument(BaseNode):
    """
    Split text semantically.
    chroma, embedding, collection, RAG, index, text, markdown, semantic
    """

    embed_model: LanguageModel = Field(
        default=LanguageModel(
            type="language_model",
            provider=Provider.Ollama,
            id="embeddinggemma",
        ),
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

    @classmethod
    def unified_recommended_models(
        cls, include_model_info: bool = False
    ) -> list[UnifiedModel]:
        return [
            UnifiedModel(
                id="embeddinggemma",
                repo_id="embeddinggemma",
                name="Embedding Gemma",
                description="Embedding model for semantic splitting",
                type="embedding_model",
            ),
            UnifiedModel(
                id="nomic-embed-text",
                repo_id="nomic-embed-text",
                name="Nomic Embed Text",
                description="Embedding model for semantic splitting",
                type="embedding_model",
            ),
            UnifiedModel(
                id="mxbai-embed-large",
                repo_id="mxbai-embed-large",
                name="MXBai Embed Large",
                description="Embedding model for semantic splitting",
                type="embedding_model",
            ),
            UnifiedModel(
                id="bge-m3",
                repo_id="bge-m3",
                name="BGE M3",
                description="Embedding model for semantic splitting",
                type="embedding_model",
            ),
            UnifiedModel(
                id="all-minilm",
                repo_id="all-minilm",
                name="All Minilm",
                description="Embedding model for semantic splitting",
                type="embedding_model",
            ),
        ]

    class OutputType(TypedDict):
        text: str
        source_id: str
        start_index: int

    async def gen_process(
        self, context: ProcessingContext
    ) -> AsyncGenerator[OutputType, None]:
        assert self.embed_model.id, "embed_model is required"

        splitter = SemanticSplitterNodeParser(
            buffer_size=self.buffer_size,
            breakpoint_percentile_threshold=self.threshold,
            embed_model=OllamaEmbedding(model_name=self.embed_model.id),
        )

        documents = [Document(text=self.document.data, doc_id=self.document.uri)]

        # Split documents semantically
        nodes = splitter.build_semantic_nodes_from_documents(documents)

        # Convert nodes to TextChunks
        for i, node in enumerate(nodes):
            yield {
                "text": node.get_content(),
                "source_id": self.document.uri,
                "start_index": i,
            }


class SplitHTML(BaseNode):
    """
    Split HTML content into semantic chunks based on HTML tags.
    html, text, semantic, tags, parsing
    """

    document: DocumentRef = Field(
        default=DocumentRef(),
        description="Document ID to associate with the HTML content",
    )

    class OutputType(TypedDict):
        text: str
        source_id: str
        start_index: int

    async def gen_process(
        self, context: ProcessingContext
    ) -> AsyncGenerator[OutputType, None]:
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

        for i, node in enumerate(nodes):
            yield {
                "text": node.text,
                "source_id": self.document.uri,
                "start_index": i,
            }


class SplitJSON(BaseNode):
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

    class OutputType(TypedDict):
        text: str
        source_id: str
        start_index: int

    async def gen_process(
        self, context: ProcessingContext
    ) -> AsyncGenerator[OutputType, None]:
        parser = JSONNodeParser(
            include_metadata=self.include_metadata,
            include_prev_next_rel=self.include_prev_next_rel,
        )

        doc = Document(text=self.document.data, doc_id=self.document.uri)

        nodes = parser.get_nodes_from_node(doc)

        for i, node in enumerate(nodes):
            yield {
                "text": node.text,
                "source_id": self.document.uri,
                "start_index": i,
            }


class SplitRecursively(BaseNode):
    """
    Splits text recursively using LangChain's RecursiveCharacterTextSplitter.
    text, split, chunks

    Use cases:
    - Splitting documents while preserving semantic relationships
    - Creating chunks for language model processing
    - Handling text in languages with/without word boundaries
    """

    document: DocumentRef = Field(default=DocumentRef())
    chunk_size: int = Field(
        default=1000,
        description="Maximum size of each chunk in characters",
    )
    chunk_overlap: int = Field(
        title="Chunk Overlap",
        default=200,
        description="Number of characters to overlap between chunks",
    )
    separators: list[str] = Field(
        default=["\n\n", "\n", "."],
        description="List of separators to use for splitting, in order of preference",
    )

    @classmethod
    def get_title(cls):
        return "Split Recursively"

    class OutputType(TypedDict):
        text: str
        source_id: str
        start_index: int

    async def gen_process(
        self, context: ProcessingContext
    ) -> AsyncGenerator[OutputType, None]:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_core.documents import Document

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
            is_separator_regex=False,
            add_start_index=True,
        )

        docs = splitter.split_documents([Document(page_content=self.document.data)])

        for i, doc in enumerate(docs):
            yield {
                "text": doc.page_content,
                "source_id": f"{self.document.uri}:{i}",
                "start_index": doc.metadata.get("start_index", 0),
            }


class SplitMarkdown(BaseNode):
    """
    Splits markdown text by headers while preserving header hierarchy in metadata.
    markdown, split, headers

    Use cases:
    - Splitting markdown documentation while preserving structure
    - Processing markdown files for semantic search
    - Creating context-aware chunks from markdown content
    """

    document: DocumentRef = Field(default=DocumentRef())
    headers_to_split_on: list[tuple[str, str]] = Field(
        default=[
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ],
        description="List of tuples containing (header_symbol, header_name)",
    )
    strip_headers: bool = Field(
        default=True,
        description="Whether to remove headers from the output content",
    )
    return_each_line: bool = Field(
        default=False,
        description="Whether to split into individual lines instead of header sections",
    )
    chunk_size: int = Field(
        default=1000,
        description="Optional maximum chunk size for further splitting",
    )
    chunk_overlap: int = Field(
        default=30,
        description="Overlap size when using chunk_size",
    )

    @classmethod
    def get_title(cls):
        return "Split Markdown"

    class OutputType(TypedDict):
        text: str
        source_id: str
        start_index: int

    async def gen_process(
        self, context: ProcessingContext
    ) -> AsyncGenerator[OutputType, None]:
        from langchain_text_splitters import (
            MarkdownHeaderTextSplitter,
            RecursiveCharacterTextSplitter,
        )

        # Initialize markdown splitter
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on,
            strip_headers=self.strip_headers,
            return_each_line=self.return_each_line,
        )

        # Split by headers
        splits = markdown_splitter.split_text(self.document.data)

        # Further split by chunk size if specified
        if self.chunk_size:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
            )
            splits = text_splitter.split_documents(splits)

        # Convert Document objects to dictionaries
        for i, doc in enumerate(splits):
            yield {
                "text": doc.page_content,
                "source_id": self.document.uri,
                "start_index": doc.metadata.get("start_index", 0),
            }