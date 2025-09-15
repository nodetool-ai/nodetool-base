from enum import Enum
import os
import platform
from nodetool.metadata.types import FolderPath, LeannSearchResult, TextChunk
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from pydantic import Field, FilePath


class LeannSearcher(BaseNode):
    """
    Search high-performance vector indexes for Retrieval-Augmented Generation (RAG) using LEANN.
    embedding, collection, RAG, index, text, chunk, batch, vector, search, leann
    """

    class PruningStrategy(str, Enum):
        _global = "global"
        local = "local"
        proportional = "proportional"

    folder: FolderPath = Field(
        default=FolderPath(),
        description="The folder where the index is stored",
    )
    name: str = Field(
        default="",
        description="The name of the index",
    )
    query: str = Field(
        default="",
        description="The query to search for",
    )
    top_k: int = Field(
        default=5,
        description="The number of results to return",
    )
    complexity: int = Field(
        default=64,
        description="The complexity of the search",
    )
    beam_width: int = Field(
        default=1,
        description="The beam width of the search",
    )
    prune_ratio: float = Field(
        default=0.0,
        description="The prune ratio of the search",
    )
    recompute_embeddings: bool = Field(
        default=True,
        description="Whether to recompute the embeddings",
    )
    pruning_strategy: PruningStrategy = Field(
        default=PruningStrategy._global,
        description="The pruning strategy of the search",
    )

    async def process(self, context: ProcessingContext) -> list[LeannSearchResult]:
        if platform.system() == "Windows":
            raise ValueError("Leann is not supported on Windows")

        import leann

        searcher = leann.LeannSearcher(
            index_path=os.path.join(self.folder.path, self.name)
        )
        results = searcher.search(
            query=self.query,
            top_k=self.top_k,
            complexity=self.complexity,
            beam_width=self.beam_width,
            prune_ratio=self.prune_ratio,
            recompute_embeddings=self.recompute_embeddings,
            pruning_strategy=self.pruning_strategy.value,
        )
        return [LeannSearchResult(**result.__dict__) for result in results]


class LeannBuilder(BaseNode):
    """
    Create high-performance vector indexes for Retrieval-Augmented Generation (RAG) using LEANN.
    embedding, collection, RAG, index, text, chunk, batch, vector, search, leann

    Embedding Model Selection: Understanding the Trade-offs
    Based on extensive testing with LEANN, embedding models fall into three performance tiers:

    🚀 SMALL MODELS (< 100M parameters)
    - sentence-transformers/all-MiniLM-L6-v2 (22M params)
    🎯 BEST FOR: Prototyping, simple queries, interactive demos, resource-constrained environments

    ⚖️ MEDIUM MODELS (100M-500M parameters)
    - facebook/contriever (110M params)
    - BAAI/bge-base-en-v1.5 (109M params)
    🎯 BEST FOR: Production RAG applications, balanced quality-performance needs

    🏆 LARGE MODELS (500M+ parameters)
    - Qwen/Qwen3-Embedding-0.6B (600M params)
    - intfloat/multilingual-e5-large (560M params)
    🎯 BEST FOR: High-stakes applications, maximum accuracy requirements, production environments

    Index Backend Selection: Optimizing for Scale and Performance

    Choose the right backend based on your dataset size and performance requirements:

    ⚡ HNSW (Hierarchical Navigable Small World)
    🎯 BEST FOR: Datasets < 10M vectors, when memory isn't a bottleneck, default recommendation

    🚀 DISKANN (Disk-based Approximate Nearest Neighbor)
    🎯 BEST FOR: Large datasets (100k+ documents), production environments, when recompute=True
    """

    class EmbeddingModel(Enum):
        """ """

        sentence_transformers_all_minilm_l6_v2 = (
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        facebook_contriever_110m_params = "facebook/contriever"
        baai_bge_base_en_v1_5 = "BAAI/bge-base-en-v1.5"
        Qwen_Qwen3_Embedding_0_6B = "Qwen/Qwen3-Embedding-0.6B"
        intfloat_multilingual_e5_large = "intfloat/multilingual-e5-large"

    class Backend(Enum):
        hnsw = "hnsw"
        diskann = "diskann"

    folder: FolderPath = Field(
        default=FolderPath(),
        description="Output folder where the LEANN index will be stored. Choose a location with sufficient disk space for your dataset.",
    )
    name: str = Field(
        default="",
        description="Unique identifier for this index. Used for organizing and retrieving specific indexes.",
    )
    model: EmbeddingModel = Field(
        default=EmbeddingModel.sentence_transformers_all_minilm_l6_v2,
        description="Select embedding model based on your quality-speed trade-off. See EmbeddingModel documentation for detailed guidance.",
    )
    backend: Backend = Field(
        default=Backend.hnsw,
        description="Index backend optimized for your scale. HNSW for most use cases, DiskANN for large datasets. See Backend documentation for details.",
    )
    text_chunks: list[TextChunk] = Field(
        default=[],
        description="Text chunks to be indexed. Each chunk should contain: text content, unique source_id, and optional metadata for filtering.",
    )

    @classmethod
    def is_streaming_input(cls) -> bool:  # type: ignore[override]
        # Consume chunks as a stream; build the index once when input stream ends
        return True

    @classmethod
    def return_type(cls):
        return {"output": bool}

    async def run(self, context: ProcessingContext, inputs, outputs) -> None:
        if platform.system() == "Windows":
            raise ValueError("Leann is not supported on Windows")

        import leann

        builder = leann.LeannBuilder(
            backend_name=self.backend.value,
            model_name=self.model.value,
        )

        # Include any statically configured chunks as initial items
        for c in self.text_chunks:
            builder.add_text(c.text, {"source_id": c.source_id})

        # Drain the input stream until EOS across all handles
        if inputs is not None:
            async for item in inputs.stream("text_chunks"):
                assert isinstance(item, list)
                for it in item:
                    assert isinstance(it, TextChunk)
                    builder.add_text(it.text, {"source_id": it.source_id})
                    await outputs.emit("output", f"Added text chunk {it.source_id}")

        builder.build_index(os.path.join(self.folder.path, self.name))

        await outputs.emit("output", "Index built successfully")
