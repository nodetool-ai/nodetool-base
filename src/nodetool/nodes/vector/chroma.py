"""
Chroma nodes for Nodetool.
"""

import re
from enum import Enum
from typing import TYPE_CHECKING, TypedDict

from nodetool.config.environment import Environment
from nodetool.integrations.vectorstores.chroma.async_chroma_client import (
    get_async_collection,
    get_async_chroma_client,
)
from nodetool.metadata.types import (
    AssetRef,
    Collection,
    ImageRef,
    LlamaModel,
    NPArray,
    TextChunk,
    TextRef,
)
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from pydantic import Field

if TYPE_CHECKING:
    from ollama import AsyncClient


def get_ollama_client() -> "AsyncClient":
    from ollama import AsyncClient

    api_url = Environment.get("OLLAMA_API_URL")
    assert api_url, "OLLAMA_API_URL not set"

    return AsyncClient(api_url)


class ChromaNode(BaseNode):
    """Base class for vector database nodes.

    vector, base, database, chroma, faiss
    """

    @classmethod
    def is_cacheable(cls):
        return False

    @classmethod
    def is_visible(cls):
        return cls is not ChromaNode

    async def load_results(
        self, context: ProcessingContext, ids: list[str]
    ) -> list[AssetRef]:
        asset_refs = []
        for id in ids:
            asset = await context.find_asset(str(id))
            if asset is None:
                continue
            url = await context.get_asset_url(asset.id)
            if asset.content_type.startswith("image"):
                ref = ImageRef(asset_id=asset.id, uri=url)
                asset_refs.append(ref)
            if asset.content_type.startswith("text"):
                ref = TextRef(asset_id=asset.id, uri=url)
                asset_refs.append(ref)
        return asset_refs


# Collection Management Nodes
class CollectionNode(ChromaNode):
    """
    Get or create a named vector database collection for storing embeddings.
    vector, embedding, collection, RAG, get, create, chroma
    """

    name: str = Field(default="", description="The name of the collection to create")
    embedding_model: LlamaModel = Field(
        default=LlamaModel(),
        description="Model to use for embedding, search for nomic-embed-text and download it",
    )

    async def process(self, context: ProcessingContext) -> Collection:
        client = await get_async_chroma_client()
        await client.get_or_create_collection(
            name=self.name,
            metadata={"embedding_model": self.embedding_model.repo_id},
        )
        return Collection(name=self.name)


class Count(ChromaNode):
    """
    Count the number of documents in a collection.
    vector, embedding, collection, RAG, chroma
    """

    collection: Collection = Field(
        default=Collection(), description="The collection to count"
    )

    async def process(self, context: ProcessingContext) -> int:
        collection = await get_async_collection(self.collection.name)
        return await collection.count()


class GetDocuments(ChromaNode):
    """
    Get documents from a chroma collection.
    vector, embedding, collection, RAG, retrieve, chroma
    """

    collection: Collection = Field(
        default=Collection(), description="The collection to get"
    )

    ids: list[str] = Field(default=[], description="The ids of the documents to get")
    limit: int = Field(default=100, description="The limit of the documents to get")
    offset: int = Field(default=0, description="The offset of the documents to get")

    async def process(self, context: ProcessingContext) -> list[str]:
        collection = await get_async_collection(self.collection.name)
        result = await collection.get(
            ids=self.ids,
            limit=self.limit,
            offset=self.offset,
        )
        assert result["documents"] is not None
        return result["documents"]


class Peek(ChromaNode):
    """
    Peek at the documents in a collection.
    vector, embedding, collection, RAG, preview, chroma
    """

    collection: Collection = Field(
        default=Collection(), description="The collection to peek"
    )
    limit: int = Field(default=100, description="The limit of the documents to peek")

    async def process(self, context: ProcessingContext) -> list[str]:
        collection = await get_async_collection(self.collection.name)
        result = await collection.peek(limit=self.limit)
        assert result["documents"] is not None
        return result["documents"]


# Indexing Nodes
class IndexImage(ChromaNode):
    """
    Index a list of image assets or files.
    vector, embedding, collection, RAG, index, image, batch, chroma
    """

    collection: Collection = Field(
        default=Collection(), description="The collection to index"
    )
    image: ImageRef = Field(default=[], description="List of image assets to index")
    index_id: str = Field(
        default="",
        description="The ID to associate with the image, defaults to the URI of the image",
    )
    metadata: dict = Field(
        default={}, description="The metadata to associate with the image"
    )
    upsert: bool = Field(default=False, description="Whether to upsert the images")

    @classmethod
    def required_inputs(cls):
        return [
            "image",
        ]

    async def process(self, context: ProcessingContext):
        if self.image.document_id is None:
            raise ValueError("document_id cannot be None for any image")

        import numpy as np

        collection = await get_async_collection(self.collection.name)
        image = await context.image_to_pil(self.image)
        image_ids = [self.index_id or self.image.document_id]
        image_arrays = [np.array(image)]
        metadata = self.metadata or {}

        if self.upsert:
            await collection.upsert(
                ids=image_ids, images=image_arrays, metadatas=[metadata]
            )
        else:
            await collection.add(
                ids=image_ids, images=image_arrays, metadatas=[metadata]
            )


class IndexEmbedding(ChromaNode):
    """
    Index a single embedding vector into a Chroma collection with optional metadata. Creates a searchable entry that can be queried for similarity matching.
    vector, index, embedding, chroma, storage, RAG
    """

    collection: Collection = Field(
        default=Collection(), description="The collection to index"
    )
    embedding: NPArray = Field(default=NPArray(), description="The embedding to index")
    index_id: str = Field(
        default="", description="The ID to associate with the embedding"
    )
    metadata: dict = Field(
        default={}, description="The metadata to associate with the embedding"
    )

    @classmethod
    def required_inputs(cls):
        return ["embedding", "id"]

    async def process(self, context: ProcessingContext):
        if self.index_id.strip() == "":
            raise ValueError("The ID cannot be empty")

        if self.embedding.is_empty():
            raise ValueError("The embedding cannot be empty")

        collection = await get_async_collection(self.collection.name)
        await collection.add(
            ids=[self.index_id],
            embeddings=[self.embedding.to_numpy()],
            metadatas=[self.metadata or {}],
        )


class IndexTextChunk(ChromaNode):
    """
    Index a single text chunk.
    vector, embedding, collection, RAG, index, text, chunk, chroma
    """

    collection: Collection = Field(
        default=Collection(), description="The collection to index"
    )
    document_id: str = Field(
        default="", description="The document ID to associate with the text chunk"
    )
    text: str = Field(default="", description="The text to index")
    metadata: dict = Field(
        default={},
        description="The metadata to associate with the text chunk",
    )

    async def process(self, context: ProcessingContext):
        if not self.document_id.strip():
            raise ValueError("The document ID cannot be empty")

        collection = await get_async_collection(self.collection.name)
        await collection.add(
            ids=[self.document_id],
            documents=[self.text],
            metadatas=[self.metadata or {}],
        )


class EmbeddingAggregation(Enum):
    MEAN = "mean"
    MAX = "max"
    MIN = "min"
    SUM = "sum"


class IndexAggregatedText(ChromaNode):
    """
    Index multiple text chunks at once with aggregated embeddings from Ollama.
    vector, embedding, collection, RAG, index, text, chunk, batch, ollama, chroma
    """

    collection: Collection = Field(
        default=Collection(), description="The collection to index"
    )
    document: str = Field(default="", description="The document to index")
    document_id: str = Field(
        default="", description="The document ID to associate with the text"
    )
    metadata: dict = Field(
        default={}, description="The metadata to associate with the text"
    )
    text_chunks: list[TextChunk | str] = Field(
        default=[], description="List of text chunks to index"
    )
    context_window: int = Field(
        default=4096,
        ge=1,
        description="The context window size to use for the model",
    )
    aggregation: EmbeddingAggregation = Field(
        default=EmbeddingAggregation.MEAN,
        description="The aggregation method to use for the embeddings.",
    )

    @classmethod
    def required_inputs(cls):
        return ["document", "text_chunks"]

    async def process(self, context: ProcessingContext):
        if not self.document_id.strip():
            raise ValueError("The document ID cannot be empty")

        if not self.document.strip():
            raise ValueError("The document cannot be empty")

        if not self.text_chunks:
            raise ValueError("The text chunks cannot be empty")

        import numpy as np

        collection = await get_async_collection(self.collection.name)

        model = collection.metadata.get("embedding_model")
        if not model:
            raise ValueError("The collection does not have an embedding model")

        # Extract document IDs and texts from chunks
        texts = [
            chunk.text if isinstance(chunk, TextChunk) else chunk
            for chunk in self.text_chunks
        ]
        client = get_ollama_client()

        # Calculate embeddings for each chunk
        embeddings = []
        for text in texts:
            response = await client.embeddings(
                model=model,
                prompt=text,
                options={"num_ctx": self.context_window},
            )
            embeddings.append(response["embedding"])

        # Aggregate embeddings based on selected method
        if self.aggregation == EmbeddingAggregation.MEAN:
            aggregated_embedding = np.mean(embeddings, axis=0)
        elif self.aggregation == EmbeddingAggregation.MAX:
            aggregated_embedding = np.max(embeddings, axis=0)
        elif self.aggregation == EmbeddingAggregation.MIN:
            aggregated_embedding = np.min(embeddings, axis=0)
        elif self.aggregation == EmbeddingAggregation.SUM:
            aggregated_embedding = np.sum(embeddings, axis=0)
        else:
            raise ValueError(f"Invalid aggregation method: {self.aggregation}")

        await collection.add(
            ids=[self.document_id],
            documents=[self.document],
            embeddings=[aggregated_embedding],
            metadatas=[self.metadata] if self.metadata else None,
        )


class IndexString(ChromaNode):
    """
    Index a string with a Document ID to a collection.
    vector, embedding, collection, RAG, index, text, string, chroma
    """

    collection: Collection = Field(
        default=Collection(), description="The collection to index"
    )
    text: str = Field(default="", description="Text content to index")
    document_id: str = Field(
        default="", description="Document ID to associate with the text content"
    )
    metadata: dict = Field(
        default={}, description="The metadata to associate with the text"
    )

    async def process(self, context: ProcessingContext):
        if not self.document_id.strip():
            raise ValueError("The document ID cannot be empty")

        collection = await get_async_collection(self.collection.name)
        await collection.add(ids=[self.document_id], documents=[self.text])


# Query Nodes
class QueryImage(ChromaNode):
    """
    Query the index for similar images.
    vector, RAG, query, image, search, similarity, chroma
    """

    collection: Collection = Field(
        default=Collection(), description="The collection to query"
    )
    image: ImageRef = Field(default=ImageRef(), description="The image to query")
    n_results: int = Field(default=1, description="The number of results to return")

    class OutputType(TypedDict):
        ids: list[str]
        documents: list[str]
        metadatas: list[dict]
        distances: list[float]

    async def process(self, context: ProcessingContext) -> OutputType:
        if not self.image.asset_id and not self.image.uri:
            raise ValueError("Image is not connected")

        import numpy as np

        collection = await get_async_collection(self.collection.name)
        image = await context.image_to_pil(self.image)
        result = await collection.query(
            query_images=[np.array(image)], n_results=self.n_results
        )
        assert result["ids"] is not None, "Ids are not returned"
        assert result["documents"] is not None, "Documents are not returned"
        assert result["metadatas"] is not None, "Metadatas are not returned"
        assert result["distances"] is not None, "Distances are not returned"

        # Create list of tuples to sort together
        combined = list(
            zip(
                result["ids"][0],
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
            )
        )
        # Sort by ID
        combined.sort(key=lambda x: str(x[0]))

        # Unzip the sorted results
        ids, documents, metadatas, distances = zip(*combined)
        ids = [str(id) for id in ids]

        return {
            "ids": ids,
            "documents": list(documents),
            "metadatas": list(metadatas),
            "distances": list(distances),
        }


class QueryText(ChromaNode):
    """
    Query the index for similar text.
    vector, RAG, query, text, search, similarity, chroma
    """

    collection: Collection = Field(
        default=Collection(), description="The collection to query"
    )
    text: str = Field(default="", description="The text to query")
    n_results: int = Field(default=1, description="The number of results to return")

    class OutputType(TypedDict):
        ids: list[str]
        documents: list[str]
        metadatas: list[dict]
        distances: list[float]

    async def process(self, context: ProcessingContext) -> OutputType:
        collection = await get_async_collection(self.collection.name)
        result = await collection.query(
            query_texts=[self.text], n_results=self.n_results
        )

        assert result["ids"] is not None, "Ids are not returned"
        assert result["documents"] is not None, "Documents are not returned"
        assert result["metadatas"] is not None, "Metadatas are not returned"
        assert result["distances"] is not None, "Distances are not returned"

        # Create list of tuples to sort together
        combined = list(
            zip(
                result["ids"][0],
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
            )
        )
        # Sort by ID
        combined.sort(key=lambda x: str(x[0]))

        # Unzip the sorted results
        ids, documents, metadatas, distances = zip(*combined)
        ids = [str(id) for id in ids]

        return {
            "ids": ids,
            "documents": list(documents),
            "metadatas": list(metadatas),
            "distances": list(distances),
        }


class RemoveOverlap(ChromaNode):
    """
    Removes overlapping words between consecutive strings in a list. Splits text into words and matches word sequences for more accurate overlap detection.
    vector, RAG, query, text, processing, overlap, deduplication
    """

    documents: list[str] = Field(
        default=[],
        description="List of strings to process for overlap removal",
    )
    min_overlap_words: int = Field(
        default=2,
        description="Minimum number of words that must overlap to be considered",
    )

    class OutputType(TypedDict):
        documents: list[str]

    def _split_into_words(self, text: str) -> list[str]:
        """Split text into words, preserving spacing."""
        return text.split()

    def _find_word_overlap(self, words1: list[str], words2: list[str]) -> int:
        """Find the number of overlapping words between the end of words1 and start of words2."""
        if len(words1) < self.min_overlap_words or len(words2) < self.min_overlap_words:
            return 0

        # Start with maximum possible overlap
        max_check = min(len(words1), len(words2))

        for overlap_size in range(max_check, self.min_overlap_words - 1, -1):
            if words1[-overlap_size:] == words2[:overlap_size]:
                return overlap_size
        return 0

    async def process(self, context: ProcessingContext) -> OutputType:
        if not self.documents:
            return {"documents": []}

        result = [self.documents[0]]

        for i in range(1, len(self.documents)):
            prev_words = self._split_into_words(result[-1])
            curr_words = self._split_into_words(self.documents[i])

            overlap_word_count = self._find_word_overlap(prev_words, curr_words)

            if overlap_word_count > 0:
                # Reconstruct the text without the overlapping words
                new_text = " ".join(curr_words[overlap_word_count:])
                if new_text:
                    result.append(new_text)
            else:
                result.append(self.documents[i])

        return {"documents": result}


class HybridSearch(ChromaNode):
    """
    Hybrid search combining semantic and keyword-based search for better retrieval. Uses reciprocal rank fusion to combine results from both methods.
    vector, RAG, query, semantic, text, similarity, chroma
    """

    collection: Collection = Field(
        default=Collection(), description="The collection to query"
    )
    text: str = Field(default="", description="The text to query")
    n_results: int = Field(
        default=5, description="The number of final results to return"
    )
    k_constant: float = Field(
        default=60.0, description="Constant for reciprocal rank fusion (default: 60.0)"
    )
    min_keyword_length: int = Field(
        default=3, description="Minimum length for keyword tokens"
    )

    class OutputType(TypedDict):
        ids: list[str]
        documents: list[str]
        metadatas: list[dict]
        distances: list[float]
        scores: list[float]

    def _normalize_scores(self, scores: list[float]) -> list[float]:
        """Normalize scores to range [0, 1]"""
        if not scores:
            return scores
        min_score = min(scores)
        max_score = max(scores)
        if max_score == min_score:
            return [1.0] * len(scores)
        return [(s - min_score) / (max_score - min_score) for s in scores]

    def _reciprocal_rank_fusion(
        self, semantic_results: dict, keyword_results: dict
    ) -> tuple[list, list, list, list, list]:
        """
        Combine results using reciprocal rank fusion.
        Returns combined and sorted results (ids, documents, metadatas, distances, scores).
        """
        # Create a map to store combined scores
        combined_scores = {}

        # Process semantic search results
        for rank, (id_, doc, meta, dist) in enumerate(
            zip(
                semantic_results["ids"][0],
                semantic_results["documents"][0],
                semantic_results["metadatas"][0],
                semantic_results["distances"][0],
            )
        ):
            score = 1 / (rank + self.k_constant)
            combined_scores[id_] = {
                "doc": doc,
                "meta": meta,
                "distance": dist,
                "score": score,
            }

        # Process keyword search results
        for rank, (id_, doc, meta, dist) in enumerate(
            zip(
                keyword_results["ids"][0],
                keyword_results["documents"][0],
                keyword_results["metadatas"][0],
                keyword_results["distances"][0],
            )
        ):
            score = 1 / (rank + self.k_constant)
            if id_ in combined_scores:
                combined_scores[id_]["score"] += score
            else:
                combined_scores[id_] = {
                    "doc": doc,
                    "meta": meta,
                    "distance": dist,
                    "score": score,
                }

        # Sort by combined score
        sorted_results = sorted(
            combined_scores.items(), key=lambda x: x[1]["score"], reverse=True
        )

        # Unzip results
        ids = []
        docs = []
        metas = []
        distances = []
        scores = []

        for id_, data in sorted_results[: self.n_results]:
            ids.append(id_)
            docs.append(data["doc"])
            metas.append(data["meta"])
            distances.append(data["distance"])
            scores.append(data["score"])

        return ids, docs, metas, distances, scores

    def _get_keyword_query(self, text: str) -> dict:
        """Create keyword query from text"""
        pattern = r"[ ,.!?\-_=|]+"
        query_tokens = [
            token.strip()
            for token in re.split(pattern, text.lower())
            if len(token.strip()) >= self.min_keyword_length
        ]

        if not query_tokens:
            return {}

        if len(query_tokens) > 1:
            return {"$or": [{"$contains": token} for token in query_tokens]}
        return {"$contains": query_tokens[0]}

    async def process(self, context: ProcessingContext) -> OutputType:
        if not self.text.strip():
            raise ValueError("Search text cannot be empty")

        collection = await get_async_collection(self.collection.name)

        # Perform semantic search
        semantic_results = await collection.query(
            query_texts=[self.text],
            n_results=self.n_results * 2,  # Get more results for better fusion
            include=["documents", "metadatas", "distances"],
        )

        # Perform keyword search if we have valid keywords
        keyword_query = self._get_keyword_query(self.text)
        if keyword_query:
            keyword_results = await collection.query(
                query_texts=[self.text],
                n_results=self.n_results * 2,
                where_document=keyword_query,
                include=["documents", "metadatas", "distances"],
            )
        else:
            keyword_results = semantic_results  # Fall back to semantic only

        # Validate results
        for results in [semantic_results, keyword_results]:
            assert results["ids"] is not None, "Ids are not returned"
            assert results["documents"] is not None, "Documents are not returned"
            assert results["metadatas"] is not None, "Metadatas are not returned"
            assert results["distances"] is not None, "Distances are not returned"

        # Combine results using reciprocal rank fusion
        ids, documents, metadatas, distances, scores = self._reciprocal_rank_fusion(
            dict(semantic_results), dict(keyword_results)
        )

        return {
            "ids": ids,
            "documents": documents,
            "metadatas": metadatas,
            "distances": distances,
            "scores": scores,
        }
