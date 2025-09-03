from pydantic import Field
from nodetool.integrations.vectorstores.chroma.async_chroma_client import (
    get_async_collection,
    get_async_chroma_client,
)
from nodetool.metadata.types import (
    Collection,
    LlamaModel,
)
from nodetool.nodes.chroma.chroma_node import ChromaNode
from nodetool.workflows.processing_context import ProcessingContext


class CollectionNode(ChromaNode):
    """
    Get or create a collection.
    chroma, embedding, collection, RAG, get, create
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
    chroma, embedding, collection, RAG
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
    chroma, embedding, collection, RAG, retrieve
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
    chroma, embedding, collection, RAG, preview
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
