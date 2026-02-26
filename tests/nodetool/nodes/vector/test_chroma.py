import sys
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import os
import numpy as np

# Ensure src is in path so we can import nodetool
sys.path.insert(0, os.path.abspath('src'))

# We rely on tests/conftest.py to set up the environment and mock basic nodetool deps.
# However, if conftest.py mocks specific modules, the imports below might return MagicMocks.
# We need to handle that gracefully.

# Try importing the actual nodes.
# If they are mocked in conftest, we will get MagicMock objects.
from nodetool.nodes.vector.chroma import (
    ChromaNode,
    CollectionNode,
    Count,
    GetDocuments,
    IndexImage,
    IndexEmbedding,
    IndexTextChunk,
    IndexAggregatedText,
    IndexString,
    Peek,
    QueryImage,
    QueryText,
    RemoveOverlap,
    HybridSearch,
    EmbeddingAggregation,
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
from nodetool.workflows.processing_context import ProcessingContext

# Helper to skip tests if the node class itself is a Mock (meaning import failed/was mocked)
def skip_if_mock(obj):
    if isinstance(obj, MagicMock):
        pytest.skip("Node class is mocked (dependencies missing), skipping test")

# --- Fixtures ---

@pytest.fixture
def context():
    """Mock ProcessingContext."""
    # Note: cannot use spec=ProcessingContext if ProcessingContext is a mock
    # because `spec` validation fails on mocks.
    # Just use a plain MagicMock if ProcessingContext is mocked.
    if isinstance(ProcessingContext, MagicMock):
        ctx = MagicMock()
    else:
        ctx = MagicMock(spec=ProcessingContext)

    ctx.find_asset = AsyncMock()
    ctx.get_asset_url = AsyncMock()
    ctx.image_to_pil = AsyncMock()
    return ctx

@pytest.fixture
def mock_chroma_client():
    with patch("nodetool.nodes.vector.chroma.get_async_chroma_client") as mock:
        yield mock

@pytest.fixture
def mock_get_collection():
    with patch("nodetool.nodes.vector.chroma.get_async_collection") as mock:
        yield mock

@pytest.fixture
def mock_env():
    # Use side_effect with a function that returns the desired values
    def get_env(key):
        if key == "OLLAMA_API_URL":
            return "http://localhost:11434"
        if key == "OLLAMA_CONTEXT_LENGTH":
            return "4096"
        return None

    with patch("nodetool.nodes.vector.chroma.Environment") as mock:
        mock.get.side_effect = get_env
        yield mock

@pytest.fixture
def mock_ollama():
    with patch("nodetool.nodes.vector.chroma.get_ollama_client") as mock:
        client = MagicMock()
        client.embeddings = AsyncMock(return_value={"embedding": [1.0, 2.0]})
        mock.return_value = client
        yield mock

# --- Tests ---

class TestChromaNode:
    def test_is_cacheable(self):
        skip_if_mock(ChromaNode)
        assert ChromaNode.is_cacheable() is False

    def test_is_visible(self):
        skip_if_mock(ChromaNode)
        # Base class should not be visible
        assert ChromaNode.is_visible() is False

        # Subclasses should be visible (unless they override it)
        class SubNode(ChromaNode):
            pass
        assert SubNode.is_visible() is True

    @pytest.mark.asyncio
    async def test_load_results(self, context):
        skip_if_mock(ChromaNode)
        node = ChromaNode()
        ids = ["asset1", "asset2", "missing_asset"]

        # Mock find_asset behavior
        async def find_asset_side_effect(asset_id):
            if asset_id == "asset1":
                asset = MagicMock()
                asset.id = "asset1"
                asset.content_type = "image/jpeg"
                return asset
            elif asset_id == "asset2":
                asset = MagicMock()
                asset.id = "asset2"
                asset.content_type = "text/plain"
                return asset
            return None

        context.find_asset.side_effect = find_asset_side_effect
        context.get_asset_url.return_value = "http://example.com/asset"

        results = await node.load_results(context, ids)

        assert len(results) == 2
        # Use simple type check or isinstance if types are real
        assert isinstance(results[0], ImageRef) or (isinstance(ImageRef, MagicMock))
        assert results[0].asset_id == "asset1"
        assert isinstance(results[1], TextRef) or (isinstance(TextRef, MagicMock))
        assert results[1].asset_id == "asset2"


class TestCollectionNode:
    @pytest.mark.asyncio
    async def test_process(self, context, mock_chroma_client):
        skip_if_mock(CollectionNode)
        """Test creating a collection."""
        node = CollectionNode(
            name="test_collection",
            embedding_model=LlamaModel(repo_id="nomic-embed-text")
        )

        client_instance = MagicMock()
        client_instance.get_or_create_collection = AsyncMock()
        mock_chroma_client.return_value = client_instance

        result = await node.process(context)

        assert isinstance(result, Collection) or isinstance(Collection, MagicMock)
        assert result.name == "test_collection"

        mock_chroma_client.assert_called_once()
        client_instance.get_or_create_collection.assert_called_once_with(
            name="test_collection",
            metadata={"embedding_model": "nomic-embed-text"}
        )


class TestCount:
    @pytest.mark.asyncio
    async def test_process(self, context, mock_get_collection):
        skip_if_mock(Count)
        """Test counting documents."""
        collection_mock = MagicMock()
        collection_mock.count = AsyncMock(return_value=10)
        mock_get_collection.return_value = collection_mock

        node = Count(collection=Collection(name="test_collection"))

        result = await node.process(context)

        assert result == 10
        mock_get_collection.assert_called_once_with("test_collection")
        collection_mock.count.assert_called_once()


class TestGetDocuments:
    @pytest.mark.asyncio
    async def test_process(self, context, mock_get_collection):
        skip_if_mock(GetDocuments)
        """Test getting documents."""
        collection_mock = MagicMock()
        collection_mock.get = AsyncMock(return_value={"documents": ["doc1", "doc2"]})
        mock_get_collection.return_value = collection_mock

        node = GetDocuments(
            collection=Collection(name="test_collection"),
            ids=["id1", "id2"],
            limit=2,
            offset=0
        )

        result = await node.process(context)

        assert result == ["doc1", "doc2"]
        collection_mock.get.assert_called_once_with(
            ids=["id1", "id2"],
            limit=2,
            offset=0
        )


class TestIndexImage:
    @pytest.mark.asyncio
    async def test_process_upsert(self, context, mock_get_collection):
        skip_if_mock(IndexImage)
        """Test indexing an image with upsert."""
        collection_mock = MagicMock()
        collection_mock.upsert = AsyncMock()
        mock_get_collection.return_value = collection_mock

        image_ref = ImageRef(document_id="img1")
        # Ensure context.image_to_pil returns something compatible with np.array
        context.image_to_pil.return_value = np.zeros((10, 10, 3), dtype=np.uint8)

        node = IndexImage(
            collection=Collection(name="test"),
            image=image_ref,
            upsert=True,
            metadata={"key": "value"}
        )

        await node.process(context)

        collection_mock.upsert.assert_called_once()
        args, kwargs = collection_mock.upsert.call_args
        assert kwargs['ids'] == ["img1"]
        assert kwargs['metadatas'] == [{"key": "value"}]

    @pytest.mark.asyncio
    async def test_process_no_upsert(self, context, mock_get_collection):
        skip_if_mock(IndexImage)
        """Test indexing an image without upsert."""
        collection_mock = MagicMock()
        collection_mock.add = AsyncMock()
        mock_get_collection.return_value = collection_mock

        image_ref = ImageRef(document_id="img1")
        context.image_to_pil.return_value = np.zeros((10, 10, 3), dtype=np.uint8)

        node = IndexImage(
            collection=Collection(name="test"),
            image=image_ref,
            upsert=False
        )

        await node.process(context)

        collection_mock.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_missing_document_id(self, context):
        skip_if_mock(IndexImage)
        """Test missing document ID raises ValueError."""
        node = IndexImage(image=ImageRef(document_id=None))

        with pytest.raises(ValueError, match="document_id cannot be None"):
            await node.process(context)


class TestIndexEmbedding:
    @pytest.mark.asyncio
    async def test_process(self, context, mock_get_collection):
        skip_if_mock(IndexEmbedding)
        """Test indexing an embedding."""
        collection_mock = MagicMock()
        collection_mock.add = AsyncMock()
        mock_get_collection.return_value = collection_mock

        embedding = NPArray.from_numpy(np.array([1.0, 2.0]))

        node = IndexEmbedding(
            collection=Collection(name="test"),
            embedding=embedding,
            index_id="emb1",
            metadata={"source": "test"}
        )

        await node.process(context)

        collection_mock.add.assert_called_once()
        args, kwargs = collection_mock.add.call_args
        assert kwargs['ids'] == ["emb1"]
        assert np.array_equal(kwargs['embeddings'][0], np.array([1.0, 2.0]))
        assert kwargs['metadatas'] == [{"source": "test"}]

    @pytest.mark.asyncio
    async def test_empty_id(self, context):
        skip_if_mock(IndexEmbedding)
        """Test empty ID raises ValueError."""
        node = IndexEmbedding(
            index_id="",
            embedding=NPArray.from_numpy(np.array([1]))
        )

        with pytest.raises(ValueError, match="The ID cannot be empty"):
            await node.process(context)

    @pytest.mark.asyncio
    async def test_empty_embedding(self, context):
        skip_if_mock(IndexEmbedding)
        """Test empty embedding raises ValueError."""
        node = IndexEmbedding(
            index_id="1",
            embedding=NPArray(value=None)
        )

        with pytest.raises(ValueError, match="The embedding cannot be empty"):
            await node.process(context)


class TestIndexTextChunk:
    @pytest.mark.asyncio
    async def test_process(self, context, mock_get_collection):
        skip_if_mock(IndexTextChunk)
        """Test indexing a text chunk."""
        collection_mock = MagicMock()
        collection_mock.add = AsyncMock()
        mock_get_collection.return_value = collection_mock

        node = IndexTextChunk(
            collection=Collection(name="test"),
            document_id="doc1",
            text="hello world",
            metadata={"source": "test"}
        )

        await node.process(context)

        collection_mock.add.assert_called_once()
        args, kwargs = collection_mock.add.call_args
        assert kwargs['ids'] == ["doc1"]
        assert kwargs['documents'] == ["hello world"]
        assert kwargs['metadatas'] == [{"source": "test"}]

    @pytest.mark.asyncio
    async def test_empty_document_id(self, context):
        skip_if_mock(IndexTextChunk)
        """Test empty document ID raises ValueError."""
        node = IndexTextChunk(
            document_id="",
            text="text"
        )

        with pytest.raises(ValueError, match="The document ID cannot be empty"):
            await node.process(context)


class TestIndexAggregatedText:
    @pytest.mark.asyncio
    async def test_process(self, context, mock_get_collection, mock_env, mock_ollama):
        skip_if_mock(IndexAggregatedText)
        """Test indexing aggregated text chunks."""
        collection_mock = MagicMock()
        collection_mock.metadata = {"embedding_model": "test-model"}
        collection_mock.add = AsyncMock()
        mock_get_collection.return_value = collection_mock

        node = IndexAggregatedText(
            collection=Collection(name="test"),
            document="full document text",
            document_id="doc1",
            text_chunks=["chunk1", "chunk2"],
            aggregation=EmbeddingAggregation.MEAN
        )

        await node.process(context)

        # Check if embeddings were calculated for chunks
        assert mock_ollama.return_value.embeddings.call_count == 2

        # Check aggregation (mean of [1,2] and [1,2] is [1,2])
        collection_mock.add.assert_called_once()
        args, kwargs = collection_mock.add.call_args
        assert np.array_equal(kwargs['embeddings'][0], np.array([1.0, 2.0]))

    @pytest.mark.asyncio
    async def test_missing_model_metadata(self, context, mock_get_collection):
        skip_if_mock(IndexAggregatedText)
        """Test missing embedding model in collection metadata."""
        collection_mock = MagicMock()
        collection_mock.metadata = {}
        mock_get_collection.return_value = collection_mock

        node = IndexAggregatedText(
            collection=Collection(name="test"),
            document="doc",
            document_id="id",
            text_chunks=["chunk"]
        )

        with pytest.raises(ValueError, match="The collection does not have an embedding model"):
            await node.process(context)

    @pytest.mark.asyncio
    async def test_empty_chunks(self, context):
        skip_if_mock(IndexAggregatedText)
        """Test empty text chunks raises ValueError."""
        node = IndexAggregatedText(
            document="doc",
            document_id="id",
            text_chunks=[]
        )

        with pytest.raises(ValueError, match="The text chunks cannot be empty"):
            await node.process(context)


class TestRemoveOverlap:
    def test_overlap_removal_sync(self):
        skip_if_mock(RemoveOverlap)
        """Test internal overlap logic."""
        node = RemoveOverlap(min_overlap_words=2)
        words1 = ["This", "is", "a", "test"]
        words2 = ["a", "test", "sentence"]
        overlap = node._find_word_overlap(words1, words2)
        assert overlap == 2

        words1 = ["No", "overlap"]
        words2 = ["here", "at", "all"]
        overlap = node._find_word_overlap(words1, words2)
        assert overlap == 0

    @pytest.mark.asyncio
    async def test_process(self, context):
        skip_if_mock(RemoveOverlap)
        """Test removing overlaps."""
        node = RemoveOverlap(min_overlap_words=2)

        # Test case 1: Standard overlap
        node.documents = [
            "This is a test sentence.",
            "test sentence. And this is another."
        ]
        result = await node.process(context)
        assert result['documents'][1] == "And this is another."

        # Test case 2: No overlap
        node.documents = [
            "First sentence.",
            "Second sentence."
        ]
        result = await node.process(context)
        assert result['documents'][1] == "Second sentence."

        # Test case 3: Partial overlap but below minimum
        node.min_overlap_words = 5
        node.documents = [
            "Just a small overlap here.",
            "overlap here. Continuation."
        ]
        result = await node.process(context)
        # Should NOT remove overlap because only 2 words overlap, min is 5
        assert result['documents'][1] == "overlap here. Continuation."

    @pytest.mark.asyncio
    async def test_empty_input(self, context):
        skip_if_mock(RemoveOverlap)
        """Test empty input handling."""
        node = RemoveOverlap(documents=[])
        result = await node.process(context)
        assert result == {"documents": []}


class TestHybridSearch:
    def test_reciprocal_rank_fusion(self):
        skip_if_mock(HybridSearch)
        """Test the RRF implementation."""
        node = HybridSearch(k_constant=1.0, n_results=5)

        # Test case 1: Identical rankings
        semantic = {
            "ids": [["1", "2"]],
            "documents": [["doc1", "doc2"]],
            "metadatas": [[{}, {}]],
            "distances": [[0.1, 0.2]]
        }
        keyword = semantic.copy()

        result = node._reciprocal_rank_fusion(semantic, keyword)
        # Expected scores: (1/(0+1) + 1/(0+1)) = 2.0 for "1"
        #                (1/(1+1) + 1/(1+1)) = 1.0 for "2"
        # result is (ids, docs, metas, dists, scores)
        # scores is index 4
        assert result[4][0] == 2.0  # scores
        assert result[0][0] == "1"  # ids

        # Test case 2: Different rankings
        semantic = {
            "ids": [["1", "2"]],
            "documents": [["doc1", "doc2"]],
            "metadatas": [[{}, {}]],
            "distances": [[0.1, 0.2]]
        }
        keyword = {
            "ids": [["2", "1"]],
            "documents": [["doc2", "doc1"]],
            "metadatas": [[{}, {}]],
            "distances": [[0.05, 0.15]]
        }

        result = node._reciprocal_rank_fusion(semantic, keyword)
        # "1": sem_rank=0 -> score=1.0 | key_rank=1 -> score=0.5 => total=1.5
        # "2": sem_rank=1 -> score=0.5 | key_rank=0 -> score=1.0 => total=1.5
        assert result[4][0] == 1.5
        assert result[4][1] == 1.5

    @pytest.mark.asyncio
    async def test_process_basic(self, context, mock_get_collection):
        skip_if_mock(HybridSearch)
        """Test basic hybrid search execution."""
        collection_mock = MagicMock()
        mock_get_collection.return_value = collection_mock

        # Mock query results
        dummy_results = {
            "ids": [["1"]],
            "documents": [["doc1"]],
            "metadatas": [[{}]],
            "distances": [[0.1]]
        }
        collection_mock.query = AsyncMock(return_value=dummy_results)

        node = HybridSearch(
            collection=Collection(name="test"),
            text="hello world",
            n_results=1,
            min_keyword_length=3
        )

        result = await node.process(context)

        # Verify both semantic and keyword queries were made
        assert collection_mock.query.call_count == 2

        # Check result structure
        assert "scores" in result
        assert len(result["ids"]) == 1
        assert result["ids"][0] == "1"

    @pytest.mark.asyncio
    async def test_empty_query(self, context):
        skip_if_mock(HybridSearch)
        """Test empty query raises ValueError."""
        node = HybridSearch(text="")

        with pytest.raises(ValueError, match="Search text cannot be empty"):
            await node.process(context)
