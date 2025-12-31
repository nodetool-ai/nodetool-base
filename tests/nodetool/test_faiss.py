import pytest
import numpy as np
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.metadata.types import FaissIndex, NPArray
from nodetool.nodes.vector.faiss import (
    FaissNode,
    CreateIndexFlatL2,
    CreateIndexFlatIP,
    CreateIndexIVFFlat,
    TrainIndex,
    AddVectors,
    AddWithIds,
    Search,
    Metric,
    _ensure_2d_float32,
    _ensure_1d_int64,
)


@pytest.fixture
def context():
    return ProcessingContext(user_id="test", auth_token="test")


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_ensure_2d_float32_1d_input(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = _ensure_2d_float32(arr)
        assert result.ndim == 2
        assert result.shape == (1, 3)
        assert result.dtype == np.float32

    def test_ensure_2d_float32_2d_input(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = _ensure_2d_float32(arr)
        assert result.ndim == 2
        assert result.shape == (2, 2)
        assert result.dtype == np.float32

    def test_ensure_1d_int64(self):
        arr = np.array([1, 2, 3])
        result = _ensure_1d_int64(arr)
        assert result.ndim == 1
        assert result.dtype == np.int64

    def test_ensure_1d_int64_2d_input(self):
        arr = np.array([[1], [2], [3]])
        result = _ensure_1d_int64(arr)
        assert result.ndim == 1
        assert len(result) == 3


class TestFaissNodeBase:
    """Tests for FaissNode base class."""

    def test_is_cacheable(self):
        assert FaissNode.is_cacheable() is False

    def test_is_visible_base_class(self):
        assert FaissNode.is_visible() is False


class TestCreateIndexFlatL2:
    """Tests for CreateIndexFlatL2 node."""

    @pytest.mark.asyncio
    async def test_create_index_flat_l2(self, context):
        node = CreateIndexFlatL2(dim=128)
        result = await node.process(context)

        assert isinstance(result, FaissIndex)
        assert result.index is not None
        assert result.index.d == 128

    @pytest.mark.asyncio
    async def test_create_index_flat_l2_default_dim(self, context):
        node = CreateIndexFlatL2()
        result = await node.process(context)

        assert result.index.d == 768  # default dimension


class TestCreateIndexFlatIP:
    """Tests for CreateIndexFlatIP node."""

    @pytest.mark.asyncio
    async def test_create_index_flat_ip(self, context):
        node = CreateIndexFlatIP(dim=64)
        result = await node.process(context)

        assert isinstance(result, FaissIndex)
        assert result.index is not None
        assert result.index.d == 64


class TestCreateIndexIVFFlat:
    """Tests for CreateIndexIVFFlat node."""

    @pytest.mark.asyncio
    async def test_create_index_ivf_flat_l2(self, context):
        node = CreateIndexIVFFlat(dim=64, nlist=16, metric=Metric.L2)
        result = await node.process(context)

        assert isinstance(result, FaissIndex)
        assert result.index is not None
        assert result.index.d == 64

    @pytest.mark.asyncio
    async def test_create_index_ivf_flat_ip(self, context):
        node = CreateIndexIVFFlat(dim=64, nlist=16, metric=Metric.IP)
        result = await node.process(context)

        assert isinstance(result, FaissIndex)
        assert result.index is not None


class TestTrainIndex:
    """Tests for TrainIndex node."""

    @pytest.mark.asyncio
    async def test_train_ivf_index(self, context):
        # Create an IVF index which requires training
        create_node = CreateIndexIVFFlat(dim=4, nlist=2, metric=Metric.L2)
        faiss_index = await create_node.process(context)

        # Create training vectors
        training_data = np.random.rand(100, 4).astype(np.float32)
        vectors = NPArray.from_numpy(training_data)

        train_node = TrainIndex(index=faiss_index, vectors=vectors)
        result = await train_node.process(context)

        assert result.index.is_trained

    @pytest.mark.asyncio
    async def test_train_index_empty_raises(self, context):
        create_node = CreateIndexFlatL2(dim=4)
        faiss_index = await create_node.process(context)

        train_node = TrainIndex(index=faiss_index, vectors=NPArray())
        with pytest.raises(ValueError, match="Training vectors are empty"):
            await train_node.process(context)

    @pytest.mark.asyncio
    async def test_train_index_no_index_raises(self, context):
        train_node = TrainIndex(index=FaissIndex(), vectors=NPArray.from_numpy(np.ones((10, 4))))
        with pytest.raises(ValueError, match="FAISS index is not set"):
            await train_node.process(context)

    @pytest.mark.asyncio
    async def test_train_dimension_mismatch_raises(self, context):
        create_node = CreateIndexIVFFlat(dim=4, nlist=2, metric=Metric.L2)
        faiss_index = await create_node.process(context)

        # Wrong dimension vectors
        training_data = np.random.rand(100, 8).astype(np.float32)  # dim=8 instead of 4
        vectors = NPArray.from_numpy(training_data)

        train_node = TrainIndex(index=faiss_index, vectors=vectors)
        with pytest.raises(ValueError, match="Vector dimension .* does not match"):
            await train_node.process(context)


class TestAddVectors:
    """Tests for AddVectors node."""

    @pytest.mark.asyncio
    async def test_add_vectors_to_flat_index(self, context):
        create_node = CreateIndexFlatL2(dim=4)
        faiss_index = await create_node.process(context)

        vectors = NPArray.from_numpy(np.random.rand(10, 4).astype(np.float32))
        add_node = AddVectors(index=faiss_index, vectors=vectors)
        result = await add_node.process(context)

        assert result.index.ntotal == 10

    @pytest.mark.asyncio
    async def test_add_vectors_empty_raises(self, context):
        create_node = CreateIndexFlatL2(dim=4)
        faiss_index = await create_node.process(context)

        add_node = AddVectors(index=faiss_index, vectors=NPArray())
        with pytest.raises(ValueError, match="Vectors are empty"):
            await add_node.process(context)

    @pytest.mark.asyncio
    async def test_add_vectors_no_index_raises(self, context):
        add_node = AddVectors(index=FaissIndex(), vectors=NPArray.from_numpy(np.ones((10, 4))))
        with pytest.raises(ValueError, match="FAISS index is not set"):
            await add_node.process(context)

    @pytest.mark.asyncio
    async def test_add_vectors_dimension_mismatch_raises(self, context):
        create_node = CreateIndexFlatL2(dim=4)
        faiss_index = await create_node.process(context)

        # Wrong dimension vectors
        vectors = NPArray.from_numpy(np.random.rand(10, 8).astype(np.float32))  # dim=8

        add_node = AddVectors(index=faiss_index, vectors=vectors)
        with pytest.raises(ValueError, match="Vector dimension .* does not match"):
            await add_node.process(context)


class TestAddWithIds:
    """Tests for AddWithIds node."""

    @pytest.mark.asyncio
    async def test_add_with_ids(self, context):
        create_node = CreateIndexFlatL2(dim=4)
        faiss_index = await create_node.process(context)

        vectors = NPArray.from_numpy(np.random.rand(5, 4).astype(np.float32))
        ids = NPArray.from_numpy(np.array([100, 200, 300, 400, 500], dtype=np.int64))

        add_node = AddWithIds(index=faiss_index, vectors=vectors, ids=ids)
        result = await add_node.process(context)

        assert result.index.ntotal == 5

    @pytest.mark.asyncio
    async def test_add_with_ids_empty_vectors_raises(self, context):
        create_node = CreateIndexFlatL2(dim=4)
        faiss_index = await create_node.process(context)

        add_node = AddWithIds(
            index=faiss_index,
            vectors=NPArray(),
            ids=NPArray.from_numpy(np.array([1, 2, 3]))
        )
        with pytest.raises(ValueError, match="Vectors are empty"):
            await add_node.process(context)

    @pytest.mark.asyncio
    async def test_add_with_ids_empty_ids_raises(self, context):
        create_node = CreateIndexFlatL2(dim=4)
        faiss_index = await create_node.process(context)

        add_node = AddWithIds(
            index=faiss_index,
            vectors=NPArray.from_numpy(np.ones((3, 4))),
            ids=NPArray()
        )
        with pytest.raises(ValueError, match="IDs are empty"):
            await add_node.process(context)

    @pytest.mark.asyncio
    async def test_add_with_ids_length_mismatch_raises(self, context):
        create_node = CreateIndexFlatL2(dim=4)
        faiss_index = await create_node.process(context)

        vectors = NPArray.from_numpy(np.random.rand(5, 4).astype(np.float32))
        ids = NPArray.from_numpy(np.array([100, 200, 300], dtype=np.int64))  # Only 3 IDs

        add_node = AddWithIds(index=faiss_index, vectors=vectors, ids=ids)
        with pytest.raises(ValueError, match="Vectors and IDs must have the same length"):
            await add_node.process(context)


class TestSearch:
    """Tests for Search node."""

    @pytest.mark.asyncio
    async def test_search_flat_index(self, context):
        # Create and populate an index
        create_node = CreateIndexFlatL2(dim=4)
        faiss_index = await create_node.process(context)

        vectors = NPArray.from_numpy(np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float32))

        add_node = AddVectors(index=faiss_index, vectors=vectors)
        await add_node.process(context)

        # Search
        query = NPArray.from_numpy(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32))
        search_node = Search(index=faiss_index, query=query, k=2)
        result = await search_node.process(context)

        assert "distances" in result
        assert "indices" in result

        indices = result["indices"].to_numpy()

        assert indices.shape == (1, 2)
        assert indices[0, 0] == 0  # First result should be exact match

    @pytest.mark.asyncio
    async def test_search_1d_query(self, context):
        # Create and populate an index
        create_node = CreateIndexFlatL2(dim=4)
        faiss_index = await create_node.process(context)

        vectors = NPArray.from_numpy(np.random.rand(10, 4).astype(np.float32))
        add_node = AddVectors(index=faiss_index, vectors=vectors)
        await add_node.process(context)

        # Search with 1D query (should be reshaped to 2D)
        query = NPArray.from_numpy(np.random.rand(4).astype(np.float32))
        search_node = Search(index=faiss_index, query=query, k=3)
        result = await search_node.process(context)

        indices = result["indices"].to_numpy()
        assert indices.shape[1] == 3

    @pytest.mark.asyncio
    async def test_search_empty_query_raises(self, context):
        create_node = CreateIndexFlatL2(dim=4)
        faiss_index = await create_node.process(context)

        search_node = Search(index=faiss_index, query=NPArray(), k=3)
        with pytest.raises(ValueError, match="Query vectors are empty"):
            await search_node.process(context)

    @pytest.mark.asyncio
    async def test_search_no_index_raises(self, context):
        query = NPArray.from_numpy(np.ones((1, 4)))
        search_node = Search(index=FaissIndex(), query=query, k=3)
        with pytest.raises(ValueError, match="FAISS index is not set"):
            await search_node.process(context)

    @pytest.mark.asyncio
    async def test_search_dimension_mismatch_raises(self, context):
        create_node = CreateIndexFlatL2(dim=4)
        faiss_index = await create_node.process(context)

        # Add some vectors
        vectors = NPArray.from_numpy(np.random.rand(10, 4).astype(np.float32))
        add_node = AddVectors(index=faiss_index, vectors=vectors)
        await add_node.process(context)

        # Query with wrong dimension
        query = NPArray.from_numpy(np.random.rand(1, 8).astype(np.float32))  # dim=8
        search_node = Search(index=faiss_index, query=query, k=3)
        with pytest.raises(ValueError, match="Query dimension .* does not match"):
            await search_node.process(context)


class TestMetricEnum:
    """Tests for Metric enum."""

    def test_metric_values(self):
        assert Metric.L2 == "L2"
        assert Metric.IP == "IP"


class TestEndToEndWorkflow:
    """End-to-end workflow tests."""

    @pytest.mark.asyncio
    async def test_full_workflow_flat_l2(self, context):
        """Test complete workflow: create -> add -> search."""
        # Create index
        create_node = CreateIndexFlatL2(dim=8)
        faiss_index = await create_node.process(context)

        # Add vectors
        np.random.seed(42)
        vectors = NPArray.from_numpy(np.random.rand(100, 8).astype(np.float32))
        add_node = AddVectors(index=faiss_index, vectors=vectors)
        await add_node.process(context)

        # Search
        query = NPArray.from_numpy(np.random.rand(5, 8).astype(np.float32))
        search_node = Search(index=faiss_index, query=query, k=10)
        result = await search_node.process(context)

        distances = result["distances"].to_numpy()
        indices = result["indices"].to_numpy()

        assert distances.shape == (5, 10)
        assert indices.shape == (5, 10)

    @pytest.mark.asyncio
    async def test_full_workflow_ivf(self, context):
        """Test complete workflow with IVF index: create -> train -> add -> search."""
        # Create IVF index
        create_node = CreateIndexIVFFlat(dim=8, nlist=4, metric=Metric.L2)
        faiss_index = await create_node.process(context)

        # Train with sufficient data
        np.random.seed(42)
        training_data = NPArray.from_numpy(np.random.rand(200, 8).astype(np.float32))
        train_node = TrainIndex(index=faiss_index, vectors=training_data)
        await train_node.process(context)

        # Add vectors
        vectors = NPArray.from_numpy(np.random.rand(50, 8).astype(np.float32))
        add_node = AddVectors(index=faiss_index, vectors=vectors)
        await add_node.process(context)

        # Search
        query = NPArray.from_numpy(np.random.rand(3, 8).astype(np.float32))
        search_node = Search(index=faiss_index, query=query, k=5, nprobe=2)
        result = await search_node.process(context)

        assert result["indices"].to_numpy().shape == (3, 5)
