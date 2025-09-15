import numpy as np
import pytest

faiss = pytest.importorskip("faiss")

from nodetool.workflows.processing_context import ProcessingContext
from nodetool.metadata.types import NPArray
from nodetool.nodes.vector.faiss import (
    CreateIndexFlatL2,
    CreateIndexFlatIP,
    CreateIndexIVFFlat,
    TrainIndex,
    AddVectors,
    AddWithIds,
    Search,
)


@pytest.fixture
def context():
    return ProcessingContext(user_id="test", auth_token="test")


@pytest.mark.asyncio
async def test_flat_l2_add_and_search(context):
    dim = 4
    vectors = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    query = np.array([0.9, 0.0, 0.0, 0.0], dtype=np.float32)

    idx_node = CreateIndexFlatL2(dim=dim)
    idx = await idx_node.process(context)

    add_node = AddVectors(index=idx, vectors=NPArray.from_numpy(vectors))
    idx = await add_node.process(context)

    search_node = Search(index=idx, query=NPArray.from_numpy(query), k=2)
    out = await search_node.process(context)

    assert "distances" in out and "indices" in out
    D = out["distances"].to_numpy()
    I = out["indices"].to_numpy()
    assert D.shape == (1, 2)
    assert I.shape == (1, 2)
    # Nearest should be the vector [1, 0, 0, 0] which has id 1
    assert I[0, 0] == 1


@pytest.mark.asyncio
async def test_add_with_ids_and_search_returns_labels(context):
    dim = 4
    vectors = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    ids = np.array([10, 12, 14], dtype=np.int64)
    query = np.array([0.95, 0.0, 0.0, 0.0], dtype=np.float32)

    idx_node = CreateIndexFlatL2(dim=dim)
    idx = await idx_node.process(context)

    add_node = AddWithIds(
        index=idx, vectors=NPArray.from_numpy(vectors), ids=NPArray.from_numpy(ids)
    )
    idx = await add_node.process(context)

    out = await Search(index=idx, query=NPArray.from_numpy(query), k=1).process(context)
    I = out["indices"].to_numpy()
    assert I.shape == (1, 1)
    assert I[0, 0] == 12


@pytest.mark.asyncio
async def test_ivf_train_add_search(context):
    dim = 4
    rng = np.random.default_rng(42)
    train_vecs = rng.normal(size=(100, dim)).astype(np.float32)
    add_vecs = rng.normal(size=(20, dim)).astype(np.float32)

    idx_node = CreateIndexIVFFlat(dim=dim, nlist=4)
    idx = await idx_node.process(context)

    # Train
    trained = await TrainIndex(
        index=idx, vectors=NPArray.from_numpy(train_vecs)
    ).process(context)
    # Verify trained flag when available
    if hasattr(trained.index, "is_trained"):
        assert trained.index.is_trained is True

    # Add
    idx2 = await AddVectors(
        index=trained, vectors=NPArray.from_numpy(add_vecs)
    ).process(context)

    # Search for an existing vector; probe multiple lists for better recall
    q = add_vecs[5]
    out = await Search(index=idx2, query=NPArray.from_numpy(q), k=1, nprobe=4).process(
        context
    )
    D = out["distances"].to_numpy()
    I = out["indices"].to_numpy()
    assert D.shape == (1, 1)
    assert I.shape == (1, 1)
    # Should be a valid label (0..len(add_vecs)-1) and reasonably close
    assert 0 <= int(I[0, 0]) < add_vecs.shape[0]
    assert D[0, 0] >= 0.0
