import numpy as np
import pytest
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.nodes.nodetool.text import Embedding
from nodetool.metadata.types import EmbeddingModel, NPArray, Provider


@pytest.fixture
def context():
    return ProcessingContext(user_id="test", auth_token="test")


@pytest.mark.asyncio
async def test_embedding_empty_input(context):
    """Test that empty input raises ValueError."""
    node = Embedding(input="")
    with pytest.raises(ValueError, match="Input text cannot be empty"):
        await node.process(context)


@pytest.mark.asyncio
async def test_embedding_empty_provider(context):
    """Test that empty provider raises ValueError."""
    node = Embedding(
        input="hello world",
        model=EmbeddingModel(provider=Provider.Empty, id="", name=""),
    )
    with pytest.raises(ValueError, match="Please select an embedding model"):
        await node.process(context)


@pytest.mark.asyncio
async def test_embedding_openai_process(context, monkeypatch):
    """Test embedding with OpenAI provider."""
    node = Embedding(
        input="hello world",
        chunk_size=5,
        model=EmbeddingModel(
            provider=Provider.OpenAI,
            id="text-embedding-3-small",
            name="Text Embedding 3 Small",
        ),
    )

    async def fake_run_prediction(
        node_id, provider, model, run_prediction_function, params=None, data=None
    ):
        assert params == {"input": ["hello", " worl", "d"]}
        assert model == "text-embedding-3-small"
        return {
            "data": [
                {"embedding": [0.1, 0.2], "index": 0, "object": "embedding"},
                {"embedding": [0.3, 0.4], "index": 1, "object": "embedding"},
                {"embedding": [0.5, 0.6], "index": 2, "object": "embedding"},
            ],
            "model": "text-embedding-3-small",
            "object": "list",
            "usage": {"prompt_tokens": 1, "total_tokens": 1},
        }

    monkeypatch.setattr(context, "run_prediction", fake_run_prediction)
    result = await node.process(context)
    assert isinstance(result, NPArray)
    # Average of [0.1, 0.2], [0.3, 0.4], [0.5, 0.6] = [0.3, 0.4]
    np.testing.assert_allclose(result.to_numpy(), np.array([0.3, 0.4]))


@pytest.mark.asyncio
async def test_embedding_unsupported_provider(context):
    """Test that unsupported provider raises ValueError."""
    node = Embedding(
        input="hello world",
        model=EmbeddingModel(
            provider=Provider.FalAI,  # FalAI is not supported for embeddings
            id="some-model",
            name="Some Model",
        ),
    )
    with pytest.raises(ValueError, match="Unsupported embedding provider"):
        await node.process(context)


@pytest.mark.asyncio
async def test_embedding_default_model():
    """Test that default model is OpenAI text-embedding-3-small."""
    node = Embedding(input="test")
    assert node.model.provider == Provider.OpenAI
    assert node.model.id == "text-embedding-3-small"


@pytest.mark.asyncio
async def test_embedding_chunk_size():
    """Test that chunk_size is respected."""
    node = Embedding(input="hello world", chunk_size=2)
    assert node.chunk_size == 2


@pytest.mark.asyncio
async def test_embedding_get_basic_fields():
    """Test get_basic_fields returns expected fields."""
    fields = Embedding.get_basic_fields()
    assert "model" in fields
    assert "input" in fields
