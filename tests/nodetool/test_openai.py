import numpy as np
import pytest
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.nodes.openai.text import WebSearch, Embedding
from nodetool.metadata.types import NPArray

@pytest.fixture
def context():
    return ProcessingContext(user_id="test", auth_token="test")

@pytest.mark.asyncio
async def test_websearch_success(context, monkeypatch):
    node = WebSearch(query="cats")

    async def fake_run_prediction(node_id, provider, model, run_prediction_function, params=None, data=None):
        return {"choices": [{"message": {"content": "Cats result"}}]}

    monkeypatch.setattr(context, "run_prediction", fake_run_prediction)
    result = await node.process(context)
    assert result == "Cats result"

@pytest.mark.asyncio
async def test_websearch_empty_query(context):
    node = WebSearch(query="")
    with pytest.raises(ValueError):
        await node.process(context)

@pytest.mark.asyncio
async def test_embedding_process(context, monkeypatch):
    node = Embedding(input="hello world", chunk_size=5)

    async def fake_run_prediction(node_id, provider, model, run_prediction_function, params=None, data=None):
        assert params == {"input": ["hello", " worl", "d"]}
        return {
            "data": [
                {"embedding": [0.1, 0.2], "index": 0, "object": "embedding"},
                {"embedding": [0.3, 0.4], "index": 1, "object": "embedding"},
            ],
            "model": node.model.value,
            "object": "list",
            "usage": {"prompt_tokens": 1, "total_tokens": 1},
        }

    monkeypatch.setattr(context, "run_prediction", fake_run_prediction)
    result = await node.process(context)
    assert isinstance(result, NPArray)
    np.testing.assert_allclose(result.to_numpy(), np.array([0.2, 0.3]))

