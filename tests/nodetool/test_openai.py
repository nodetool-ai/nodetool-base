import numpy as np
import pytest
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.nodes.openai.text import WebSearch, Embedding, Moderation, ChatComplete, GPTModel
from nodetool.nodes.openai.image import EditImage
from nodetool.metadata.types import NPArray, ImageRef


@pytest.fixture
def context():
    return ProcessingContext(user_id="test", auth_token="test")


# ChatComplete Tests


@pytest.mark.asyncio
async def test_chat_complete_success(context, monkeypatch):
    """Test successful GPT chat completion."""
    node = ChatComplete(prompt="Hello, world!", model=GPTModel.GPT_4O_MINI)

    async def fake_run_prediction(node_id, provider, model, run_prediction_function, params=None, data=None):
        return {"choices": [{"message": {"content": "Hello! How can I help you?"}}]}

    monkeypatch.setattr(context, "run_prediction", fake_run_prediction)
    result = await node.process(context)
    assert result == "Hello! How can I help you?"


@pytest.mark.asyncio
async def test_chat_complete_empty_prompt(context):
    """Test that empty prompt raises ValueError."""
    node = ChatComplete(prompt="")
    with pytest.raises(ValueError, match="Prompt cannot be empty"):
        await node.process(context)


@pytest.mark.asyncio
async def test_chat_complete_with_system_prompt(context, monkeypatch):
    """Test GPT chat completion with system prompt."""
    node = ChatComplete(
        prompt="Write a poem",
        system_prompt="You are a creative poet.",
        model=GPTModel.GPT_4O,
    )

    async def fake_run_prediction(node_id, provider, model, run_prediction_function, params=None, data=None):
        # Verify system prompt is included in messages
        assert len(params["messages"]) == 2
        assert params["messages"][0]["role"] == "system"
        assert params["messages"][0]["content"] == "You are a creative poet."
        return {"choices": [{"message": {"content": "A beautiful poem"}}]}

    monkeypatch.setattr(context, "run_prediction", fake_run_prediction)
    result = await node.process(context)
    assert result == "A beautiful poem"


def test_chat_complete_basic_fields():
    """Test ChatComplete basic fields."""
    assert ChatComplete.get_basic_fields() == ["prompt", "model"]


def test_chat_complete_default_values():
    """Test ChatComplete default field values."""
    node = ChatComplete()
    assert node.model == GPTModel.GPT_4O_MINI
    assert node.prompt == ""
    assert node.system_prompt == ""
    assert node.temperature == 1.0
    assert node.max_tokens == 1024


def test_gpt_model_enum_values():
    """Test that all GPT model enum values are valid strings."""
    for model in GPTModel:
        assert isinstance(model.value, str)


# WebSearch Tests


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


@pytest.mark.asyncio
async def test_moderation_success(context, monkeypatch):
    node = Moderation(input="Hello, this is a friendly message.")

    async def fake_run_prediction(node_id, provider, model, run_prediction_function, params=None, data=None):
        assert params == {"input": "Hello, this is a friendly message."}
        return {
            "id": "modr-123",
            "model": "omni-moderation-latest",
            "results": [
                {
                    "flagged": False,
                    "categories": {
                        "harassment": False,
                        "hate": False,
                        "self_harm": False,
                        "sexual": False,
                        "violence": False,
                    },
                    "category_scores": {
                        "harassment": 0.001,
                        "hate": 0.002,
                        "self_harm": 0.0001,
                        "sexual": 0.0005,
                        "violence": 0.0003,
                    },
                }
            ],
        }

    monkeypatch.setattr(context, "run_prediction", fake_run_prediction)
    result = await node.process(context)
    assert result["flagged"] is False
    assert "harassment" in result["categories"]
    assert "harassment" in result["category_scores"]


@pytest.mark.asyncio
async def test_moderation_flagged_content(context, monkeypatch):
    node = Moderation(input="Some harmful content")

    async def fake_run_prediction(node_id, provider, model, run_prediction_function, params=None, data=None):
        return {
            "id": "modr-456",
            "model": "omni-moderation-latest",
            "results": [
                {
                    "flagged": True,
                    "categories": {
                        "harassment": True,
                        "hate": False,
                        "violence": True,
                    },
                    "category_scores": {
                        "harassment": 0.95,
                        "hate": 0.1,
                        "violence": 0.85,
                    },
                }
            ],
        }

    monkeypatch.setattr(context, "run_prediction", fake_run_prediction)
    result = await node.process(context)
    assert result["flagged"] is True
    assert result["categories"]["harassment"] is True
    assert result["category_scores"]["harassment"] == 0.95


@pytest.mark.asyncio
async def test_moderation_empty_input(context):
    node = Moderation(input="")
    with pytest.raises(ValueError):
        await node.process(context)


@pytest.mark.asyncio
async def test_edit_image_empty_prompt(context):
    # Create a mock ImageRef that is_set() returns True
    image = ImageRef(uri="file:///test.png")
    node = EditImage(image=image, prompt="")
    with pytest.raises(ValueError, match="Edit prompt cannot be empty"):
        await node.process(context)


@pytest.mark.asyncio
async def test_edit_image_missing_image(context):
    node = EditImage(prompt="Add a hat to the person")
    with pytest.raises(ValueError, match="Input image is required"):
        await node.process(context)

