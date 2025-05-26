import pytest

from nodetool.workflows.processing_context import ProcessingContext
from nodetool.nodes.llms.synthesizer import Synthesizer
from nodetool.metadata.types import LanguageModel, Provider
from nodetool.chat.providers import Chunk


@pytest.fixture
def context():
    return ProcessingContext(user_id="test", auth_token="test")


@pytest.mark.asyncio
async def test_synthesizer_process(context, monkeypatch):
    node = Synthesizer(
        prompt="Hello {{ name }}!",
        model=LanguageModel(provider=Provider.OpenAI, id="gpt"),
    )
    node._dynamic_properties = {"name": "Alice"}

    async def fake_generate_messages(**kwargs):
        messages = kwargs.get("messages")
        assert messages[1].content[0].text == "Hello Alice!"
        yield Chunk(content="Hi Alice", content_type="text")

    monkeypatch.setattr(context, "generate_messages", fake_generate_messages)

    result = await node.process(context)
    assert result == "Hi Alice"


