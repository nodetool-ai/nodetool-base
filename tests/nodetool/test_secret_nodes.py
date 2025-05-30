import pytest
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.nodes.nodetool.secret import SetSecret, GetSecret


@pytest.fixture
def context(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    return ProcessingContext(user_id="test", auth_token="test")


@pytest.mark.asyncio
async def test_secret_nodes(context: ProcessingContext):
    set_node = SetSecret(name="TEST_SECRET", value="value")
    await set_node.process(context)
    get_node = GetSecret(name="TEST_SECRET")
    result = await get_node.process(context)
    assert result == "value"
