import pytest
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.nodes.nodetool.text import FormatText


@pytest.fixture
def context():
    return ProcessingContext(user_id="test", auth_token="test")


@pytest.mark.asyncio
async def test_format_text_renders_with_dynamic_props(context: ProcessingContext):
    node = FormatText(template="Hello, {{ Name|upper }}! {{ city|default('X') }}")
    # Simulate dynamic props injection
    node._dynamic_properties = {"Name": "Alice"}
    out = await node.process(context)
    assert out == "Hello, ALICE! X"

