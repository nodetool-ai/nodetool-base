import pytest
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.nodes.nodetool.textwrap import Fill, Wrap, Shorten, Indent, Dedent


@pytest.fixture
def context():
    return ProcessingContext(user_id="test", auth_token="test")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "node, expected",
    [
        (Fill(text="hello world", width=5), "hello\nworld"),
        (Wrap(text="hello world", width=5), ["hello", "world"]),
        (
            Shorten(text="hello wonderful world", width=10, placeholder="..."),
            "hello...",
        ),
        (Indent(text="a\nb", prefix="*>"), "*>a\n*>b"),
        (Dedent(text="    a\n    b"), "a\nb"),
    ],
)
async def test_textwrap_nodes(context: ProcessingContext, node, expected):
    result = await node.process(context)
    assert result == expected
