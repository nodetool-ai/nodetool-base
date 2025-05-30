import pytest
from nodetool.dsl.graph import graph_result
from nodetool.dsl.lib.html import Escape, Unescape
from nodetool.dsl.nodetool.output import StringOutput

escape_node = StringOutput(
    name="escape_node", value=Escape(text="Hello <world> & 'AI'")
)

unescape_node = StringOutput(
    name="unescape_node",
    value=Unescape(
        text="Hello &lt;world&gt; &amp; &#39;AI&#39;",
    ),
)


@pytest.mark.asyncio
async def test_escape():
    result = await graph_result(escape_node)
    assert isinstance(result, list)
    assert result[0] == "Hello &lt;world&gt; &amp; &#39;AI&#39;"


@pytest.mark.asyncio
async def test_unescape():
    result = await graph_result(unescape_node)
    assert isinstance(result, list)
    assert result[0] == "Hello <world> & 'AI'"
