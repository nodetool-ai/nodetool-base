import pytest
from nodetool.dsl.graph import graph_result
from nodetool.dsl.lib.html import Escape, Unescape
from nodetool.dsl.nodetool.output import StringOutput

escape = Escape(text="Hello <world> & 'AI'")
escape_node = StringOutput(
    name="escape_node", value=escape.output
)

unescape = Unescape(
    text="Hello &lt;world&gt; &amp; &#39;AI&#39;",
)
unescape_node = StringOutput(
    name="unescape_node",
    value=unescape.output,
)


@pytest.mark.asyncio
async def test_escape():
    result = await graph_result(escape_node)
    assert result["escape_node"] == "Hello &lt;world&gt; &amp; &#39;AI&#39;"


@pytest.mark.asyncio
async def test_unescape():
    result = await graph_result(unescape_node)
    assert result["unescape_node"] == "Hello <world> & 'AI'"
