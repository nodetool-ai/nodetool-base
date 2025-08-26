import pytest
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.nodes.lib.markdown import (
    ExtractLinks,
    ExtractHeaders,
    ExtractBulletLists,
    ExtractNumberedLists,
    ExtractCodeBlocks,
    ExtractTables,
)


@pytest.fixture
def context():
    return ProcessingContext(user_id="test", auth_token="test")


MD = """
# Title

Intro with a [link](https://a.example) and a bare <https://b.example>.

## Section

- item A
- item B

1. first
2. second

```py
print("hi")
```

```
code block
```

| col1 | col2 |
| ---- | ---- |
| a    | b    |
| c    | d    |
"""


@pytest.mark.asyncio
async def test_md_extract_links(context: ProcessingContext):
    node = ExtractLinks(markdown=MD)
    links = await node.process(context)
    urls = {l["url"] for l in links}
    assert "https://a.example" in urls and "https://b.example" in urls
    assert any(l.get("title") for l in links)


@pytest.mark.asyncio
async def test_md_extract_headers(context: ProcessingContext):
    node = ExtractHeaders(markdown=MD, max_level=2)
    headers = await node.process(context)
    levels = [h["level"] for h in headers]
    texts = [h["text"] for h in headers]
    assert levels == [1, 2]
    assert texts == ["Title", "Section"]


@pytest.mark.asyncio
async def test_md_extract_lists(context: ProcessingContext):
    bullets = await ExtractBulletLists(markdown=MD).process(context)
    numbers = await ExtractNumberedLists(markdown=MD).process(context)
    assert [i["text"] for i in bullets[0]] == ["item A", "item B"]
    assert numbers[0] == ["first", "second"]


@pytest.mark.asyncio
async def test_md_extract_code_and_tables(context: ProcessingContext):
    blocks = await ExtractCodeBlocks(markdown=MD).process(context)
    assert {b["language"] for b in blocks} == {"py", "text"}
    df = await ExtractTables(markdown=MD).process(context)
    col_names = [c.name for c in df.columns]
    assert col_names == ["col1", "col2"]
    assert df.data == [["a", "b"], ["c", "d"]]

