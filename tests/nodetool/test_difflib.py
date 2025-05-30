import pytest
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.nodes.lib.difflib import (
    SimilarityRatio,
    GetCloseMatches,
    UnifiedDiff,
)


@pytest.fixture
def context():
    return ProcessingContext(user_id="test", auth_token="test")


@pytest.mark.asyncio
async def test_similarity_ratio(context: ProcessingContext):
    node = SimilarityRatio(a="hello", b="hallo")
    result = await node.process(context)
    assert result > 0.7


@pytest.mark.asyncio
async def test_get_close_matches(context: ProcessingContext):
    node = GetCloseMatches(
        word="appel", possibilities=["ape", "apple", "peach"], n=2, cutoff=0.6
    )
    result = await node.process(context)
    assert "apple" in result


@pytest.mark.asyncio
async def test_unified_diff(context: ProcessingContext):
    node = UnifiedDiff(
        a="a\nb\n", b="a\nc\n", fromfile="old", tofile="new", lineterm=""
    )
    result = await node.process(context)
    assert "--- old" in result and "+++ new" in result
