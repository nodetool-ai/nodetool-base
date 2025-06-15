import pytest
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.nodes.lib.random import (
    RandomInt,
    RandomFloat,
    RandomChoice,
    RandomBool,
)


@pytest.fixture
def context():
    return ProcessingContext(user_id="test", auth_token="test")


@pytest.mark.asyncio
async def test_random_int_range(context: ProcessingContext):
    node = RandomInt(minimum=1, maximum=3)
    result = await node.process(context)
    assert 1 <= result <= 3


@pytest.mark.asyncio
async def test_random_float_range(context: ProcessingContext):
    node = RandomFloat(minimum=0.5, maximum=1.5)
    result = await node.process(context)
    assert 0.5 <= result <= 1.5


@pytest.mark.asyncio
async def test_random_choice(context: ProcessingContext):
    options = ["a", "b", "c"]
    node = RandomChoice(options=options)
    result = await node.process(context)
    assert result in options


@pytest.mark.asyncio
async def test_random_bool(context: ProcessingContext):
    node = RandomBool()
    result = await node.process(context)
    assert isinstance(result, bool)
