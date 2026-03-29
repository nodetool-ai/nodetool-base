import asyncio
import pytest
import sys
from pydantic import BaseModel

class BaseNodeMock(BaseModel):
    pass

class ProcessingContextMock:
    pass

sys.modules['nodetool.workflows.base_node'] = type('base_node', (), {'BaseNode': BaseNodeMock})
sys.modules['nodetool.workflows.processing_context'] = type('processing_context', (), {'ProcessingContext': ProcessingContextMock})
sys.modules['nodetool.metadata.types'] = type('types', (), {'TextRef': str})

from nodetool.nodes.nodetool.list import Sum, Average, Minimum, Maximum, Product

@pytest.mark.asyncio
async def test_sum():
    node = Sum(values=[1, 2.5, 3])
    assert await node.process(None) == 6.5

    with pytest.raises(ValueError):
        await Sum(values=[]).process(None)

    with pytest.raises(ValueError):
        await Sum(values=[1, "a"]).process(None)

@pytest.mark.asyncio
async def test_average():
    node = Average(values=[1, 2, 3])
    assert await node.process(None) == 2.0

    with pytest.raises(ValueError):
        await Average(values=[]).process(None)

    with pytest.raises(ValueError):
        await Average(values=[1, "a"]).process(None)

@pytest.mark.asyncio
async def test_minimum():
    node = Minimum(values=[5, 2.5, 8])
    assert await node.process(None) == 2.5

    with pytest.raises(ValueError):
        await Minimum(values=[]).process(None)

    with pytest.raises(ValueError):
        await Minimum(values=[1, "a"]).process(None)

@pytest.mark.asyncio
async def test_maximum():
    node = Maximum(values=[5, 2.5, 8])
    assert await node.process(None) == 8.0

    with pytest.raises(ValueError):
        await Maximum(values=[]).process(None)

    with pytest.raises(ValueError):
        await Maximum(values=[1, "a"]).process(None)

@pytest.mark.asyncio
async def test_product():
    node = Product(values=[2, 3.5, 4])
    assert await node.process(None) == 28.0

    with pytest.raises(ValueError):
        await Product(values=[]).process(None)

    with pytest.raises(ValueError):
        await Product(values=[1, "a"]).process(None)

if __name__ == '__main__':
    asyncio.run(test_sum())
    asyncio.run(test_average())
    asyncio.run(test_minimum())
    asyncio.run(test_maximum())
    asyncio.run(test_product())
    print("All tests passed!")
