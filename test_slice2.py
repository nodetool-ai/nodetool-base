# Force reimport by reloading the module
import importlib
import nodetool.nodes.nodetool.list as list_module
importlib.reload(list_module)

import asyncio
from nodetool.nodes.nodetool.list import Slice
from nodetool.workflows.processing_context import ProcessingContext

# Check the source code
import inspect
print("Current process method source:")
print(inspect.getsource(Slice.process))
print("")

async def test():
    test_values = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    ctx = ProcessingContext(user_id='test', auth_token='test')
    
    # Test 1: stop=0 should now return all elements
    node = Slice(values=test_values, start=0, stop=0, step=1)
    result = await node.process(ctx)
    print(f'Test 1 - stop=0: {len(result)} elements')

asyncio.run(test())
