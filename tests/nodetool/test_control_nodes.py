import pytest

from nodetool.workflows.processing_context import ProcessingContext
from nodetool.metadata.types import Event
from nodetool.nodes.nodetool.control import If, IteratorNode, CollectorNode


@pytest.fixture
def context():
    return ProcessingContext(user_id="test", auth_token="test")


@pytest.mark.asyncio
async def test_if_node_true_false(context: ProcessingContext):
    true_node = If(condition=True, value=123)
    false_node = If(condition=False, value=456)

    async def collect(node):
        outs = []
        async for k, v in node.gen_process(context):
            outs.append((k, v))
        return outs

    true_outs = await collect(true_node)
    false_outs = await collect(false_node)

    assert ("if_true", 123) in true_outs and all(k != "if_false" for k, _ in true_outs)
    assert ("if_false", 456) in false_outs and all(k != "if_true" for k, _ in false_outs)


@pytest.mark.asyncio
async def test_iterator_and_collector(context: ProcessingContext):
    items = ["a", "b", "c"]
    it = IteratorNode(input_list=items)
    coll = CollectorNode()

    # Iterate and feed events into collector
    emitted = []
    async for k, v in it.gen_process(context):
        emitted.append((k, v))
        if k == "output":
            coll.input_item = v
        if k == "event" and isinstance(v, Event):
            # handle events from iterator (iterator and done)
            async for ck, cv in coll.handle_event(context, v):
                emitted.append((ck, cv))

    # Find final output from collector
    collected = None
    for k, v in emitted:
        if k == "output" and isinstance(v, list):
            collected = v
            break

    assert collected == items

