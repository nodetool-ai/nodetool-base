import pytest
from nodetool.dsl.graph import graph_result
from nodetool.dsl.nodetool.list import (
    Append,
    Chunk,
    Flatten,
    Sort,
    Union,
)
from nodetool.dsl.nodetool.output import Output
from nodetool.dsl.nodetool.text import Join

# Basic list operations
append_list = Append(values=["banana", "apple", "cherry"], value="date")
sort_list = Sort(
    values=append_list.output,
    order=Sort.SortOrder("ascending"),
)
join_list = Join(
    strings=sort_list.output,
    separator=", ",
)
basic_list_ops = Output(
    name="basic_list_ops",
    value=join_list.output,
)

# # List aggregation operations
# list_aggregation = DictionaryOutput(
#     name="list_aggregation",
#     value={
#         "sum": Sum(values=[1, 2, 3, 4, 5]),
#         "average": Average(values=[1, 2, 3, 4, 5]),
#         "max": Maximum(values=[1, 2, 3, 4, 5]),
#         "min": Minimum(values=[1, 2, 3, 4, 5]),
#     },
# )

# List set operations
union_lists = Union(list1=[1, 2, 3, 4], list2=[3, 4, 5, 6])
list_sets = Output(
    name="list_sets", value=union_lists.output
)

chunk_node = Chunk(values=[1, 2, 3, 4, 5], chunk_size=2)
chunk_list = Output(
    name="chunk_list", value=chunk_node.output
)

# Complex list manipulation
flatten_list = Flatten(values=[[1, 2], [3, 4], [5]], max_depth=1)
complex_list = Output(
    name="complex_list",
    value=flatten_list.output,
)

@pytest.mark.asyncio
async def test_basic_list_ops():
    result = await graph_result(basic_list_ops)
    assert result["basic_list_ops"] == "apple, banana, cherry, date"


# @pytest.mark.asyncio
# async def test_list_aggregation():
#     result = await graph_result(list_aggregation)
#     assert result["sum"] == 15
#     assert result["average"] == 3.0
#     assert result["max"] == 5
#     assert result["min"] == 1


@pytest.mark.asyncio
async def test_list_sets():
    result = await graph_result(list_sets)
    assert isinstance(result["list_sets"], list)
    assert set(result["list_sets"]) == {1, 2, 3, 4, 5, 6}


@pytest.mark.asyncio
async def test_complex_list():
    result = await graph_result(complex_list)
    assert isinstance(result["complex_list"], list)
    assert result["complex_list"] == [1, 2, 3, 4, 5]


@pytest.mark.asyncio
async def test_chunk_list():
    result = await graph_result(chunk_list)
    assert isinstance(result["chunk_list"], list)
    assert result["chunk_list"] == [[1, 2], [3, 4], [5]]