import pytest
from nodetool.dsl.graph import graph_result
from nodetool.dsl.nodetool.list import (
    Append,
    Chunk,
    FilterDictsByValue,
    FilterNone,
    FilterNumbers,
    Flatten,
    Sort,
    Transform,
    Union,
)
from nodetool.dsl.nodetool.output import StringOutput, ListOutput
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
basic_list_ops = StringOutput(
    name="basic_list_ops",
    value=join_list.output,
)

# List transformations and filtering
transform_list = Transform(
    values=["1", "2", "3", "4", "5"],
    transform_type=Transform.TransformType("to_float"),
)
filter_numbers_node = FilterNumbers(
    values=transform_list.output,
    filter_type=FilterNumbers.FilterNumberType("greater_than"),
    value=2.5,
)
list_transform = ListOutput(
    name="list_transform",
    value=filter_numbers_node.output,
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
list_sets = ListOutput(
    name="list_sets", value=union_lists.output
)

# Complex list manipulation
flatten_list = Flatten(values=[[1, None, 2], [3, None], [4, 5]], max_depth=1)
filter_none_list = FilterNone(values=flatten_list.output)
chunk_node = Chunk(
    values=filter_none_list.output,
    chunk_size=2,
)
complex_list = ListOutput(
    name="complex_list",
    value=chunk_node.output,
)

# Dictionary list operations
filter_dicts = FilterDictsByValue(
    values=[
        {"name": "Alice", "age": 25},
        {"name": "Bob", "age": 30},
        {"name": "Charlie", "age": 35},
    ],
    key="name",
    filter_type=FilterDictsByValue.FilterType("contains"),
    criteria="Bob",
)
dict_list_ops = ListOutput(
    name="dict_list_ops",
    value=filter_dicts.output,
)


@pytest.mark.asyncio
async def test_basic_list_ops():
    result = await graph_result(basic_list_ops)
    assert result["basic_list_ops"] == "apple, banana, cherry, date"


@pytest.mark.asyncio
async def test_list_transform():
    result = await graph_result(list_transform)
    assert isinstance(result["list_transform"], list)
    assert all(x > 2.5 for x in result["list_transform"])


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
    assert result["complex_list"] == [[1, 2], [3, 4], [5]]
