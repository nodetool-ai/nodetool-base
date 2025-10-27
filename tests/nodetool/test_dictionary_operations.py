import pytest
from nodetool.dsl.graph import graph_result
from nodetool.dsl.nodetool.dictionary import (
    ArgMax,
    Combine,
    Filter,
    GetValue,
    Update,
    Zip,
)
from nodetool.dsl.nodetool.output import StringOutput, IntegerOutput, DictionaryOutput

# Create and manipulate dictionaries
update_dict = Update(
    dictionary={"name": "Alice", "age": 30},
    new_pairs={"city": "New York", "role": "Developer"},
)
get_role = GetValue(
    dictionary=update_dict.output,
    key="role",
    default="Unknown",
)
make_dict = StringOutput(
    name="make_dict",
    value=get_role.output,
)

# Combine dictionaries
combine_dicts = Combine(
    dict_a={"a": 1, "b": 2},
    dict_b={"b": 3, "c": 4},
)
get_combined = GetValue(
    dictionary=combine_dicts.output,
    key="b",
    default=0,
)
combined_dict = IntegerOutput(
    name="combined_dict",
    value=get_combined.output,
)

# Filter dictionary keys
filter_dict = Filter(
    dictionary={"name": "Bob", "age": 25, "city": "London", "country": "UK"},
    keys=["name", "city"],
)
filtered_dict = DictionaryOutput(
    name="filtered_dict",
    value=filter_dict.output,
)

# Create dictionary from parallel lists
zip_dicts = Zip(keys=["a", "b", "c"], values=[1, 2, 3])
zipped_dict = DictionaryOutput(
    name="zipped_dict", value=zip_dicts.output
)

# Find maximum value in dictionary
argmax_node = ArgMax(scores={"cat": 0.7, "dog": 0.9, "bird": 0.3})
argmax_example = StringOutput(
    name="argmax_example",
    value=argmax_node.output,
)


@pytest.mark.asyncio
async def test_make_dict():
    result = await graph_result(make_dict)
    assert result["make_dict"] == "Developer"


@pytest.mark.asyncio
async def test_combined_dict():
    result = await graph_result(combined_dict)
    assert result["combined_dict"] == 3


@pytest.mark.asyncio
async def test_filtered_dict():
    result = await graph_result(filtered_dict)
    assert result["filtered_dict"] == {"name": "Bob", "city": "London"}


@pytest.mark.asyncio
async def test_zipped_dict():
    result = await graph_result(zipped_dict)
    assert result["zipped_dict"] == {"a": 1, "b": 2, "c": 3}


@pytest.mark.asyncio
async def test_argmax_example():
    result = await graph_result(argmax_example)
    assert result["argmax_example"] == "dog"
