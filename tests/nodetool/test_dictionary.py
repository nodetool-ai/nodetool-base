import pytest
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.nodes.nodetool.dictionary import (
    GetValue,
    Update,
    Remove,
    ParseJSON,
    Zip,
    Combine,
    Filter,
    ReduceDictionaries,
    MakeDictionary,
    ArgMax,
)

sample_dict = {"a": 1, "b": 2, "c": 3}
sample_json = '{"a": 1, "b": 2, "c": 3}'


@pytest.fixture
def context():
    return ProcessingContext(user_id="test", auth_token="test")


@pytest.mark.asyncio
async def test_get_value(context: ProcessingContext):
    node = GetValue(dictionary=sample_dict, key="a")
    result = await node.process(context)
    assert result == 1


@pytest.mark.asyncio
async def test_update(context: ProcessingContext):
    node = Update(dictionary=sample_dict.copy(), new_pairs={"d": 4})
    result = await node.process(context)
    assert result == {"a": 1, "b": 2, "c": 3, "d": 4}


@pytest.mark.asyncio
async def test_remove(context: ProcessingContext):
    node = Remove(dictionary=sample_dict.copy(), key="a")
    result = await node.process(context)
    assert result == {"b": 2, "c": 3}


@pytest.mark.asyncio
async def test_parse_json(context: ProcessingContext):
    node = ParseJSON(json_string=sample_json)
    result = await node.process(context)
    assert result == sample_dict


@pytest.mark.asyncio
async def test_zip(context: ProcessingContext):
    node = Zip(keys=["a", "b", "c"], values=[1, 2, 3])
    result = await node.process(context)
    assert result == sample_dict


@pytest.mark.asyncio
async def test_combine(context: ProcessingContext):
    node = Combine(dict_a={"a": 1}, dict_b={"b": 2})
    result = await node.process(context)
    assert result == {"a": 1, "b": 2}


@pytest.mark.asyncio
async def test_filter(context: ProcessingContext):
    node = Filter(dictionary=sample_dict, keys=["a", "b"])
    result = await node.process(context)
    assert result == {"a": 1, "b": 2}


@pytest.mark.asyncio
async def test_reduce_dictionaries(context: ProcessingContext):
    node = ReduceDictionaries(
        dictionaries=[
            {"id": 1, "value": "a"},
            {"id": 2, "value": "b"},
            {"id": 3, "value": "c"},
        ],
        key_field="id",
        value_field="value",
    )
    result = await node.process(context)
    assert result == {1: "a", 2: "b", 3: "c"}


@pytest.mark.asyncio
async def test_empty_dictionary(context: ProcessingContext):
    node = GetValue(dictionary={}, key="a")
    result = await node.process(context)
    assert result == ()


@pytest.mark.asyncio
async def test_update_existing_key(context: ProcessingContext):
    node = Update(dictionary=sample_dict.copy(), new_pairs={"a": 10})
    result = await node.process(context)
    assert result == {"a": 10, "b": 2, "c": 3}


@pytest.mark.asyncio
async def test_remove_nonexistent_key(context: ProcessingContext):
    node = Remove(dictionary=sample_dict.copy(), key="z")
    result = await node.process(context)
    assert result == sample_dict


@pytest.mark.asyncio
async def test_parse_invalid_json(context: ProcessingContext):
    node = ParseJSON(json_string='{"a": 1, "b": 2, "c": 3')
    with pytest.raises(Exception):
        await node.process(context)


@pytest.mark.asyncio
async def test_zip_mismatched_lengths(context: ProcessingContext):
    node = Zip(keys=["a", "b"], values=[1, 2, 3])
    await node.process(context)


@pytest.mark.asyncio
async def test_combine_with_overlap(context: ProcessingContext):
    node = Combine(dict_a={"a": 1}, dict_b={"a": 2, "b": 2})
    result = await node.process(context)
    assert result == {"a": 2, "b": 2}


@pytest.mark.asyncio
async def test_filter_nonexistent_keys(context: ProcessingContext):
    node = Filter(dictionary=sample_dict, keys=["x", "y"])
    result = await node.process(context)
    assert result == {}


@pytest.mark.asyncio
async def test_argmax(context: ProcessingContext):
    """Test ArgMax returns the key with highest value."""
    node = ArgMax(scores={"cat": 0.7, "dog": 0.2, "bird": 0.1})
    result = await node.process(context)
    assert result == "cat"


@pytest.mark.asyncio
async def test_argmax_with_equal_values(context: ProcessingContext):
    """Test ArgMax when there are ties (returns first max)."""
    node = ArgMax(scores={"a": 0.5, "b": 0.3, "c": 0.5})
    result = await node.process(context)
    assert result in ["a", "c"]  # Either is valid due to dict ordering


@pytest.mark.asyncio
async def test_argmax_empty_raises(context: ProcessingContext):
    """Test ArgMax raises error on empty dictionary."""
    node = ArgMax(scores={})
    with pytest.raises(ValueError, match="cannot be empty"):
        await node.process(context)


@pytest.mark.asyncio
async def test_make_dictionary(context: ProcessingContext):
    """Test MakeDictionary creates dictionary from dynamic properties."""
    node = MakeDictionary()
    node._dynamic_properties = {"key1": "value1", "key2": 42}
    result = await node.process(context)
    assert result == {"key1": "value1", "key2": 42}


@pytest.mark.asyncio
async def test_make_dictionary_empty(context: ProcessingContext):
    """Test MakeDictionary with no dynamic properties."""
    node = MakeDictionary()
    node._dynamic_properties = {}
    result = await node.process(context)
    assert result == {}


@pytest.mark.asyncio
async def test_reduce_dictionaries_no_value_field(context: ProcessingContext):
    """Test ReduceDictionaries without value_field uses remaining dict as value."""
    node = ReduceDictionaries(
        dictionaries=[
            {"id": 1, "name": "Alice", "age": 30},
            {"id": 2, "name": "Bob", "age": 25},
        ],
        key_field="id",
        value_field="",  # No value field
    )
    result = await node.process(context)
    assert result == {
        1: {"name": "Alice", "age": 30},
        2: {"name": "Bob", "age": 25},
    }


@pytest.mark.asyncio
async def test_reduce_dictionaries_conflict_last(context: ProcessingContext):
    """Test ReduceDictionaries with conflict resolution LAST."""
    node = ReduceDictionaries(
        dictionaries=[
            {"id": 1, "value": "first"},
            {"id": 1, "value": "second"},
        ],
        key_field="id",
        value_field="value",
        conflict_resolution=ReduceDictionaries.ConflictResolution.LAST,
    )
    result = await node.process(context)
    assert result == {1: "second"}


@pytest.mark.asyncio
async def test_reduce_dictionaries_conflict_error(context: ProcessingContext):
    """Test ReduceDictionaries with conflict resolution ERROR."""
    node = ReduceDictionaries(
        dictionaries=[
            {"id": 1, "value": "first"},
            {"id": 1, "value": "second"},
        ],
        key_field="id",
        value_field="value",
        conflict_resolution=ReduceDictionaries.ConflictResolution.ERROR,
    )
    with pytest.raises(ValueError, match="Duplicate key"):
        await node.process(context)


@pytest.mark.asyncio
async def test_reduce_dictionaries_missing_key_field(context: ProcessingContext):
    """Test ReduceDictionaries raises error when key_field is missing."""
    node = ReduceDictionaries(
        dictionaries=[
            {"name": "Alice"},  # Missing 'id' field
        ],
        key_field="id",
        value_field="name",
    )
    with pytest.raises(ValueError, match="Key field 'id' not found"):
        await node.process(context)


@pytest.mark.asyncio
async def test_reduce_dictionaries_missing_value_field(context: ProcessingContext):
    """Test ReduceDictionaries raises error when value_field is missing."""
    node = ReduceDictionaries(
        dictionaries=[
            {"id": 1, "name": "Alice"},  # Missing 'value' field
        ],
        key_field="id",
        value_field="value",
    )
    with pytest.raises(ValueError, match="Value field 'value' not found"):
        await node.process(context)


@pytest.mark.asyncio
async def test_parse_json_non_dict(context: ProcessingContext):
    """Test ParseJSON raises error when input is not a dictionary."""
    node = ParseJSON(json_string='[1, 2, 3]')  # Valid JSON but not a dict
    with pytest.raises(ValueError, match="not a dictionary"):
        await node.process(context)


@pytest.mark.asyncio
async def test_get_value_with_default(context: ProcessingContext):
    """Test GetValue uses default when key not found."""
    node = GetValue(dictionary={"a": 1}, key="missing", default="default_value")
    result = await node.process(context)
    assert result == "default_value"
