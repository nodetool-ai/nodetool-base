import pytest
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.nodes.lib.json import (
    ParseDict,
    ParseList,
    StringifyJSON,
    GetJSONPathStr,
    GetJSONPathInt,
    GetJSONPathFloat,
    GetJSONPathBool,
    GetJSONPathList,
    GetJSONPathDict,
    ValidateJSON,
    FilterJSON,
    JSONTemplate,
)


@pytest.fixture
def context():
    return ProcessingContext(user_id="test", auth_token="test")


@pytest.mark.asyncio
async def test_parse_dict(context: ProcessingContext):
    node = ParseDict(json_string='{"a": 1, "b": 2}')
    result = await node.process(context)
    assert result == {"a": 1, "b": 2}


@pytest.mark.asyncio
async def test_parse_dict_invalid_raises(context: ProcessingContext):
    node = ParseDict(json_string='[1, 2, 3]')
    with pytest.raises(ValueError):
        await node.process(context)


@pytest.mark.asyncio
async def test_parse_list(context: ProcessingContext):
    node = ParseList(json_string='[1, 2, 3]')
    result = await node.process(context)
    assert result == [1, 2, 3]


@pytest.mark.asyncio
async def test_parse_list_invalid_raises(context: ProcessingContext):
    node = ParseList(json_string='{"a": 1}')
    with pytest.raises(ValueError):
        await node.process(context)


@pytest.mark.asyncio
async def test_stringify_json(context: ProcessingContext):
    data = {"a": 1, "b": [2, 3]}
    node = StringifyJSON(data=data, indent=2)
    result = await node.process(context)
    # Round-trip to avoid strict spacing comparisons
    import json

    assert json.loads(result) == data


@pytest.mark.asyncio
async def test_get_json_path_variants(context: ProcessingContext):
    data = {"a": {"b": {"c": 1}, "d": [10, 20]}, "e": "x", "f": True}

    s = await GetJSONPathStr(data=data, path="e", default="").process(context)
    i = await GetJSONPathInt(data=data, path="a.d.1", default=0).process(context)
    fl = await GetJSONPathFloat(data=data, path="a.b.c", default=0.0).process(
        context
    )
    b = await GetJSONPathBool(data=data, path="f", default=False).process(context)
    lst = await GetJSONPathList(data=data, path="a.d", default=[]).process(context)
    dct = await GetJSONPathDict(data=data, path="a.b", default={}).process(context)

    assert s == "x"
    assert i == 20
    assert fl == 1.0
    assert b is True
    assert lst == [10, 20]
    assert dct == {"c": 1}


@pytest.mark.asyncio
async def test_get_json_path_defaults(context: ProcessingContext):
    data = {"a": {}}
    s = await GetJSONPathStr(data=data, path="missing", default="def").process(
        context
    )
    i = await GetJSONPathInt(data=data, path="a.d.5", default=7).process(context)
    assert s == "def"
    assert i == 7


@pytest.mark.asyncio
async def test_validate_json(context: ProcessingContext):
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "number"}},
        "required": ["name"],
        "additionalProperties": False,
    }
    valid = ValidateJSON(data={"name": "Alice", "age": 30}, json_schema=schema)
    invalid = ValidateJSON(data={"age": 30}, json_schema=schema)

    assert await valid.process(context) is True
    assert await invalid.process(context) is False


@pytest.mark.asyncio
async def test_filter_json(context: ProcessingContext):
    items = [
        {"id": 1, "type": "a"},
        {"id": 2, "type": "b"},
        {"id": 3, "type": "a"},
    ]
    node = FilterJSON(array=items, key="type", value="a")
    result = await node.process(context)
    assert result == [{"id": 1, "type": "a"}, {"id": 3, "type": "a"}]


@pytest.mark.asyncio
async def test_json_template(context: ProcessingContext):
    node = JSONTemplate(
        template='{"name": "$user", "age": $age}',
        values={"user": "John", "age": 30},
    )
    result = await node.process(context)
    assert result == {"name": "John", "age": 30}

    # Invalid JSON after substitution should raise
    bad = JSONTemplate(template="{invalid}", values={})
    with pytest.raises(ValueError):
        await bad.process(context)

