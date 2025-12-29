import pytest
from nodetool.dsl.graph import graph_result
from nodetool.dsl.nodetool.text import (
    Concat,
    Template,
    RegexReplace,
    Split,
    Join,
    Contains,
)
from nodetool.dsl.nodetool.output import Output

# Example 1: Basic text concatenation
concat_node = Concat(a="Hello, ", b="World!")
text_concat = Output(name="text_concat", value=concat_node.output)

# Example 2: Template with variable substitution
template_node = Template(
    string="Hello, {{ name }}! Today is {{ day }}.",
    values={"name": "Alice", "day": "Monday"},
)
template_text = Output(
    name="template_text",
    value=template_node.output,
)

# Example 3: Regex replacement
regex_node = RegexReplace(
    text="The color is grey and gray",
    pattern="gr[ae]y",
    replacement="blue",
    count=1,
)
regex_replace = Output(
    name="regex_replace",
    value=regex_node.output,
)

# Example 4: Split and join operation
split_node = Split(text="apple,banana,orange", delimiter=",")
join_node = Join(
    strings=split_node.output,
    separator=" | ",
)
split_join = Output(
    name="split_join",
    value=join_node.output,
)

# Example 5: Text contains check
contains_node = Contains(
    text="Python programming is fun", substring="programming", case_sensitive=True
)
contains_check = Output(
    name="contains_check",
    value=contains_node.output,
)


@pytest.mark.asyncio
async def test_text_concat():
    result = await graph_result(text_concat)
    assert result["text_concat"] == "Hello, World!"


@pytest.mark.asyncio
async def test_template_text():
    result = await graph_result(template_text)
    assert result["template_text"] == "Hello, Alice! Today is Monday."


@pytest.mark.asyncio
async def test_regex_replace():
    result = await graph_result(regex_replace)
    assert result["regex_replace"] == "The color is blue and gray"


@pytest.mark.asyncio
async def test_split_join():
    result = await graph_result(split_join)
    assert result["split_join"] == "apple | banana | orange"


@pytest.mark.asyncio
async def test_contains_check():
    result = await graph_result(contains_check)
    assert result["contains_check"]
