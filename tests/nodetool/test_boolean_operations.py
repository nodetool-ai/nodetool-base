import pytest
from nodetool.dsl.graph import graph_result
from nodetool.dsl.nodetool.boolean import (
    Compare,
    LogicalOperator,
    ConditionalSwitch,
    IsIn,
)
from nodetool.dsl.nodetool.output import Output

# Basic comparison
comparison = Compare(a=5, b=3, comparison=Compare.Comparison(">"))
basic_comparison = Output(
    name="basic_comparison",
    value=comparison.output,
)

# Logical operators
comparison1 = Compare(a=10, b=5, comparison=Compare.Comparison(">"))
comparison2 = Compare(a=20, b=15, comparison=Compare.Comparison(">"))
logical_op = LogicalOperator(
    a=comparison1.output,
    b=comparison2.output,
    operation=LogicalOperator.BooleanOperation("and"),
)
logical_ops = Output(
    name="logical_ops",
    value=logical_op.output,
)

# Conditional switch
condition_check = Compare(a=42, b=42, comparison=Compare.Comparison("=="))
conditional_switch = ConditionalSwitch(
    condition=condition_check.output,
    if_true="Values are equal",
    if_false="Values are different",
)
conditional = Output(
    name="conditional",
    value=conditional_switch.output,
)

# List membership check
membership_check = IsIn(value=5, options=[1, 3, 5, 7, 9])
membership = Output(
    name="membership",
    value=membership_check.output,
)


@pytest.mark.asyncio
async def test_basic_comparison():
    result = await graph_result(basic_comparison)
    assert result["basic_comparison"]  # 5 > 3


@pytest.mark.asyncio
async def test_logical_ops():
    result = await graph_result(logical_ops)
    assert result["logical_ops"]  # (10 > 5) and (20 > 15)


@pytest.mark.asyncio
async def test_conditional():
    result = await graph_result(conditional)
    assert result["conditional"] == "Values are equal"


@pytest.mark.asyncio
async def test_membership():
    result = await graph_result(membership)
    assert result["membership"]  # 5 in [1, 3, 5, 7, 9]
