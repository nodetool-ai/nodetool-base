"""
Decision-engine workflow tests.

These scenarios model the kind of conditional routing and scoring
logic that users build in the visual editor: numeric thresholds,
compound boolean gates, negations, and multi-branch outputs.
All DSL-graph-based except for list-input nodes which use direct calls.
"""

import pytest

from nodetool.dsl.graph import create_graph, run_graph_async
from nodetool.dsl.nodetool.constant import Integer, Float, Bool
from nodetool.dsl.nodetool.boolean import (
    Compare,
    ConditionalSwitch,
    LogicalOperator,
    Not as BoolNot,
)
from nodetool.dsl.nodetool.text import FormatText, ToUppercase
from nodetool.dsl.nodetool.output import Output

from nodetool.workflows.processing_context import ProcessingContext
from nodetool.nodes.nodetool.boolean import (
    IsNone,
    IsIn,
    All as AllNode,
    Some as SomeNode,
)


@pytest.fixture
def ctx():
    return ProcessingContext(user_id="test", auth_token="test")


# ---------------------------------------------------------------------------
# Scenario: Traffic-light classifier
# ---------------------------------------------------------------------------
class TestTrafficLight:
    """speed ≤ 30 → green, ≤ 60 → amber, else red."""

    @pytest.mark.asyncio
    async def test_slow_speed_is_green(self):
        spd = Integer(value=25)
        ok = Compare(
            a=spd.output, b=30,
            comparison=Compare.Comparison.LESS_THAN_OR_EQUAL,
        )
        colour = ConditionalSwitch(
            condition=ok.output, if_true="green", if_false="not-green"
        )
        sink = Output(name="light", value=colour.output)
        bag = await run_graph_async(create_graph(sink))
        assert bag["light"] == "green"

    @pytest.mark.asyncio
    async def test_fast_speed_is_not_green(self):
        spd = Integer(value=80)
        ok = Compare(
            a=spd.output, b=30,
            comparison=Compare.Comparison.LESS_THAN_OR_EQUAL,
        )
        colour = ConditionalSwitch(
            condition=ok.output, if_true="green", if_false="not-green"
        )
        sink = Output(name="light", value=colour.output)
        bag = await run_graph_async(create_graph(sink))
        assert bag["light"] == "not-green"


# ---------------------------------------------------------------------------
# Scenario: Compound range check  (lo ≤ x ≤ hi)
# ---------------------------------------------------------------------------
class TestRangeGate:
    @pytest.mark.asyncio
    async def test_value_inside_band(self):
        temp = Float(value=36.8)
        above_lo = Compare(
            a=temp.output, b=36.0,
            comparison=Compare.Comparison.GREATER_THAN_OR_EQUAL,
        )
        below_hi = Compare(
            a=temp.output, b=37.5,
            comparison=Compare.Comparison.LESS_THAN_OR_EQUAL,
        )
        normal = LogicalOperator(
            a=above_lo.output, b=below_hi.output,
            operation=LogicalOperator.BooleanOperation.AND,
        )
        diagnosis = ConditionalSwitch(
            condition=normal.output, if_true="normal", if_false="abnormal"
        )
        sink = Output(name="dx", value=diagnosis.output)
        bag = await run_graph_async(create_graph(sink))
        assert bag["dx"] == "normal"

    @pytest.mark.asyncio
    async def test_value_above_band(self):
        temp = Float(value=39.1)
        above_lo = Compare(
            a=temp.output, b=36.0,
            comparison=Compare.Comparison.GREATER_THAN_OR_EQUAL,
        )
        below_hi = Compare(
            a=temp.output, b=37.5,
            comparison=Compare.Comparison.LESS_THAN_OR_EQUAL,
        )
        normal = LogicalOperator(
            a=above_lo.output, b=below_hi.output,
            operation=LogicalOperator.BooleanOperation.AND,
        )
        diagnosis = ConditionalSwitch(
            condition=normal.output, if_true="normal", if_false="abnormal"
        )
        sink = Output(name="dx", value=diagnosis.output)
        bag = await run_graph_async(create_graph(sink))
        assert bag["dx"] == "abnormal"


# ---------------------------------------------------------------------------
# Scenario: NOT gate — invert a condition
# ---------------------------------------------------------------------------
class TestInvertedCondition:
    @pytest.mark.asyncio
    async def test_not_gate_flips_true(self):
        flag = Bool(value=True)
        flipped = BoolNot(value=flag.output)
        sink = Output(name="inv", value=flipped.output)
        bag = await run_graph_async(create_graph(sink))
        assert bag["inv"] is False

    @pytest.mark.asyncio
    async def test_double_negation(self):
        flag = Bool(value=False)
        flip1 = BoolNot(value=flag.output)
        flip2 = BoolNot(value=flip1.output)
        sink = Output(name="same", value=flip2.output)
        bag = await run_graph_async(create_graph(sink))
        assert bag["same"] is False


# ---------------------------------------------------------------------------
# Scenario: XOR — exactly-one-of-two
# ---------------------------------------------------------------------------
class TestExclusiveOr:
    @pytest.mark.asyncio
    async def test_one_true_gives_true(self):
        a = Bool(value=True)
        b = Bool(value=False)
        xor = LogicalOperator(
            a=a.output, b=b.output,
            operation=LogicalOperator.BooleanOperation.XOR,
        )
        sink = Output(name="xor", value=xor.output)
        bag = await run_graph_async(create_graph(sink))
        assert bag["xor"] is True

    @pytest.mark.asyncio
    async def test_both_true_gives_false(self):
        a = Bool(value=True)
        b = Bool(value=True)
        xor = LogicalOperator(
            a=a.output, b=b.output,
            operation=LogicalOperator.BooleanOperation.XOR,
        )
        sink = Output(name="xor", value=xor.output)
        bag = await run_graph_async(create_graph(sink))
        assert bag["xor"] is False


# ---------------------------------------------------------------------------
# Scenario: Multi-output scoring dashboard
# ---------------------------------------------------------------------------
class TestScoringDashboard:
    @pytest.mark.asyncio
    async def test_three_metrics(self):
        score = Integer(value=72)

        above_50 = Compare(
            a=score.output, b=50,
            comparison=Compare.Comparison.GREATER_THAN,
        )
        above_90 = Compare(
            a=score.output, b=90,
            comparison=Compare.Comparison.GREATER_THAN,
        )
        exactly_72 = Compare(
            a=score.output, b=72,
            comparison=Compare.Comparison.EQUAL,
        )

        o1 = Output(name="pass", value=above_50.output)
        o2 = Output(name="distinction", value=above_90.output)
        o3 = Output(name="exact", value=exactly_72.output)

        bag = await run_graph_async(create_graph(o1, o2, o3))
        assert bag["pass"] is True
        assert bag["distinction"] is False
        assert bag["exact"] is True


# ---------------------------------------------------------------------------
# Scenario: Conditional message composition
# ---------------------------------------------------------------------------
class TestConditionalMessage:
    @pytest.mark.asyncio
    async def test_vip_greeting(self):
        is_vip = Bool(value=True)
        greeting = ConditionalSwitch(
            condition=is_vip.output,
            if_true="Welcome back, VIP!",
            if_false="Hello, guest.",
        )
        upper = ToUppercase(text=greeting.output)
        sink = Output(name="msg", value=upper.output)
        bag = await run_graph_async(create_graph(sink))
        assert bag["msg"] == "WELCOME BACK, VIP!"

    @pytest.mark.asyncio
    async def test_guest_greeting(self):
        is_vip = Bool(value=False)
        greeting = ConditionalSwitch(
            condition=is_vip.output,
            if_true="Welcome back, VIP!",
            if_false="Hello, guest.",
        )
        upper = ToUppercase(text=greeting.output)
        sink = Output(name="msg", value=upper.output)
        bag = await run_graph_async(create_graph(sink))
        assert bag["msg"] == "HELLO, GUEST."


# ---------------------------------------------------------------------------
# Scenario: NAND gate (direct node test)
# ---------------------------------------------------------------------------
class TestNandGate:
    @pytest.mark.asyncio
    async def test_nand_both_true(self):
        a = Bool(value=True)
        b = Bool(value=True)
        nand = LogicalOperator(
            a=a.output, b=b.output,
            operation=LogicalOperator.BooleanOperation.NAND,
        )
        sink = Output(name="out", value=nand.output)
        bag = await run_graph_async(create_graph(sink))
        assert bag["out"] is False

    @pytest.mark.asyncio
    async def test_nand_one_false(self):
        a = Bool(value=True)
        b = Bool(value=False)
        nand = LogicalOperator(
            a=a.output, b=b.output,
            operation=LogicalOperator.BooleanOperation.NAND,
        )
        sink = Output(name="out", value=nand.output)
        bag = await run_graph_async(create_graph(sink))
        assert bag["out"] is True


# ---------------------------------------------------------------------------
# Direct-invocation tests for list-input boolean helpers
# ---------------------------------------------------------------------------
class TestListBooleanHelpers:
    """IsIn, IsNone, All, Some don't work well with DSL list-handles so
    we test them via direct process() calls."""

    @pytest.mark.asyncio
    async def test_is_in_found(self, ctx):
        assert await IsIn(
            value="banana", options=["apple", "banana", "cherry"]
        ).process(ctx) is True

    @pytest.mark.asyncio
    async def test_is_in_missing(self, ctx):
        assert await IsIn(
            value="fig", options=["apple", "banana"]
        ).process(ctx) is False

    @pytest.mark.asyncio
    async def test_is_none_with_none(self, ctx):
        assert await IsNone(value=None).process(ctx) is True

    @pytest.mark.asyncio
    async def test_is_none_with_value(self, ctx):
        assert await IsNone(value=42).process(ctx) is False

    @pytest.mark.asyncio
    async def test_all_true(self, ctx):
        assert await AllNode(values=[True, True, True]).process(ctx) is True

    @pytest.mark.asyncio
    async def test_all_with_false(self, ctx):
        assert await AllNode(values=[True, False, True]).process(ctx) is False

    @pytest.mark.asyncio
    async def test_some_with_one_true(self, ctx):
        assert await SomeNode(values=[False, True, False]).process(ctx) is True

    @pytest.mark.asyncio
    async def test_some_all_false(self, ctx):
        assert await SomeNode(values=[False, False]).process(ctx) is False


# ---------------------------------------------------------------------------
# Scenario: Comparison chained into template
# ---------------------------------------------------------------------------
class TestCompareAndFormat:
    @pytest.mark.asyncio
    async def test_threshold_label(self):
        val = Integer(value=88)
        above = Compare(
            a=val.output, b=70,
            comparison=Compare.Comparison.GREATER_THAN_OR_EQUAL,
        )
        lbl = ConditionalSwitch(
            condition=above.output, if_true="PASS", if_false="FAIL"
        )
        msg = FormatText(
            template="Result: {{ status }}",
            status=lbl.output,
        )
        sink = Output(name="report", value=msg.output)
        bag = await run_graph_async(create_graph(sink))
        assert bag["report"] == "Result: PASS"
