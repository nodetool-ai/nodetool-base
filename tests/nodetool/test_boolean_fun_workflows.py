"""
Boolean workflow integration tests with humorous scenarios.
Each test builds a real-world-ish graph using Compare, ConditionalSwitch,
LogicalOperator, and BoolNot nodes, then asserts deterministic results.
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


# ---------------------------------------------------------------------------
# 1-2  Boss Approval Gateway
# ---------------------------------------------------------------------------
class TestBossApprovalGatewayWorkflow:
    @pytest.mark.asyncio
    async def test_budget_approved_when_under_limit(self):
        budget = Integer(value=4500)
        under_limit = Compare(
            a=budget.output,
            b=5000,
            comparison=Compare.Comparison.LESS_THAN_OR_EQUAL,
        )
        decision = ConditionalSwitch(
            condition=under_limit.output,
            if_true="APPROVED",
            if_false="DENIED",
        )
        sink = Output(name="verdict", value=decision.output)
        bag = await run_graph_async(create_graph(sink))
        assert bag["verdict"] == "APPROVED"

    @pytest.mark.asyncio
    async def test_budget_denied_when_over_limit(self):
        budget = Integer(value=12000)
        under_limit = Compare(
            a=budget.output,
            b=5000,
            comparison=Compare.Comparison.LESS_THAN_OR_EQUAL,
        )
        decision = ConditionalSwitch(
            condition=under_limit.output,
            if_true="APPROVED",
            if_false="DENIED",
        )
        sink = Output(name="verdict", value=decision.output)
        bag = await run_graph_async(create_graph(sink))
        assert bag["verdict"] == "DENIED"


# ---------------------------------------------------------------------------
# 3-4  Roller Coaster Height Check
# ---------------------------------------------------------------------------
class TestRollerCoasterHeightCheckWorkflow:
    @pytest.mark.asyncio
    async def test_tall_enough_to_ride(self):
        height = Integer(value=140)
        tall = Compare(
            a=height.output,
            b=120,
            comparison=Compare.Comparison.GREATER_THAN_OR_EQUAL,
        )
        msg = ConditionalSwitch(
            condition=tall.output,
            if_true="Enjoy the ride!",
            if_false="Sorry, too short!",
        )
        sink = Output(name="msg", value=msg.output)
        bag = await run_graph_async(create_graph(sink))
        assert bag["msg"] == "Enjoy the ride!"

    @pytest.mark.asyncio
    async def test_too_short_to_ride(self):
        height = Integer(value=100)
        tall = Compare(
            a=height.output,
            b=120,
            comparison=Compare.Comparison.GREATER_THAN_OR_EQUAL,
        )
        msg = ConditionalSwitch(
            condition=tall.output,
            if_true="Enjoy the ride!",
            if_false="Sorry, too short!",
        )
        sink = Output(name="msg", value=msg.output)
        bag = await run_graph_async(create_graph(sink))
        assert bag["msg"] == "Sorry, too short!"


# ---------------------------------------------------------------------------
# 5-6  Thermostat Brain
# ---------------------------------------------------------------------------
class TestThermostatBrainWorkflow:
    @pytest.mark.asyncio
    async def test_comfy_temperature(self):
        temp = Float(value=22.0)
        above_lo = Compare(
            a=temp.output,
            b=18.0,
            comparison=Compare.Comparison.GREATER_THAN_OR_EQUAL,
        )
        below_hi = Compare(
            a=temp.output,
            b=26.0,
            comparison=Compare.Comparison.LESS_THAN_OR_EQUAL,
        )
        comfy = LogicalOperator(
            a=above_lo.output,
            b=below_hi.output,
            operation=LogicalOperator.BooleanOperation.AND,
        )
        status = ConditionalSwitch(
            condition=comfy.output,
            if_true="comfy",
            if_false="too hot",
        )
        formatted = FormatText(
            template="Thermostat says: {{ status }}",
            status=status.output,
        )
        sink = Output(name="report", value=formatted.output)
        bag = await run_graph_async(create_graph(sink))
        assert bag["report"] == "Thermostat says: comfy"

    @pytest.mark.asyncio
    async def test_too_hot_temperature(self):
        temp = Float(value=35.0)
        above_lo = Compare(
            a=temp.output,
            b=18.0,
            comparison=Compare.Comparison.GREATER_THAN_OR_EQUAL,
        )
        below_hi = Compare(
            a=temp.output,
            b=26.0,
            comparison=Compare.Comparison.LESS_THAN_OR_EQUAL,
        )
        comfy = LogicalOperator(
            a=above_lo.output,
            b=below_hi.output,
            operation=LogicalOperator.BooleanOperation.AND,
        )
        status = ConditionalSwitch(
            condition=comfy.output,
            if_true="comfy",
            if_false="too hot",
        )
        formatted = FormatText(
            template="Thermostat says: {{ status }}",
            status=status.output,
        )
        sink = Output(name="report", value=formatted.output)
        bag = await run_graph_async(create_graph(sink))
        assert bag["report"] == "Thermostat says: too hot"


# ---------------------------------------------------------------------------
# 7-8  Monday Morning Mood Detector
# ---------------------------------------------------------------------------
class TestMondayMorningMoodDetectorWorkflow:
    @pytest.mark.asyncio
    async def test_good_mood_with_coffee_and_sleep(self):
        coffee = Integer(value=2)
        sleep = Integer(value=7)
        has_coffee = Compare(
            a=coffee.output,
            b=0,
            comparison=Compare.Comparison.GREATER_THAN,
        )
        enough_sleep = Compare(
            a=sleep.output,
            b=6,
            comparison=Compare.Comparison.GREATER_THAN_OR_EQUAL,
        )
        happy = LogicalOperator(
            a=has_coffee.output,
            b=enough_sleep.output,
            operation=LogicalOperator.BooleanOperation.AND,
        )
        mood = ConditionalSwitch(
            condition=happy.output,
            if_true="Ready to conquer Monday!",
            if_false="Do NOT talk to me.",
        )
        sink = Output(name="mood", value=mood.output)
        bag = await run_graph_async(create_graph(sink))
        assert bag["mood"] == "Ready to conquer Monday!"

    @pytest.mark.asyncio
    async def test_bad_mood_no_coffee(self):
        coffee = Integer(value=0)
        sleep = Integer(value=8)
        has_coffee = Compare(
            a=coffee.output,
            b=0,
            comparison=Compare.Comparison.GREATER_THAN,
        )
        enough_sleep = Compare(
            a=sleep.output,
            b=6,
            comparison=Compare.Comparison.GREATER_THAN_OR_EQUAL,
        )
        happy = LogicalOperator(
            a=has_coffee.output,
            b=enough_sleep.output,
            operation=LogicalOperator.BooleanOperation.AND,
        )
        mood = ConditionalSwitch(
            condition=happy.output,
            if_true="Ready to conquer Monday!",
            if_false="Do NOT talk to me.",
        )
        sink = Output(name="mood", value=mood.output)
        bag = await run_graph_async(create_graph(sink))
        assert bag["mood"] == "Do NOT talk to me."


# ---------------------------------------------------------------------------
# 9  Paranoia Security Audit (double NOT)
# ---------------------------------------------------------------------------
class TestParanoiaSecurityAuditWorkflow:
    @pytest.mark.asyncio
    async def test_double_not_preserves_truth(self):
        flag = Bool(value=True)
        flipped = BoolNot(value=flag.output)
        restored = BoolNot(value=flipped.output)
        o_original = Output(name="original", value=flag.output)
        o_restored = Output(name="restored", value=restored.output)
        bag = await run_graph_async(create_graph(o_original, o_restored))
        assert bag["original"] is True
        assert bag["restored"] is True


# ---------------------------------------------------------------------------
# 10-11  Dating App Compatibility
# ---------------------------------------------------------------------------
class TestDatingAppCompatibilityWorkflow:
    @pytest.mark.asyncio
    async def test_compatible_couple(self):
        age_diff = Integer(value=3)
        shared_hobbies = Integer(value=4)
        small_age_gap = Compare(
            a=age_diff.output,
            b=10,
            comparison=Compare.Comparison.LESS_THAN_OR_EQUAL,
        )
        enough_hobbies = Compare(
            a=shared_hobbies.output,
            b=2,
            comparison=Compare.Comparison.GREATER_THAN_OR_EQUAL,
        )
        compatible = LogicalOperator(
            a=small_age_gap.output,
            b=enough_hobbies.output,
            operation=LogicalOperator.BooleanOperation.AND,
        )
        verdict = ConditionalSwitch(
            condition=compatible.output,
            if_true="It's a match!",
            if_false="Maybe next time.",
        )
        sink = Output(name="result", value=verdict.output)
        bag = await run_graph_async(create_graph(sink))
        assert bag["result"] == "It's a match!"

    @pytest.mark.asyncio
    async def test_incompatible_couple(self):
        age_diff = Integer(value=25)
        shared_hobbies = Integer(value=0)
        small_age_gap = Compare(
            a=age_diff.output,
            b=10,
            comparison=Compare.Comparison.LESS_THAN_OR_EQUAL,
        )
        enough_hobbies = Compare(
            a=shared_hobbies.output,
            b=2,
            comparison=Compare.Comparison.GREATER_THAN_OR_EQUAL,
        )
        compatible = LogicalOperator(
            a=small_age_gap.output,
            b=enough_hobbies.output,
            operation=LogicalOperator.BooleanOperation.AND,
        )
        verdict = ConditionalSwitch(
            condition=compatible.output,
            if_true="It's a match!",
            if_false="Maybe next time.",
        )
        sink = Output(name="result", value=verdict.output)
        bag = await run_graph_async(create_graph(sink))
        assert bag["result"] == "Maybe next time."


# ---------------------------------------------------------------------------
# 12-13  Pirate Ship Speed Regulator
# ---------------------------------------------------------------------------
class TestPirateShipSpeedRegulatorWorkflow:
    @pytest.mark.asyncio
    async def test_full_speed_ahead(self):
        speed = Integer(value=30)
        fast = Compare(
            a=speed.output,
            b=20,
            comparison=Compare.Comparison.GREATER_THAN,
        )
        order = ConditionalSwitch(
            condition=fast.output,
            if_true="full sail",
            if_false="row harder",
        )
        announcement = FormatText(
            template="Captain says: {{ order }}!",
            order=order.output,
        )
        loud = ToUppercase(text=announcement.output)
        sink = Output(name="shout", value=loud.output)
        bag = await run_graph_async(create_graph(sink))
        assert bag["shout"] == "CAPTAIN SAYS: FULL SAIL!"

    @pytest.mark.asyncio
    async def test_slow_ship_needs_rowing(self):
        speed = Integer(value=5)
        fast = Compare(
            a=speed.output,
            b=20,
            comparison=Compare.Comparison.GREATER_THAN,
        )
        order = ConditionalSwitch(
            condition=fast.output,
            if_true="full sail",
            if_false="row harder",
        )
        announcement = FormatText(
            template="Captain says: {{ order }}!",
            order=order.output,
        )
        loud = ToUppercase(text=announcement.output)
        sink = Output(name="shout", value=loud.output)
        bag = await run_graph_async(create_graph(sink))
        assert bag["shout"] == "CAPTAIN SAYS: ROW HARDER!"


# ---------------------------------------------------------------------------
# 14-15  Alien Invasion Threat Level
# ---------------------------------------------------------------------------
class TestAlienInvasionThreatLevelWorkflow:
    @pytest.mark.asyncio
    async def test_safe_distant_single_ship(self):
        distance = Integer(value=9000)
        ships = Integer(value=1)
        too_close = Compare(
            a=distance.output,
            b=500,
            comparison=Compare.Comparison.LESS_THAN,
        )
        many_ships = Compare(
            a=ships.output,
            b=5,
            comparison=Compare.Comparison.GREATER_THAN_OR_EQUAL,
        )
        danger = LogicalOperator(
            a=too_close.output,
            b=many_ships.output,
            operation=LogicalOperator.BooleanOperation.OR,
        )
        alert = ConditionalSwitch(
            condition=danger.output,
            if_true="RED ALERT",
            if_false="all clear",
        )
        sink = Output(name="status", value=alert.output)
        bag = await run_graph_async(create_graph(sink))
        assert bag["status"] == "all clear"

    @pytest.mark.asyncio
    async def test_danger_close_armada(self):
        distance = Integer(value=100)
        ships = Integer(value=50)
        too_close = Compare(
            a=distance.output,
            b=500,
            comparison=Compare.Comparison.LESS_THAN,
        )
        many_ships = Compare(
            a=ships.output,
            b=5,
            comparison=Compare.Comparison.GREATER_THAN_OR_EQUAL,
        )
        danger = LogicalOperator(
            a=too_close.output,
            b=many_ships.output,
            operation=LogicalOperator.BooleanOperation.OR,
        )
        alert = ConditionalSwitch(
            condition=danger.output,
            if_true="RED ALERT",
            if_false="all clear",
        )
        sink = Output(name="status", value=alert.output)
        bag = await run_graph_async(create_graph(sink))
        assert bag["status"] == "RED ALERT"


# ---------------------------------------------------------------------------
# 16-17  Pizza Delivery ETA Classifier
# ---------------------------------------------------------------------------
class TestPizzaDeliveryETAClassifierWorkflow:
    @pytest.mark.asyncio
    async def test_fast_delivery(self):
        minutes = Integer(value=12)
        is_fast = Compare(
            a=minutes.output,
            b=15,
            comparison=Compare.Comparison.LESS_THAN_OR_EQUAL,
        )
        label = ConditionalSwitch(
            condition=is_fast.output,
            if_true="fast",
            if_false="not fast",
        )
        is_late = Compare(
            a=minutes.output,
            b=45,
            comparison=Compare.Comparison.GREATER_THAN,
        )
        final = ConditionalSwitch(
            condition=is_late.output,
            if_true="late",
            if_false=label.output,
        )
        sink = Output(name="eta_class", value=final.output)
        bag = await run_graph_async(create_graph(sink))
        assert bag["eta_class"] == "fast"

    @pytest.mark.asyncio
    async def test_late_delivery(self):
        minutes = Integer(value=60)
        is_fast = Compare(
            a=minutes.output,
            b=15,
            comparison=Compare.Comparison.LESS_THAN_OR_EQUAL,
        )
        label = ConditionalSwitch(
            condition=is_fast.output,
            if_true="fast",
            if_false="not fast",
        )
        is_late = Compare(
            a=minutes.output,
            b=45,
            comparison=Compare.Comparison.GREATER_THAN,
        )
        final = ConditionalSwitch(
            condition=is_late.output,
            if_true="late",
            if_false=label.output,
        )
        sink = Output(name="eta_class", value=final.output)
        bag = await run_graph_async(create_graph(sink))
        assert bag["eta_class"] == "late"


# ---------------------------------------------------------------------------
# 18-19  Gamer Rage-Quit Predictor
# ---------------------------------------------------------------------------
class TestGamerRageQuitPredictorWorkflow:
    @pytest.mark.asyncio
    async def test_rage_quit_imminent(self):
        deaths = Integer(value=15)
        score = Integer(value=50)
        too_many_deaths = Compare(
            a=deaths.output,
            b=10,
            comparison=Compare.Comparison.GREATER_THAN,
        )
        terrible_score = Compare(
            a=score.output,
            b=100,
            comparison=Compare.Comparison.LESS_THAN,
        )
        rage = LogicalOperator(
            a=too_many_deaths.output,
            b=terrible_score.output,
            operation=LogicalOperator.BooleanOperation.OR,
        )
        prediction = ConditionalSwitch(
            condition=rage.output,
            if_true="Rage quit in 3... 2... 1...",
            if_false="Still having fun!",
        )
        sink = Output(name="prediction", value=prediction.output)
        bag = await run_graph_async(create_graph(sink))
        assert bag["prediction"] == "Rage quit in 3... 2... 1..."

    @pytest.mark.asyncio
    async def test_calm_gamer(self):
        deaths = Integer(value=2)
        score = Integer(value=500)
        too_many_deaths = Compare(
            a=deaths.output,
            b=10,
            comparison=Compare.Comparison.GREATER_THAN,
        )
        terrible_score = Compare(
            a=score.output,
            b=100,
            comparison=Compare.Comparison.LESS_THAN,
        )
        rage = LogicalOperator(
            a=too_many_deaths.output,
            b=terrible_score.output,
            operation=LogicalOperator.BooleanOperation.OR,
        )
        prediction = ConditionalSwitch(
            condition=rage.output,
            if_true="Rage quit in 3... 2... 1...",
            if_false="Still having fun!",
        )
        sink = Output(name="prediction", value=prediction.output)
        bag = await run_graph_async(create_graph(sink))
        assert bag["prediction"] == "Still having fun!"


# ---------------------------------------------------------------------------
# 20  Schrödinger's Bool
# ---------------------------------------------------------------------------
class TestSchrodingersBoolWorkflow:
    @pytest.mark.asyncio
    async def test_not_of_true_and_false_is_true(self):
        t = Bool(value=True)
        f = Bool(value=False)
        both = LogicalOperator(
            a=t.output,
            b=f.output,
            operation=LogicalOperator.BooleanOperation.AND,
        )
        negated = BoolNot(value=both.output)
        sink = Output(name="result", value=negated.output)
        bag = await run_graph_async(create_graph(sink))
        # NOT(True AND False) == NOT(False) == True
        assert bag["result"] is True


# ---------------------------------------------------------------------------
# 21-22  Weather Dress Code Advisor
# ---------------------------------------------------------------------------
class TestWeatherDressCodeAdvisorWorkflow:
    @pytest.mark.asyncio
    async def test_cold_weather_bundle_up(self):
        temp = Float(value=5.0)
        is_cold = Compare(
            a=temp.output,
            b=10.0,
            comparison=Compare.Comparison.LESS_THAN,
        )
        advice = ConditionalSwitch(
            condition=is_cold.output,
            if_true="Wear a parka and two scarves!",
            if_false="Normal clothes are fine.",
        )
        sink = Output(name="advice", value=advice.output)
        bag = await run_graph_async(create_graph(sink))
        assert bag["advice"] == "Wear a parka and two scarves!"

    @pytest.mark.asyncio
    async def test_hot_weather_shorts(self):
        temp = Float(value=35.0)
        is_hot = Compare(
            a=temp.output,
            b=30.0,
            comparison=Compare.Comparison.GREATER_THAN,
        )
        advice = ConditionalSwitch(
            condition=is_hot.output,
            if_true="Shorts and sunscreen!",
            if_false="Normal clothes are fine.",
        )
        sink = Output(name="advice", value=advice.output)
        bag = await run_graph_async(create_graph(sink))
        assert bag["advice"] == "Shorts and sunscreen!"


# ---------------------------------------------------------------------------
# 23  Existential Crisis Calculator
# ---------------------------------------------------------------------------
class TestExistentialCrisisCalculatorWorkflow:
    @pytest.mark.asyncio
    async def test_crisis_mode_activated(self):
        age = Integer(value=35)
        has_plan = Bool(value=False)
        over_30 = Compare(
            a=age.output,
            b=30,
            comparison=Compare.Comparison.GREATER_THAN,
        )
        no_plan = BoolNot(value=has_plan.output)
        crisis = LogicalOperator(
            a=over_30.output,
            b=no_plan.output,
            operation=LogicalOperator.BooleanOperation.AND,
        )
        msg = ConditionalSwitch(
            condition=crisis.output,
            if_true="Existential crisis loading...",
            if_false="Everything is under control.",
        )
        sink = Output(name="status", value=msg.output)
        bag = await run_graph_async(create_graph(sink))
        assert bag["status"] == "Existential crisis loading..."


# ---------------------------------------------------------------------------
# 24-25  Ninja Turtle Pizza Rating
# ---------------------------------------------------------------------------
class TestNinjaTurtlePizzaRatingWorkflow:
    @pytest.mark.asyncio
    async def test_excellent_pizza(self):
        score = Integer(value=9)
        is_excellent = Compare(
            a=score.output,
            b=7,
            comparison=Compare.Comparison.GREATER_THAN_OR_EQUAL,
        )
        label = ConditionalSwitch(
            condition=is_excellent.output,
            if_true="excellent",
            if_false="meh",
        )
        review = FormatText(
            template="Cowabunga! This pizza is {{ rating }}!",
            rating=label.output,
        )
        sink = Output(name="review", value=review.output)
        bag = await run_graph_async(create_graph(sink))
        assert bag["review"] == "Cowabunga! This pizza is excellent!"

    @pytest.mark.asyncio
    async def test_meh_pizza(self):
        score = Integer(value=4)
        is_excellent = Compare(
            a=score.output,
            b=7,
            comparison=Compare.Comparison.GREATER_THAN_OR_EQUAL,
        )
        label = ConditionalSwitch(
            condition=is_excellent.output,
            if_true="excellent",
            if_false="meh",
        )
        review = FormatText(
            template="Cowabunga! This pizza is {{ rating }}!",
            rating=label.output,
        )
        sink = Output(name="review", value=review.output)
        bag = await run_graph_async(create_graph(sink))
        assert bag["review"] == "Cowabunga! This pizza is meh!"
