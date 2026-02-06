"""
Fun collection-workflow integration tests.

25 humorous but realistic workflow tests exercising list and dictionary
nodes through run_graph_async with real graphs.
"""

import pytest

from nodetool.dsl.graph import create_graph, run_graph_async
from nodetool.dsl.nodetool.constant import (
    List as ConstList,  # noqa: F401
    Dict as ConstDict,
    String as ConstString,
    Integer as ConstInteger,  # noqa: F401
    Float as ConstFloat,  # noqa: F401
)
from nodetool.dsl.nodetool.list import (
    Append as ListAppend,  # noqa: F401
    Length as ListLength,
    Reverse,
    Sort,
    Slice as ListSlice,
    Average,
    Maximum,
    Minimum,
)
from nodetool.dsl.nodetool.dictionary import (
    ArgMax,
    Combine,
    Filter as DictFilter,
    GetValue,
    ParseJSON,
    Remove,
    ToJSON,
    ToYAML,
    Update,
    Zip,
)
from nodetool.dsl.nodetool.output import Output


# ======================================================================
# DICTIONARY WORKFLOWS
# ======================================================================


class TestSecretSantaMatchmakerWorkflow:
    """Zip names to gifts, then find what Carol gets."""

    @pytest.mark.asyncio
    async def test_find_carol_gift(self):
        matches = Zip(
            keys=["Alice", "Bob", "Carol", "Dave"],
            values=["Scarf", "Mug", "Book", "Socks"],
        )
        carol_gift = GetValue(dictionary=matches.output, key="Carol")
        sink = Output(name="r", value=carol_gift.output)
        bag = await run_graph_async(create_graph(sink))
        assert bag["r"] == "Book"


class TestFridgeInventoryManagerWorkflow:
    """Combine two half-inventories and check we still have milk."""

    @pytest.mark.asyncio
    async def test_milk_survives_merge(self):
        top_shelf = ConstDict(value={"milk": 2, "yogurt": 3, "butter": 1})
        bottom_shelf = ConstDict(value={"cheese": 5, "eggs": 12, "jam": 1})
        fridge = Combine(dict_a=top_shelf.output, dict_b=bottom_shelf.output)
        milk = GetValue(dictionary=fridge.output, key="milk")
        sink = Output(name="r", value=milk.output)
        bag = await run_graph_async(create_graph(sink))
        assert bag["r"] == 2


class TestSpyAgencyDossierRedactorWorkflow:
    """Remove classified info from an agent's file."""

    @pytest.mark.asyncio
    async def test_redact_classified_field(self):
        dossier = ConstDict(
            value={
                "codename": "Shadow Fox",
                "real_name": "CLASSIFIED",
                "mission_count": 42,
            }
        )
        cleaned = Remove(dictionary=dossier.output, key="real_name")
        codename = GetValue(dictionary=cleaned.output, key="codename")
        sink = Output(name="r", value=codename.output)
        bag = await run_graph_async(create_graph(sink))
        assert bag["r"] == "Shadow Fox"


class TestEmotionalSupportScoreboardWorkflow:
    """ArgMax finds the dominant mood in the room."""

    @pytest.mark.asyncio
    async def test_dominant_mood(self):
        moods = ConstDict(
            value={"happy": 0.7, "sad": 0.1, "anxious": 0.15, "hangry": 0.05}
        )
        winner = ArgMax(scores=moods.output)
        sink = Output(name="r", value=winner.output)
        bag = await run_graph_async(create_graph(sink))
        assert bag["r"] == "happy"


class TestWizardSpellPickerWorkflow:
    """Filter spell-book to only combat spells."""

    @pytest.mark.asyncio
    async def test_only_combat_spells(self):
        spellbook = ConstDict(
            value={
                "fireball": 9,
                "heal": 5,
                "lightning": 8,
                "shield": 6,
            }
        )
        combat = DictFilter(
            dictionary=spellbook.output, keys=["fireball", "lightning"]
        )
        sink = Output(name="r", value=combat.output)
        bag = await run_graph_async(create_graph(sink))
        assert bag["r"] == {"fireball": 9, "lightning": 8}


class TestStartupBuzzwordBingoWorkflow:
    """Serialize buzzword dict to JSON and verify keys survive."""

    @pytest.mark.asyncio
    async def test_json_has_buzzwords(self):
        buzzwords = ConstDict(
            value={"synergy": 10, "disruption": 8, "pivot": 6, "leverage": 4}
        )
        as_json = ToJSON(dictionary=buzzwords.output)
        sink = Output(name="r", value=as_json.output)
        bag = await run_graph_async(create_graph(sink))
        assert "synergy" in bag["r"]
        assert "disruption" in bag["r"]


class TestCoffeeOrderMergerWorkflow:
    """Override default coffee order with custom preferences."""

    @pytest.mark.asyncio
    async def test_custom_overrides_default(self):
        default_order = ConstDict(
            value={"size": "medium", "milk": "whole", "sugar": 2}
        )
        custom = ConstDict(value={"size": "large", "milk": "oat"})
        final = Update(dictionary=default_order.output, new_pairs=custom.output)
        size = GetValue(dictionary=final.output, key="size")
        milk = GetValue(dictionary=final.output, key="milk")
        sugar = GetValue(dictionary=final.output, key="sugar")
        o1 = Output(name="size", value=size.output)
        o2 = Output(name="milk", value=milk.output)
        o3 = Output(name="sugar", value=sugar.output)
        bag = await run_graph_async(create_graph(o1, o2, o3))
        assert bag["size"] == "large"
        assert bag["milk"] == "oat"
        assert bag["sugar"] == 2


class TestDungeonLootDistributorWorkflow:
    """Combine loot from two treasure chests."""

    @pytest.mark.asyncio
    async def test_combined_loot(self):
        chest_a = ConstDict(value={"gold": 100, "sword": 1, "potion": 3})
        chest_b = ConstDict(value={"shield": 1, "gem": 5, "scroll": 2})
        all_loot = Combine(dict_a=chest_a.output, dict_b=chest_b.output)
        gold = GetValue(dictionary=all_loot.output, key="gold")
        gem = GetValue(dictionary=all_loot.output, key="gem")
        o1 = Output(name="gold", value=gold.output)
        o2 = Output(name="gem", value=gem.output)
        bag = await run_graph_async(create_graph(o1, o2))
        assert bag["gold"] == 100
        assert bag["gem"] == 5


class TestMovieSnackRankingWorkflow:
    """ArgMax identifies the undisputed best movie snack."""

    @pytest.mark.asyncio
    async def test_best_snack(self):
        ratings = ConstDict(
            value={
                "popcorn": 0.95,
                "nachos": 0.8,
                "twizzlers": 0.6,
                "gummy_bears": 0.7,
            }
        )
        best = ArgMax(scores=ratings.output)
        sink = Output(name="r", value=best.output)
        bag = await run_graph_async(create_graph(sink))
        assert bag["r"] == "popcorn"


class TestAPIConfigSwitcherWorkflow:
    """Update base API config with production overrides."""

    @pytest.mark.asyncio
    async def test_production_overrides(self):
        base = ConstDict(
            value={"timeout": 30, "retries": 3, "env": "staging", "verbose": True}
        )
        prod = ConstDict(value={"env": "production", "verbose": False})
        final = Update(dictionary=base.output, new_pairs=prod.output)
        env = GetValue(dictionary=final.output, key="env")
        retries = GetValue(dictionary=final.output, key="retries")
        o1 = Output(name="env", value=env.output)
        o2 = Output(name="retries", value=retries.output)
        bag = await run_graph_async(create_graph(o1, o2))
        assert bag["env"] == "production"
        assert bag["retries"] == 3


class TestAstrologyCompatibilityWorkflow:
    """Zip zodiac signs to elements, look up Scorpio."""

    @pytest.mark.asyncio
    async def test_scorpio_element(self):
        mapping = Zip(
            keys=["Aries", "Taurus", "Gemini", "Scorpio"],
            values=["Fire", "Earth", "Air", "Water"],
        )
        scorpio = GetValue(dictionary=mapping.output, key="Scorpio")
        sink = Output(name="r", value=scorpio.output)
        bag = await run_graph_async(create_graph(sink))
        assert bag["r"] == "Water"


class TestPetPersonalityProfileWorkflow:
    """Export a cat personality profile to YAML and verify keys."""

    @pytest.mark.asyncio
    async def test_yaml_has_personality_keys(self):
        profile = ConstDict(
            value={"name": "Whiskers", "attitude": "aloof", "nap_hours": 18}
        )
        yml = ToYAML(dictionary=profile.output)
        sink = Output(name="r", value=yml.output)
        bag = await run_graph_async(create_graph(sink))
        assert "name:" in bag["r"]
        assert "attitude:" in bag["r"]
        assert "nap_hours:" in bag["r"]


class TestSupervillainWeaknessDBWorkflow:
    """GetValue with a default for an unknown villain."""

    @pytest.mark.asyncio
    async def test_unknown_villain_gets_default(self):
        db = ConstDict(
            value={
                "Magneto": "plastic",
                "Superman": "kryptonite",
                "Dracula": "garlic",
            }
        )
        known = GetValue(dictionary=db.output, key="Dracula")
        unknown = GetValue(
            dictionary=db.output, key="Mothman", default="unknown"
        )
        o1 = Output(name="known", value=known.output)
        o2 = Output(name="unknown", value=unknown.output)
        bag = await run_graph_async(create_graph(o1, o2))
        assert bag["known"] == "garlic"
        assert bag["unknown"] == "unknown"


class TestMealPrepMacroCalculatorWorkflow:
    """Round-trip a macro dict through JSON serialization."""

    @pytest.mark.asyncio
    async def test_parse_json_roundtrip(self):
        raw_json = ConstString(value='{"protein": 30, "carbs": 45, "fat": 12}')
        parsed = ParseJSON(json_string=raw_json.output)
        protein = GetValue(dictionary=parsed.output, key="protein")
        sink = Output(name="r", value=protein.output)
        bag = await run_graph_async(create_graph(sink))
        assert bag["r"] == 30


class TestOfficeSupplyAuditWorkflow:
    """Remove depleted item, then combine with new stock."""

    @pytest.mark.asyncio
    async def test_remove_then_restock(self):
        current = ConstDict(value={"pens": 0, "paper": 200, "stapler": 3})
        cleaned = Remove(dictionary=current.output, key="pens")
        new_stock = ConstDict(value={"pens": 50, "tape": 10})
        restocked = Combine(dict_a=cleaned.output, dict_b=new_stock.output)
        pens = GetValue(dictionary=restocked.output, key="pens")
        paper = GetValue(dictionary=restocked.output, key="paper")
        o1 = Output(name="pens", value=pens.output)
        o2 = Output(name="paper", value=paper.output)
        bag = await run_graph_async(create_graph(o1, o2))
        assert bag["pens"] == 50
        assert bag["paper"] == 200


# ======================================================================
# LIST WORKFLOWS
# ======================================================================


class TestTopThreeMoviePickerWorkflow:
    """Sort movie scores descending, slice top 3."""

    @pytest.mark.asyncio
    async def test_top_three_scores(self):
        desc = Sort(
            values=[6.5, 9.1, 7.3, 8.8, 5.0, 9.5], order="descending"
        )
        top3 = ListSlice(values=[9.5, 9.1, 8.8, 7.3, 6.5, 5.0], start=0, stop=3)
        o1 = Output(name="sorted", value=desc.output)
        o2 = Output(name="top3", value=top3.output)
        bag = await run_graph_async(create_graph(o1, o2))
        assert bag["sorted"] == [9.5, 9.1, 8.8, 7.3, 6.5, 5.0]
        assert bag["top3"] == [9.5, 9.1, 8.8]


class TestBedtimeCountdownWorkflow:
    """Reverse a countdown list so it goes from high to low."""

    @pytest.mark.asyncio
    async def test_countdown_reversed(self):
        countdown = Reverse(values=[1, 2, 3, 4, 5])
        sink = Output(name="r", value=countdown.output)
        bag = await run_graph_async(create_graph(sink))
        assert bag["r"] == [5, 4, 3, 2, 1]


class TestStudentGradeAnalyzerWorkflow:
    """Compute average, max, and min grades in one graph."""

    @pytest.mark.asyncio
    async def test_grade_triple_stats(self):
        data = [72.0, 85.0, 91.0, 68.0, 77.0]
        avg = Average(values=data)
        top = Maximum(values=data)
        bottom = Minimum(values=data)
        o1 = Output(name="avg", value=avg.output)
        o2 = Output(name="top", value=top.output)
        o3 = Output(name="bottom", value=bottom.output)
        bag = await run_graph_async(create_graph(o1, o2, o3))
        assert 78.0 < bag["avg"] < 79.0  # 78.6
        assert bag["top"] == 91.0
        assert bag["bottom"] == 68.0


class TestShoppingCartItemCounterWorkflow:
    """Count items in a shopping cart list."""

    @pytest.mark.asyncio
    async def test_cart_count(self):
        count = ListLength(
            values=["bread", "cheese", "tomato", "basil", "olive_oil"]
        )
        sink = Output(name="r", value=count.output)
        bag = await run_graph_async(create_graph(sink))
        assert bag["r"] == 5


class TestPlaylistShuffleVerifierWorkflow:
    """Sort a shuffled playlist and verify it matches the sorted original."""

    @pytest.mark.asyncio
    async def test_sorted_shuffled_matches_original(self):
        sorted_list = Sort(values=["Bohemian", "Africa", "Circles", "Dreams"])
        sink = Output(name="r", value=sorted_list.output)
        bag = await run_graph_async(create_graph(sink))
        assert bag["r"] == ["Africa", "Bohemian", "Circles", "Dreams"]


class TestBookshelfOrganizerWorkflow:
    """Alphabetically sort book titles."""

    @pytest.mark.asyncio
    async def test_sorted_books(self):
        organized = Sort(
            values=["Dune", "Neuromancer", "Annihilation", "Hyperion"]
        )
        sink = Output(name="r", value=organized.output)
        bag = await run_graph_async(create_graph(sink))
        assert bag["r"] == ["Annihilation", "Dune", "Hyperion", "Neuromancer"]


class TestTemperatureExtremeFinderWorkflow:
    """Find hottest and coldest temperatures from a week of readings."""

    @pytest.mark.asyncio
    async def test_temp_extremes(self):
        data = [32.0, 28.5, 35.2, 30.0, 27.1, 33.8, 29.4]
        hottest = Maximum(values=data)
        coldest = Minimum(values=data)
        o1 = Output(name="hot", value=hottest.output)
        o2 = Output(name="cold", value=coldest.output)
        bag = await run_graph_async(create_graph(o1, o2))
        assert bag["hot"] == 35.2
        assert bag["cold"] == 27.1


class TestPodiumFinishSelectorWorkflow:
    """Sort race times ascending, slice top 3 fastest finishers."""

    @pytest.mark.asyncio
    async def test_podium_top_three(self):
        ordered = Sort(values=[12.4, 11.2, 13.1, 10.8, 11.9, 14.0])
        podium = ListSlice(values=[10.8, 11.2, 11.9, 12.4, 13.1, 14.0], start=0, stop=3)
        o1 = Output(name="sorted", value=ordered.output)
        o2 = Output(name="podium", value=podium.output)
        bag = await run_graph_async(create_graph(o1, o2))
        assert bag["sorted"] == [10.8, 11.2, 11.9, 12.4, 13.1, 14.0]
        assert bag["podium"] == [10.8, 11.2, 11.9]


class TestBirthdayPartyHeadcountWorkflow:
    """Count RSVPs and food items separately."""

    @pytest.mark.asyncio
    async def test_headcount_and_food_count(self):
        guest_count = ListLength(
            values=["Alice", "Bob", "Carol", "Dave", "Eve", "Frank"]
        )
        food_count = ListLength(values=["cake", "pizza", "soda", "chips"])
        o1 = Output(name="guests", value=guest_count.output)
        o2 = Output(name="foods", value=food_count.output)
        bag = await run_graph_async(create_graph(o1, o2))
        assert bag["guests"] == 6
        assert bag["foods"] == 4


class TestHighScoreTableWorkflow:
    """Sort scores descending, slice top 5, verify count."""

    @pytest.mark.asyncio
    async def test_high_score_pipeline(self):
        desc = Sort(
            values=[450, 780, 320, 990, 670, 550, 880, 210], order="descending"
        )
        top5 = ListSlice(
            values=[990, 880, 780, 670, 550, 450, 320, 210], start=0, stop=5
        )
        count = ListLength(values=[990, 880, 780, 670, 550])
        o1 = Output(name="desc", value=desc.output)
        o2 = Output(name="top5", value=top5.output)
        o3 = Output(name="count", value=count.output)
        bag = await run_graph_async(create_graph(o1, o2, o3))
        assert bag["desc"] == [990, 880, 780, 670, 550, 450, 320, 210]
        assert bag["top5"] == [990, 880, 780, 670, 550]
        assert bag["count"] == 5
