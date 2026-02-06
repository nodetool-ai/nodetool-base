"""
Fun data workflow integration tests.

Each test builds a humorous but realistic data-processing pipeline using the
DSL graph system and run_graph_async: CSV ingest → transform → assert.
"""

import pytest

from nodetool.dsl.graph import create_graph, run_graph_async
from nodetool.dsl.nodetool.constant import String as ConstString
from nodetool.dsl.nodetool.data import (
    Aggregate,
    Append as DataAppend,
    DropDuplicates,
    DropNA,
    ExtractColumn,
    FillNA,
    Filter as DataFilter,
    FindRow,
    ImportCSV,
    Merge,
    Rename,
    SelectColumn,
    SortByColumn,
    ToList,
)
from nodetool.dsl.nodetool.output import Output


# ---------------------------------------------------------------------------
# Inline CSV data
# ---------------------------------------------------------------------------

CAFFEINE_CSV = (
    "employee,coffees_per_day\n"
    "Alice,2\n"
    "Bob,5\n"
    "Carol,7\n"
    "Dave,1\n"
    "Eve,6\n"
)

CAT_EMPLOYEE_CSV = (
    "cat_name,department,naps_per_day\n"
    "Whiskers,Engineering,12\n"
    "Mittens,Sales,8\n"
    "Shadow,Engineering,15\n"
    "Socks,Marketing,10\n"
)

ZOMBIE_INVENTORY_CSV = (
    "item,quantity,category\n"
    "Baseball Bat,5,Weapons\n"
    "Canned Beans,20,Food\n"
    "Shotgun,2,Weapons\n"
    "Water Bottles,30,Food\n"
    "Bandages,15,Medical\n"
)

PIRATE_PAYROLL_CSV = (
    "pirate,gold_coins\n"
    "Blackbeard,5000\n"
    "Anne Bonny,4200\n"
    "Calico Jack,3100\n"
)

UNICORN_STABLE_CSV = (
    "unicorn,sparkle_level,stable\n"
    "Stardust,9,North\n"
    "Stardust,9,North\n"
    "Rainbow,7,South\n"
    "Glimmer,8,East\n"
)

ALIEN_LOG_CSV = (
    "date,alien_type,humans_taken\n"
    "2024-01-10,Grey,3\n"
    "2024-02-14,Reptilian,1\n"
    "2024-03-20,Grey,5\n"
    "2024-04-01,Nordic,0\n"
)

DOG_PARK_CSV = (
    "park,dogs,rating\n"
    "Barkington,45,4.8\n"
    "Woofside,30,4.2\n"
    "Pawville,60,4.9\n"
    "Fetchmore,25,3.7\n"
)

MEME_CSV = (
    "meme_id,likes,shares\n"
    "M001,15000,3200\n"
    "M002,420,69\n"
    "M003,99999,12000\n"
)

NINJA_BUDGET_CSV = (
    "dept,weapon,cost\n"
    "Stealth,Shuriken,500\n"
    "Stealth,Smoke Bomb,300\n"
    "Combat,Katana,2000\n"
    "Combat,Nunchucks,800\n"
)

HAUNTED_HOUSE_CSV = (
    "room,ghosts,scary_level\n"
    "Attic,3,9\n"
    "Basement,5,10\n"
    "Kitchen,1,4\n"
    "Library,2,7\n"
)

SPACESHIP_LEFT_CSV = (
    "ship,fuel_a\n"
    "Enterprise,500\n"
    "Falcon,300\n"
)

SPACESHIP_RIGHT_CSV = (
    "fuel_b\n"
    "200\n"
    "150\n"
)

WIZARD_SPELLBOOK_CSV = (
    "spell,element,power_level\n"
    "Fireball,Fire,85\n"
    "Ice Storm,Ice,70\n"
    "Lightning Bolt,Electric,90\n"
    "Healing Light,Holy,60\n"
)

ROBOT_QA_CSV = (
    "robot_id,defects,model\n"
    "R-001,0,Alpha\n"
    "R-002,3,Beta\n"
    "R-003,0,Alpha\n"
    "R-004,1,Gamma\n"
)

TACO_CSV = (
    "taco_type,price,quantity\n"
    "Carnitas,3.50,10\n"
    "Al Pastor,4.00,8\n"
    "Barbacoa,4.50,6\n"
    "Fish,5.00,4\n"
)

BABY_NAMES_A_CSV = (
    "name,votes_mom,votes_dad\n"
    "Luna,8,3\n"
    "Orion,5,9\n"
)

BABY_NAMES_B_CSV = (
    "name,votes_mom,votes_dad\n"
    "Willow,7,6\n"
    "Felix,4,8\n"
)

DRAGON_FEEDING_CSV = (
    "dragon,food_type,kg_per_day\n"
    "Smaug,Sheep,\n"
    "Toothless,Fish,50\n"
    "Drogon,Cattle,\n"
)

TIME_TRAVEL_CSV = (
    "year,event,importance\n"
    "1969,Moon Landing,10\n"
    "2000,,\n"
    "2042,Mars Colony,9\n"
)

PENGUIN_CSV = (
    "colony,adults,chicks\n"
    "Ross Island,120,45\n"
    "Ross Island,130,50\n"
    "Snow Hill,80,30\n"
    "Snow Hill,90,35\n"
)

GALAXY_BRAIN_CSV = (
    "topic,duration_min,speaker\n"
    "Synergy,45,Chad\n"
    "Blockchain,90,Karen\n"
    "AI Ethics,30,Dev\n"
    "Lunch Plans,120,Everyone\n"
)

SOCK_DRAWER_CSV = (
    "color,pattern,pairs\n"
    "Black,Solid,5\n"
    "White,Striped,3\n"
    "Blue,Polka Dot,2\n"
    "Black,Striped,4\n"
)

PLANT_CSV = (
    "plant,water_ml,last_watered\n"
    "Monstera,200,2024-01-15\n"
    "Fern,150,2024-01-10\n"
    "Cactus,50,2024-01-01\n"
)

SUPERHERO_CSV = (
    "hero,power_cost,gadget_cost\n"
    "Batman,0,5000000\n"
    "Superman,100,0\n"
    "Iron Man,200,9000000\n"
)

VIBE_CSV = (
    "quarter,vibes,productivity\n"
    "Q1,80,70\n"
    "Q1,90,75\n"
    "Q2,60,50\n"
    "Q2,70,65\n"
)

COFFEE_SHOP_CSV = (
    "barista,orders,avg_time\n"
    "Jake,42,3.5\n"
    "Rosa,55,2.8\n"
    "Terry,38,4.1\n"
)

PROCRASTINATION_CSV = (
    "person,tasks_delayed,excuses_used\n"
    "Alice,12,8\n"
    "Bob,3,2\n"
    "Carol,20,15\n"
    "Dave,7,5\n"
)


# ---------------------------------------------------------------------------
# 1. Caffeine Addiction Tracker — filter heavy drinkers
# ---------------------------------------------------------------------------
class TestCaffeineAddictionTrackerWorkflow:
    @pytest.mark.asyncio
    async def test_filter_caffeine_addicts(self):
        csv = ConstString(value=CAFFEINE_CSV)
        df = ImportCSV(csv_data=csv.output)
        addicts = DataFilter(df=df.output, condition="coffees_per_day > 4")
        rows = ToList(dataframe=addicts.output)
        sink = Output(name="r", value=rows.output)
        bag = await run_graph_async(create_graph(sink))
        assert len(bag["r"]) == 3
        names = {row["employee"] for row in bag["r"]}
        assert names == {"Bob", "Carol", "Eve"}


# ---------------------------------------------------------------------------
# 2. Cat Employee Roster — sort by naps per day
# ---------------------------------------------------------------------------
class TestCatEmployeeRosterWorkflow:
    @pytest.mark.asyncio
    async def test_sort_cats_by_naps(self):
        csv = ConstString(value=CAT_EMPLOYEE_CSV)
        df = ImportCSV(csv_data=csv.output)
        sorted_df = SortByColumn(df=df.output, column="naps_per_day")
        rows = ToList(dataframe=sorted_df.output)
        sink = Output(name="r", value=rows.output)
        bag = await run_graph_async(create_graph(sink))
        naps = [row["naps_per_day"] for row in bag["r"]]
        assert naps == sorted(naps)
        assert bag["r"][0]["cat_name"] == "Mittens"


# ---------------------------------------------------------------------------
# 3. Zombie Survival Inventory — aggregate sum by category
# ---------------------------------------------------------------------------
class TestZombieSurvivalInventoryWorkflow:
    @pytest.mark.asyncio
    async def test_aggregate_supplies_by_category(self):
        csv = ConstString(value=ZOMBIE_INVENTORY_CSV)
        df = ImportCSV(csv_data=csv.output)
        agg = Aggregate(dataframe=df.output, columns="category", aggregation="sum")
        rows = ToList(dataframe=agg.output)
        sink = Output(name="r", value=rows.output)
        bag = await run_graph_async(create_graph(sink))
        by_cat = {row["category"]: row["quantity"] for row in bag["r"]}
        assert by_cat["Food"] == 50
        assert by_cat["Weapons"] == 7
        assert by_cat["Medical"] == 15


# ---------------------------------------------------------------------------
# 4. Pirate Payroll Processor — rename columns to fancy names
# ---------------------------------------------------------------------------
class TestPiratePayrollProcessorWorkflow:
    @pytest.mark.asyncio
    async def test_rename_pirate_columns(self):
        csv = ConstString(value=PIRATE_PAYROLL_CSV)
        df = ImportCSV(csv_data=csv.output)
        renamed = Rename(
            dataframe=df.output,
            rename_map="pirate:Captain Name,gold_coins:Doubloons Earned",
        )
        rows = ToList(dataframe=renamed.output)
        sink = Output(name="r", value=rows.output)
        bag = await run_graph_async(create_graph(sink))
        assert "Captain Name" in bag["r"][0]
        assert "Doubloons Earned" in bag["r"][0]
        assert bag["r"][0]["Captain Name"] == "Blackbeard"


# ---------------------------------------------------------------------------
# 5. Unicorn Stable Cleanup — drop duplicates
# ---------------------------------------------------------------------------
class TestUnicornStableCleanupWorkflow:
    @pytest.mark.asyncio
    async def test_remove_duplicate_unicorns(self):
        csv = ConstString(value=UNICORN_STABLE_CSV)
        df = ImportCSV(csv_data=csv.output)
        deduped = DropDuplicates(df=df.output)
        rows = ToList(dataframe=deduped.output)
        sink = Output(name="r", value=rows.output)
        bag = await run_graph_async(create_graph(sink))
        assert len(bag["r"]) == 3
        names = {row["unicorn"] for row in bag["r"]}
        assert names == {"Stardust", "Rainbow", "Glimmer"}


# ---------------------------------------------------------------------------
# 6. Alien Abduction Log — filter by alien type
# ---------------------------------------------------------------------------
class TestAlienAbductionLogWorkflow:
    @pytest.mark.asyncio
    async def test_filter_grey_aliens(self):
        csv = ConstString(value=ALIEN_LOG_CSV)
        df = ImportCSV(csv_data=csv.output)
        greys = DataFilter(df=df.output, condition="alien_type == 'Grey'")
        rows = ToList(dataframe=greys.output)
        sink = Output(name="r", value=rows.output)
        bag = await run_graph_async(create_graph(sink))
        assert len(bag["r"]) == 2
        total = sum(row["humans_taken"] for row in bag["r"])
        assert total == 8


# ---------------------------------------------------------------------------
# 7. Dog Park Popularity Ranking — sort by rating
# ---------------------------------------------------------------------------
class TestDogParkPopularityRankingWorkflow:
    @pytest.mark.asyncio
    async def test_rank_parks_by_rating(self):
        csv = ConstString(value=DOG_PARK_CSV)
        df = ImportCSV(csv_data=csv.output)
        sorted_df = SortByColumn(df=df.output, column="rating")
        rows = ToList(dataframe=sorted_df.output)
        sink = Output(name="r", value=rows.output)
        bag = await run_graph_async(create_graph(sink))
        ratings = [row["rating"] for row in bag["r"]]
        assert ratings == sorted(ratings)
        assert bag["r"][-1]["park"] == "Pawville"


# ---------------------------------------------------------------------------
# 8. Meme Quality Auditor — extract likes column
# ---------------------------------------------------------------------------
class TestMemeQualityAuditorWorkflow:
    @pytest.mark.asyncio
    async def test_extract_likes(self):
        csv = ConstString(value=MEME_CSV)
        df = ImportCSV(csv_data=csv.output)
        likes = ExtractColumn(dataframe=df.output, column_name="likes")
        sink = Output(name="r", value=likes.output)
        bag = await run_graph_async(create_graph(sink))
        assert len(bag["r"]) == 3
        assert 15000 in bag["r"]
        assert 420 in bag["r"]
        assert 99999 in bag["r"]


# ---------------------------------------------------------------------------
# 9. Ninja Budget Report — aggregate sum by department
# ---------------------------------------------------------------------------
class TestNinjaBudgetReportWorkflow:
    @pytest.mark.asyncio
    async def test_total_cost_by_dept(self):
        csv = ConstString(value=NINJA_BUDGET_CSV)
        df = ImportCSV(csv_data=csv.output)
        agg = Aggregate(dataframe=df.output, columns="dept", aggregation="sum")
        rows = ToList(dataframe=agg.output)
        sink = Output(name="r", value=rows.output)
        bag = await run_graph_async(create_graph(sink))
        by_dept = {row["dept"]: row["cost"] for row in bag["r"]}
        assert by_dept["Stealth"] == 800
        assert by_dept["Combat"] == 2800


# ---------------------------------------------------------------------------
# 10. Haunted House Inspector — find a specific room
# ---------------------------------------------------------------------------
class TestHauntedHouseInspectorWorkflow:
    @pytest.mark.asyncio
    async def test_find_basement(self):
        csv = ConstString(value=HAUNTED_HOUSE_CSV)
        df = ImportCSV(csv_data=csv.output)
        found = FindRow(df=df.output, condition="room == 'Basement'")
        rows = ToList(dataframe=found.output)
        sink = Output(name="r", value=rows.output)
        bag = await run_graph_async(create_graph(sink))
        assert len(bag["r"]) == 1
        assert bag["r"][0]["ghosts"] == 5
        assert bag["r"][0]["scary_level"] == 10


# ---------------------------------------------------------------------------
# 11. Spaceship Fuel Calculator — merge two CSVs side by side
# ---------------------------------------------------------------------------
class TestSpaceshipFuelCalculatorWorkflow:
    @pytest.mark.asyncio
    async def test_merge_fuel_sources(self):
        csv_l = ConstString(value=SPACESHIP_LEFT_CSV)
        csv_r = ConstString(value=SPACESHIP_RIGHT_CSV)
        df_l = ImportCSV(csv_data=csv_l.output)
        df_r = ImportCSV(csv_data=csv_r.output)
        merged = Merge(dataframe_a=df_l.output, dataframe_b=df_r.output)
        rows = ToList(dataframe=merged.output)
        sink = Output(name="r", value=rows.output)
        bag = await run_graph_async(create_graph(sink))
        assert len(bag["r"]) == 2
        assert bag["r"][0]["ship"] == "Enterprise"
        assert bag["r"][0]["fuel_a"] == 500
        assert bag["r"][0]["fuel_b"] == 200


# ---------------------------------------------------------------------------
# 12. Wizard Spellbook Index — select two columns
# ---------------------------------------------------------------------------
class TestWizardSpellbookIndexWorkflow:
    @pytest.mark.asyncio
    async def test_select_spell_and_element(self):
        csv = ConstString(value=WIZARD_SPELLBOOK_CSV)
        df = ImportCSV(csv_data=csv.output)
        selected = SelectColumn(dataframe=df.output, columns="spell,element")
        rows = ToList(dataframe=selected.output)
        sink = Output(name="r", value=rows.output)
        bag = await run_graph_async(create_graph(sink))
        assert set(bag["r"][0].keys()) == {"spell", "element"}
        assert bag["r"][0]["spell"] == "Fireball"


# ---------------------------------------------------------------------------
# 13. Robot Factory QA — filter for defects > 0
# ---------------------------------------------------------------------------
class TestRobotFactoryQAWorkflow:
    @pytest.mark.asyncio
    async def test_find_defective_robots(self):
        csv = ConstString(value=ROBOT_QA_CSV)
        df = ImportCSV(csv_data=csv.output)
        defective = DataFilter(df=df.output, condition="defects > 0")
        rows = ToList(dataframe=defective.output)
        sink = Output(name="r", value=rows.output)
        bag = await run_graph_async(create_graph(sink))
        assert len(bag["r"]) == 2
        ids = {row["robot_id"] for row in bag["r"]}
        assert ids == {"R-002", "R-004"}


# ---------------------------------------------------------------------------
# 14. Taco Tuesday Budget — rename + sort by price
# ---------------------------------------------------------------------------
class TestTacoTuesdayBudgetWorkflow:
    @pytest.mark.asyncio
    async def test_rename_and_sort_tacos(self):
        csv = ConstString(value=TACO_CSV)
        df = ImportCSV(csv_data=csv.output)
        renamed = Rename(
            dataframe=df.output,
            rename_map="taco_type:Taco,price:Unit Price",
        )
        sorted_df = SortByColumn(df=renamed.output, column="Unit Price")
        rows = ToList(dataframe=sorted_df.output)
        sink = Output(name="r", value=rows.output)
        bag = await run_graph_async(create_graph(sink))
        prices = [row["Unit Price"] for row in bag["r"]]
        assert prices == sorted(prices)
        assert bag["r"][0]["Taco"] == "Carnitas"


# ---------------------------------------------------------------------------
# 15. Baby Name Debate Settler — append two CSVs vertically
# ---------------------------------------------------------------------------
class TestBabyNameDebateSettlerWorkflow:
    @pytest.mark.asyncio
    async def test_combine_name_lists(self):
        csv_a = ConstString(value=BABY_NAMES_A_CSV)
        csv_b = ConstString(value=BABY_NAMES_B_CSV)
        df_a = ImportCSV(csv_data=csv_a.output)
        df_b = ImportCSV(csv_data=csv_b.output)
        stacked = DataAppend(dataframe_a=df_a.output, dataframe_b=df_b.output)
        rows = ToList(dataframe=stacked.output)
        sink = Output(name="r", value=rows.output)
        bag = await run_graph_async(create_graph(sink))
        assert len(bag["r"]) == 4
        names = {row["name"] for row in bag["r"]}
        assert names == {"Luna", "Orion", "Willow", "Felix"}


# ---------------------------------------------------------------------------
# 16. Dragon Feeding Schedule — fill missing values
# ---------------------------------------------------------------------------
class TestDragonFeedingScheduleWorkflow:
    @pytest.mark.asyncio
    async def test_fill_missing_kg(self):
        csv = ConstString(value=DRAGON_FEEDING_CSV)
        df = ImportCSV(csv_data=csv.output)
        filled = FillNA(
            dataframe=df.output, value=100.0, method="value", columns="kg_per_day"
        )
        rows = ToList(dataframe=filled.output)
        sink = Output(name="r", value=rows.output)
        bag = await run_graph_async(create_graph(sink))
        kg_vals = [row["kg_per_day"] for row in bag["r"]]
        assert kg_vals == [100.0, 50.0, 100.0]


# ---------------------------------------------------------------------------
# 17. Time Travel 404 Fixer — drop NA rows
# ---------------------------------------------------------------------------
class TestTimeTravel404FixerWorkflow:
    @pytest.mark.asyncio
    async def test_drop_broken_timeline_entries(self):
        csv = ConstString(value=TIME_TRAVEL_CSV)
        df = ImportCSV(csv_data=csv.output)
        clean = DropNA(df=df.output)
        rows = ToList(dataframe=clean.output)
        sink = Output(name="r", value=rows.output)
        bag = await run_graph_async(create_graph(sink))
        assert len(bag["r"]) == 2
        years = {row["year"] for row in bag["r"]}
        assert years == {1969, 2042}


# ---------------------------------------------------------------------------
# 18. Penguin Census — aggregate count by colony
# ---------------------------------------------------------------------------
class TestPenguinCensusWorkflow:
    @pytest.mark.asyncio
    async def test_count_observations_per_colony(self):
        csv = ConstString(value=PENGUIN_CSV)
        df = ImportCSV(csv_data=csv.output)
        agg = Aggregate(dataframe=df.output, columns="colony", aggregation="count")
        rows = ToList(dataframe=agg.output)
        sink = Output(name="r", value=rows.output)
        bag = await run_graph_async(create_graph(sink))
        by_colony = {row["colony"]: row["adults"] for row in bag["r"]}
        assert by_colony["Ross Island"] == 2
        assert by_colony["Snow Hill"] == 2


# ---------------------------------------------------------------------------
# 19. Galaxy Brain Meeting Notes — sort by duration
# ---------------------------------------------------------------------------
class TestGalaxyBrainMeetingNotesWorkflow:
    @pytest.mark.asyncio
    async def test_sort_meetings_by_duration(self):
        csv = ConstString(value=GALAXY_BRAIN_CSV)
        df = ImportCSV(csv_data=csv.output)
        sorted_df = SortByColumn(df=df.output, column="duration_min")
        rows = ToList(dataframe=sorted_df.output)
        sink = Output(name="r", value=rows.output)
        bag = await run_graph_async(create_graph(sink))
        durations = [row["duration_min"] for row in bag["r"]]
        assert durations == sorted(durations)
        assert bag["r"][-1]["topic"] == "Lunch Plans"


# ---------------------------------------------------------------------------
# 20. Sock Drawer Organizer — filter by specific pattern
# ---------------------------------------------------------------------------
class TestSockDrawerOrganizerWorkflow:
    @pytest.mark.asyncio
    async def test_filter_striped_socks(self):
        csv = ConstString(value=SOCK_DRAWER_CSV)
        df = ImportCSV(csv_data=csv.output)
        striped = DataFilter(df=df.output, condition="pattern == 'Striped'")
        rows = ToList(dataframe=striped.output)
        sink = Output(name="r", value=rows.output)
        bag = await run_graph_async(create_graph(sink))
        assert len(bag["r"]) == 2
        colors = {row["color"] for row in bag["r"]}
        assert colors == {"White", "Black"}


# ---------------------------------------------------------------------------
# 21. Plant Parent Tracker — find specific plant
# ---------------------------------------------------------------------------
class TestPlantParentTrackerWorkflow:
    @pytest.mark.asyncio
    async def test_find_cactus(self):
        csv = ConstString(value=PLANT_CSV)
        df = ImportCSV(csv_data=csv.output)
        found = FindRow(df=df.output, condition="plant == 'Cactus'")
        rows = ToList(dataframe=found.output)
        sink = Output(name="r", value=rows.output)
        bag = await run_graph_async(create_graph(sink))
        assert len(bag["r"]) == 1
        assert bag["r"][0]["water_ml"] == 50


# ---------------------------------------------------------------------------
# 22. Superhero Budget Cuts — extract column + verify
# ---------------------------------------------------------------------------
class TestSuperheroBudgetCutsWorkflow:
    @pytest.mark.asyncio
    async def test_extract_gadget_costs(self):
        csv = ConstString(value=SUPERHERO_CSV)
        df = ImportCSV(csv_data=csv.output)
        costs = ExtractColumn(dataframe=df.output, column_name="gadget_cost")
        sink = Output(name="r", value=costs.output)
        bag = await run_graph_async(create_graph(sink))
        assert len(bag["r"]) == 3
        assert 5000000 in bag["r"]
        assert 9000000 in bag["r"]
        assert 0 in bag["r"]


# ---------------------------------------------------------------------------
# 23. Vibe Check Analytics — aggregate mean
# ---------------------------------------------------------------------------
class TestVibeCheckAnalyticsWorkflow:
    @pytest.mark.asyncio
    async def test_average_vibes_by_quarter(self):
        csv = ConstString(value=VIBE_CSV)
        df = ImportCSV(csv_data=csv.output)
        agg = Aggregate(dataframe=df.output, columns="quarter", aggregation="mean")
        rows = ToList(dataframe=agg.output)
        sink = Output(name="r", value=rows.output)
        bag = await run_graph_async(create_graph(sink))
        by_q = {row["quarter"]: row["vibes"] for row in bag["r"]}
        assert by_q["Q1"] == 85.0
        assert by_q["Q2"] == 65.0


# ---------------------------------------------------------------------------
# 24. Coffee Shop Queue Optimizer — select columns
# ---------------------------------------------------------------------------
class TestCoffeeShopQueueOptimizerWorkflow:
    @pytest.mark.asyncio
    async def test_select_barista_and_orders(self):
        csv = ConstString(value=COFFEE_SHOP_CSV)
        df = ImportCSV(csv_data=csv.output)
        selected = SelectColumn(dataframe=df.output, columns="barista,orders")
        rows = ToList(dataframe=selected.output)
        sink = Output(name="r", value=rows.output)
        bag = await run_graph_async(create_graph(sink))
        assert set(bag["r"][0].keys()) == {"barista", "orders"}
        assert bag["r"][1]["barista"] == "Rosa"
        assert bag["r"][1]["orders"] == 55


# ---------------------------------------------------------------------------
# 25. Procrastination Leaderboard — sort + rename
# ---------------------------------------------------------------------------
class TestProcrastinationLeaderboardWorkflow:
    @pytest.mark.asyncio
    async def test_sort_and_rename_procrastinators(self):
        csv = ConstString(value=PROCRASTINATION_CSV)
        df = ImportCSV(csv_data=csv.output)
        sorted_df = SortByColumn(df=df.output, column="tasks_delayed")
        renamed = Rename(
            dataframe=sorted_df.output,
            rename_map="person:Champion,tasks_delayed:Delays,excuses_used:Creative Excuses",
        )
        rows = ToList(dataframe=renamed.output)
        sink = Output(name="r", value=rows.output)
        bag = await run_graph_async(create_graph(sink))
        assert "Champion" in bag["r"][0]
        assert "Delays" in bag["r"][0]
        assert "Creative Excuses" in bag["r"][0]
        delays = [row["Delays"] for row in bag["r"]]
        assert delays == sorted(delays)
        assert bag["r"][-1]["Champion"] == "Carol"
