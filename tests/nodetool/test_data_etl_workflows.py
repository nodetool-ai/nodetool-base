"""
Data ETL (Extract-Transform-Load) workflow tests.

Each test builds a realistic data-processing pipeline using the DSL
graph system and run_graph_async: CSV ingest → column operations →
filtering/sorting → aggregation → export.
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
# Helper: inline CSV data
# ---------------------------------------------------------------------------
EMPLOYEE_CSV = (
    "emp_name,department,salary\n"
    "Alice,Engineering,120000\n"
    "Bob,Sales,85000\n"
    "Carol,Engineering,115000\n"
    "Dave,Marketing,78000\n"
    "Eve,Engineering,130000\n"
)

SENSOR_CSV = (
    "sensor,reading\n"
    "A,22.5\n"
    "B,\n"
    "C,19.3\n"
)

SALES_CSV = (
    "region,revenue\n"
    "North,1000\n"
    "North,1500\n"
    "South,800\n"
    "South,900\n"
)

PEOPLE_CSV = (
    "pid,name,amount\n"
    "1,Alice,0\n"
    "2,Bob,0\n"
    "3,Carol,0\n"
)

ORDERS_CSV = (
    "pid,name,amount\n"
    "1,,250\n"
    "2,,180\n"
    "3,,310\n"
)

DEDUP_CSV = (
    "sensor,reading\n"
    "A,10.0\n"
    "A,10.0\n"
    "B,20.0\n"
)

TICKET_CSV = (
    "ticket,priority\n"
    "T-100,low\n"
    "T-200,high\n"
    "T-300,medium\n"
)


# ---------------------------------------------------------------------------
# Scenario: Employee roster — import CSV, rename, filter, sort
# ---------------------------------------------------------------------------
class TestEmployeeRoster:
    @pytest.mark.asyncio
    async def test_csv_ingest_to_list(self):
        csv = ConstString(value=EMPLOYEE_CSV)
        df = ImportCSV(csv_data=csv.output)
        rows = ToList(dataframe=df.output)
        sink = Output(name="r", value=rows.output)
        bag = await run_graph_async(create_graph(sink))
        assert len(bag["r"]) == 5
        assert bag["r"][0]["emp_name"] == "Alice"

    @pytest.mark.asyncio
    async def test_rename_columns(self):
        csv = ConstString(value=EMPLOYEE_CSV)
        df = ImportCSV(csv_data=csv.output)
        renamed = Rename(dataframe=df.output, rename_map="emp_name:name,department:dept")
        rows = ToList(dataframe=renamed.output)
        sink = Output(name="r", value=rows.output)
        bag = await run_graph_async(create_graph(sink))
        assert "name" in bag["r"][0]
        assert "dept" in bag["r"][0]

    @pytest.mark.asyncio
    async def test_filter_engineering(self):
        csv = ConstString(value=EMPLOYEE_CSV)
        df = ImportCSV(csv_data=csv.output)
        eng = DataFilter(df=df.output, condition="department == 'Engineering'")
        rows = ToList(dataframe=eng.output)
        sink = Output(name="r", value=rows.output)
        bag = await run_graph_async(create_graph(sink))
        assert len(bag["r"]) == 3
        names = {row["emp_name"] for row in bag["r"]}
        assert names == {"Alice", "Carol", "Eve"}

    @pytest.mark.asyncio
    async def test_sort_by_salary(self):
        csv = ConstString(value=EMPLOYEE_CSV)
        df = ImportCSV(csv_data=csv.output)
        sorted_df = SortByColumn(df=df.output, column="salary")
        rows = ToList(dataframe=sorted_df.output)
        sink = Output(name="r", value=rows.output)
        bag = await run_graph_async(create_graph(sink))
        salaries = [row["salary"] for row in bag["r"]]
        assert salaries == sorted(salaries)

    @pytest.mark.asyncio
    async def test_extract_salary_column(self):
        csv = ConstString(value=EMPLOYEE_CSV)
        df = ImportCSV(csv_data=csv.output)
        salaries = ExtractColumn(dataframe=df.output, column_name="salary")
        sink = Output(name="r", value=salaries.output)
        bag = await run_graph_async(create_graph(sink))
        assert len(bag["r"]) == 5
        assert 120000 in bag["r"]

    @pytest.mark.asyncio
    async def test_select_two_columns(self):
        csv = ConstString(value=EMPLOYEE_CSV)
        df = ImportCSV(csv_data=csv.output)
        selected = SelectColumn(dataframe=df.output, columns="emp_name,salary")
        rows = ToList(dataframe=selected.output)
        sink = Output(name="r", value=rows.output)
        bag = await run_graph_async(create_graph(sink))
        assert set(bag["r"][0].keys()) == {"emp_name", "salary"}


# ---------------------------------------------------------------------------
# Scenario: Sensor data cleaning — fill missing, drop dupes
# ---------------------------------------------------------------------------
class TestSensorCleaning:
    @pytest.mark.asyncio
    async def test_fill_missing_readings(self):
        csv = ConstString(value=SENSOR_CSV)
        df = ImportCSV(csv_data=csv.output)
        filled = FillNA(dataframe=df.output, value=0.0, method="value", columns="reading")
        rows = ToList(dataframe=filled.output)
        sink = Output(name="r", value=rows.output)
        bag = await run_graph_async(create_graph(sink))
        readings = [row["reading"] for row in bag["r"]]
        assert readings == [22.5, 0.0, 19.3]

    @pytest.mark.asyncio
    async def test_drop_null_rows(self):
        csv = ConstString(value=SENSOR_CSV)
        df = ImportCSV(csv_data=csv.output)
        clean = DropNA(df=df.output)
        rows = ToList(dataframe=clean.output)
        sink = Output(name="r", value=rows.output)
        bag = await run_graph_async(create_graph(sink))
        assert len(bag["r"]) == 2

    @pytest.mark.asyncio
    async def test_remove_duplicate_rows(self):
        csv = ConstString(value=DEDUP_CSV)
        df = ImportCSV(csv_data=csv.output)
        deduped = DropDuplicates(df=df.output)
        rows = ToList(dataframe=deduped.output)
        sink = Output(name="r", value=rows.output)
        bag = await run_graph_async(create_graph(sink))
        assert len(bag["r"]) == 2


# ---------------------------------------------------------------------------
# Scenario: Sales aggregation — group-by region and sum
# ---------------------------------------------------------------------------
class TestSalesAggregation:
    @pytest.mark.asyncio
    async def test_total_by_region(self):
        csv = ConstString(value=SALES_CSV)
        df = ImportCSV(csv_data=csv.output)
        agg = Aggregate(dataframe=df.output, columns="region", aggregation="sum")
        rows = ToList(dataframe=agg.output)
        sink = Output(name="r", value=rows.output)
        bag = await run_graph_async(create_graph(sink))
        by_region = {row["region"]: row["revenue"] for row in bag["r"]}
        assert by_region["North"] == 2500
        assert by_region["South"] == 1700


# ---------------------------------------------------------------------------
# Scenario: Merge two datasets side-by-side
# ---------------------------------------------------------------------------
class TestDatasetMerge:
    @pytest.mark.asyncio
    async def test_horizontal_merge(self):
        csv_l = ConstString(value="product\nA\nB\n")
        csv_r = ConstString(value="price\n9.99\n14.50\n")
        df_l = ImportCSV(csv_data=csv_l.output)
        df_r = ImportCSV(csv_data=csv_r.output)
        merged = Merge(dataframe_a=df_l.output, dataframe_b=df_r.output)
        rows = ToList(dataframe=merged.output)
        sink = Output(name="r", value=rows.output)
        bag = await run_graph_async(create_graph(sink))
        assert bag["r"][0]["product"] == "A"
        assert bag["r"][0]["price"] == 9.99


# ---------------------------------------------------------------------------
# Scenario: Append rows from two sources
# ---------------------------------------------------------------------------
class TestRowAppend:
    @pytest.mark.asyncio
    async def test_vertical_stack(self):
        csv1 = ConstString(value="sku,qty\nX,3\n")
        csv2 = ConstString(value="sku,qty\nY,7\n")
        df1 = ImportCSV(csv_data=csv1.output)
        df2 = ImportCSV(csv_data=csv2.output)
        stacked = DataAppend(dataframe_a=df1.output, dataframe_b=df2.output)
        rows = ToList(dataframe=stacked.output)
        sink = Output(name="r", value=rows.output)
        bag = await run_graph_async(create_graph(sink))
        assert len(bag["r"]) == 2
        skus = {row["sku"] for row in bag["r"]}
        assert skus == {"X", "Y"}


# ---------------------------------------------------------------------------
# Scenario: Find a specific row by condition
# ---------------------------------------------------------------------------
class TestRowLookup:
    @pytest.mark.asyncio
    async def test_find_by_ticket(self):
        csv = ConstString(value=TICKET_CSV)
        df = ImportCSV(csv_data=csv.output)
        found = FindRow(df=df.output, condition="ticket == 'T-200'")
        rows = ToList(dataframe=found.output)
        sink = Output(name="r", value=rows.output)
        bag = await run_graph_async(create_graph(sink))
        assert len(bag["r"]) == 1
        assert bag["r"][0]["priority"] == "high"


# ---------------------------------------------------------------------------
# Scenario: Full ETL pipeline — ingest → clean → aggregate → export
# ---------------------------------------------------------------------------
class TestFullETLPipeline:
    @pytest.mark.asyncio
    async def test_end_to_end(self):
        raw_csv = (
            "dept,spend\n"
            "Eng,500\n"
            "Eng,\n"
            "Sales,300\n"
            "Sales,200\n"
            "Eng,500\n"
        )
        csv = ConstString(value=raw_csv)
        df = ImportCSV(csv_data=csv.output)

        # Fill nulls, drop exact duplicates, aggregate
        filled = FillNA(dataframe=df.output, value=0, method="value", columns="spend")
        deduped = DropDuplicates(df=filled.output)
        agg = Aggregate(dataframe=deduped.output, columns="dept", aggregation="sum")
        rows = ToList(dataframe=agg.output)
        sink = Output(name="r", value=rows.output)
        bag = await run_graph_async(create_graph(sink))

        by_dept = {row["dept"]: row["spend"] for row in bag["r"]}
        assert by_dept["Eng"] == 500  # 500 + 0, duplicate 500 removed
        assert isinstance(bag["r"], list)
        assert all(isinstance(r, dict) for r in bag["r"])
