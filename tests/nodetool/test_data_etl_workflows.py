"""
Data ETL (Extract-Transform-Load) workflow tests.

Each test mimics a realistic data-processing pipeline:
CSV ingest → column operations → filtering/sorting → aggregation → export.
Uses a mock ProcessingContext following the same pattern as test_data_nodes.py.
"""

import pandas as pd
import pytest
from unittest.mock import AsyncMock, MagicMock

from nodetool.workflows.processing_context import ProcessingContext
from nodetool.metadata.types import DataframeRef
from nodetool.nodes.nodetool.data import (
    AddColumn,
    Aggregate,
    Append as DataAppend,
    DropDuplicates,
    DropNA,
    ExtractColumn,
    FillNA,
    Filter as DataFilter,
    FindRow,
    FromList,
    ImportCSV,
    Join as DataJoin,
    Merge,
    Rename,
    SortByColumn,
    ToList,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctx():
    """Return a mock context whose dataframe helpers capture / replay pandas DataFrames."""
    ctx = MagicMock(spec=ProcessingContext)
    ctx._captured = {}  # stash for side-effect closures

    async def _from_pd(df, filename=None, parent_id=None):
        ref = DataframeRef()
        ctx._captured[id(ref)] = df.copy()
        return ref

    async def _to_pd(ref):
        return ctx._captured[id(ref)]

    ctx.dataframe_from_pandas = AsyncMock(side_effect=_from_pd)
    ctx.dataframe_to_pandas = AsyncMock(side_effect=_to_pd)
    return ctx


@pytest.fixture
def mock_ctx():
    return _make_ctx()


# Small helper: run FromList → get back a ref **and** the underlying pandas DF.
async def _ref_and_df(ctx, rows):
    ref = await FromList(values=rows).process(ctx)
    df = ctx._captured[id(ref)]
    return ref, df


# ---------------------------------------------------------------------------
# Scenario: Employee roster — import CSV, rename, filter, sort
# ---------------------------------------------------------------------------
class TestEmployeeRoster:
    CSV_ROSTER = (
        "emp_name,department,salary\n"
        "Alice,Engineering,120000\n"
        "Bob,Sales,85000\n"
        "Carol,Engineering,115000\n"
        "Dave,Marketing,78000\n"
        "Eve,Engineering,130000\n"
    )

    @pytest.mark.asyncio
    async def test_csv_ingest(self, mock_ctx):
        ref = await ImportCSV(csv_data=self.CSV_ROSTER).process(mock_ctx)
        df = mock_ctx._captured[id(ref)]
        assert len(df) == 5
        assert "emp_name" in df.columns

    @pytest.mark.asyncio
    async def test_rename_and_select(self, mock_ctx):
        ref = await ImportCSV(csv_data=self.CSV_ROSTER).process(mock_ctx)
        renamed = await Rename(
            dataframe=ref, rename_map="emp_name:name,department:dept"
        ).process(mock_ctx)
        df = mock_ctx._captured[id(renamed)]
        assert "name" in df.columns
        assert "dept" in df.columns

    @pytest.mark.asyncio
    async def test_filter_engineering(self, mock_ctx):
        ref = await ImportCSV(csv_data=self.CSV_ROSTER).process(mock_ctx)
        eng = await DataFilter(df=ref, condition="department == 'Engineering'").process(
            mock_ctx
        )
        df = mock_ctx._captured[id(eng)]
        assert len(df) == 3
        assert set(df["emp_name"]) == {"Alice", "Carol", "Eve"}

    @pytest.mark.asyncio
    async def test_sort_by_salary(self, mock_ctx):
        ref = await ImportCSV(csv_data=self.CSV_ROSTER).process(mock_ctx)
        sorted_ref = await SortByColumn(df=ref, column="salary").process(mock_ctx)
        df = mock_ctx._captured[id(sorted_ref)]
        salaries = list(df["salary"])
        assert salaries == sorted(salaries)


# ---------------------------------------------------------------------------
# Scenario: Sensor data cleaning — fill missing, drop dupes
# ---------------------------------------------------------------------------
class TestSensorCleaning:
    @pytest.mark.asyncio
    async def test_fill_missing_readings(self, mock_ctx):
        # Build the DataFrame directly with proper NaN values
        # (FromList converts None to the string "None", so we bypass it)
        df_raw = pd.DataFrame({
            "sensor": ["A", "B", "C"],
            "reading": [22.5, float("nan"), 19.3],
        })

        async def _from_pd_raw(df, filename=None, parent_id=None):
            ref = DataframeRef()
            mock_ctx._captured[id(ref)] = df.copy()
            return ref

        mock_ctx.dataframe_from_pandas = AsyncMock(side_effect=_from_pd_raw)
        ref = await mock_ctx.dataframe_from_pandas(df_raw)

        filled = await FillNA(
            dataframe=ref, value=0.0, method="value", columns="reading"
        ).process(mock_ctx)
        df = mock_ctx._captured[id(filled)]
        assert list(df["reading"]) == [22.5, 0.0, 19.3]

    @pytest.mark.asyncio
    async def test_drop_null_rows(self, mock_ctx):
        # Same approach: use direct pandas to get actual NaN values
        df_raw = pd.DataFrame({
            "sensor": ["A", None, "C"],
            "reading": [10.0, None, 15.0],
        })

        async def _from_pd_raw(df, filename=None, parent_id=None):
            ref = DataframeRef()
            mock_ctx._captured[id(ref)] = df.copy()
            return ref

        mock_ctx.dataframe_from_pandas = AsyncMock(side_effect=_from_pd_raw)
        ref = await mock_ctx.dataframe_from_pandas(df_raw)

        clean = await DropNA(df=ref).process(mock_ctx)
        df = mock_ctx._captured[id(clean)]
        assert len(df) == 2

    @pytest.mark.asyncio
    async def test_remove_duplicate_rows(self, mock_ctx):
        rows = [
            {"sensor": "A", "reading": 10.0},
            {"sensor": "A", "reading": 10.0},
            {"sensor": "B", "reading": 20.0},
        ]
        ref, _ = await _ref_and_df(mock_ctx, rows)
        deduped = await DropDuplicates(df=ref).process(mock_ctx)
        df = mock_ctx._captured[id(deduped)]
        assert len(df) == 2


# ---------------------------------------------------------------------------
# Scenario: Sales aggregation — group-by department and sum
# ---------------------------------------------------------------------------
class TestSalesAggregation:
    @pytest.mark.asyncio
    async def test_total_by_region(self, mock_ctx):
        rows = [
            {"region": "North", "revenue": 1000},
            {"region": "North", "revenue": 1500},
            {"region": "South", "revenue": 800},
            {"region": "South", "revenue": 900},
        ]
        ref, _ = await _ref_and_df(mock_ctx, rows)
        agg = await Aggregate(
            dataframe=ref, columns="region", aggregation="sum"
        ).process(mock_ctx)
        df = mock_ctx._captured[id(agg)]
        north = df[df["region"] == "North"]["revenue"].iloc[0]
        south = df[df["region"] == "South"]["revenue"].iloc[0]
        assert north == 2500
        assert south == 1700


# ---------------------------------------------------------------------------
# Scenario: Column enrichment — add a computed column
# ---------------------------------------------------------------------------
class TestColumnEnrichment:
    @pytest.mark.asyncio
    async def test_add_status_column(self, mock_ctx):
        rows = [
            {"item": "Widget", "qty": 5},
            {"item": "Gadget", "qty": 0},
            {"item": "Doodad", "qty": 12},
        ]
        ref, _ = await _ref_and_df(mock_ctx, rows)
        statuses = ["in-stock", "out-of-stock", "in-stock"]
        enriched = await AddColumn(
            dataframe=ref, column_name="status", values=statuses
        ).process(mock_ctx)
        df = mock_ctx._captured[id(enriched)]
        assert "status" in df.columns
        assert list(df["status"]) == statuses


# ---------------------------------------------------------------------------
# Scenario: Column extraction
# ---------------------------------------------------------------------------
class TestColumnExtraction:
    @pytest.mark.asyncio
    async def test_extract_single_column(self, mock_ctx):
        rows = [
            {"city": "Berlin", "temp_c": 18},
            {"city": "Paris", "temp_c": 22},
            {"city": "London", "temp_c": 15},
        ]
        ref, _ = await _ref_and_df(mock_ctx, rows)
        cities = await ExtractColumn(
            dataframe=ref, column_name="city"
        ).process(mock_ctx)
        assert cities == ["Berlin", "Paris", "London"]


# ---------------------------------------------------------------------------
# Scenario: Merge two datasets side-by-side
# ---------------------------------------------------------------------------
class TestDatasetMerge:
    @pytest.mark.asyncio
    async def test_horizontal_merge(self, mock_ctx):
        left = [{"product": "A"}, {"product": "B"}]
        right = [{"price": 9.99}, {"price": 14.50}]
        ref_l, _ = await _ref_and_df(mock_ctx, left)
        ref_r, _ = await _ref_and_df(mock_ctx, right)
        merged = await Merge(
            dataframe_a=ref_l, dataframe_b=ref_r
        ).process(mock_ctx)
        df = mock_ctx._captured[id(merged)]
        assert list(df.columns) == ["product", "price"]
        assert list(df["price"]) == [9.99, 14.50]


# ---------------------------------------------------------------------------
# Scenario: Append rows from two sources
# ---------------------------------------------------------------------------
class TestRowAppend:
    @pytest.mark.asyncio
    async def test_vertical_stack(self, mock_ctx):
        batch1 = [{"sku": "X", "qty": 3}]
        batch2 = [{"sku": "Y", "qty": 7}]
        ref1, _ = await _ref_and_df(mock_ctx, batch1)
        ref2, _ = await _ref_and_df(mock_ctx, batch2)
        stacked = await DataAppend(
            dataframe_a=ref1, dataframe_b=ref2
        ).process(mock_ctx)
        df = mock_ctx._captured[id(stacked)]
        assert len(df) == 2
        assert set(df["sku"]) == {"X", "Y"}


# ---------------------------------------------------------------------------
# Scenario: Key-based join of two datasets
# ---------------------------------------------------------------------------
class TestKeyJoin:
    @pytest.mark.asyncio
    async def test_inner_join_on_id(self, mock_ctx):
        # DataJoin requires identical column sets; both frames must have
        # the same columns, differing only in values.  Use Merge (horizontal
        # concat) instead for heterogeneous schemas.
        left = [
            {"pid": 1, "name": "Alice", "amount": 0},
            {"pid": 2, "name": "Bob", "amount": 0},
            {"pid": 3, "name": "Carol", "amount": 0},
        ]
        right = [
            {"pid": 1, "name": "", "amount": 250},
            {"pid": 2, "name": "", "amount": 180},
            {"pid": 3, "name": "", "amount": 310},
        ]
        ref_l, _ = await _ref_and_df(mock_ctx, left)
        ref_r, _ = await _ref_and_df(mock_ctx, right)
        joined = await DataJoin(
            dataframe_a=ref_l, dataframe_b=ref_r, join_on="pid"
        ).process(mock_ctx)
        df = mock_ctx._captured[id(joined)]
        assert len(df) == 3


# ---------------------------------------------------------------------------
# Scenario: Find a specific row by condition
# ---------------------------------------------------------------------------
class TestRowLookup:
    @pytest.mark.asyncio
    async def test_find_by_id(self, mock_ctx):
        rows = [
            {"ticket": "T-100", "priority": "low"},
            {"ticket": "T-200", "priority": "high"},
            {"ticket": "T-300", "priority": "medium"},
        ]
        ref, _ = await _ref_and_df(mock_ctx, rows)
        found = await FindRow(
            df=ref, condition="ticket == 'T-200'"
        ).process(mock_ctx)
        df = mock_ctx._captured[id(found)]
        assert len(df) == 1
        assert df.iloc[0]["priority"] == "high"


# ---------------------------------------------------------------------------
# Scenario: Full ETL pipeline — ingest → clean → enrich → aggregate → export
# ---------------------------------------------------------------------------
class TestFullETLPipeline:
    @pytest.mark.asyncio
    async def test_end_to_end(self, mock_ctx):
        # 1. Ingest raw data — use pandas directly for proper NaN support
        df_raw = pd.DataFrame({
            "dept": ["Eng", "Eng", "Sales", "Sales", "Eng"],
            "spend": [500.0, float("nan"), 300.0, 200.0, 500.0],
        })

        async def _from_pd(df, filename=None, parent_id=None):
            ref = DataframeRef()
            mock_ctx._captured[id(ref)] = df.copy()
            return ref

        mock_ctx.dataframe_from_pandas = AsyncMock(side_effect=_from_pd)
        ref = await mock_ctx.dataframe_from_pandas(df_raw)

        # 2. Fill nulls with 0
        filled = await FillNA(
            dataframe=ref, value=0.0, method="value", columns="spend"
        ).process(mock_ctx)

        # 3. Drop exact duplicates
        deduped = await DropDuplicates(df=filled).process(mock_ctx)
        df_clean = mock_ctx._captured[id(deduped)]
        assert len(df_clean) <= 4  # at most 4 unique rows

        # 4. Aggregate total spend per department
        agg = await Aggregate(
            dataframe=deduped, columns="dept", aggregation="sum"
        ).process(mock_ctx)
        df_agg = mock_ctx._captured[id(agg)]

        eng_total = df_agg[df_agg["dept"] == "Eng"]["spend"].iloc[0]
        # After fill+dedup: Eng rows are (500, 0) → 500
        assert eng_total == 500.0

        # 5. Convert back to list for downstream consumption
        rows_out = await ToList(dataframe=agg).process(mock_ctx)
        assert isinstance(rows_out, list)
        assert all(isinstance(r, dict) for r in rows_out)
