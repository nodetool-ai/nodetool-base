import pytest
import pandas as pd
from unittest.mock import AsyncMock, MagicMock

from nodetool.workflows.processing_context import ProcessingContext
from nodetool.metadata.types import DataframeRef
from nodetool.nodes.nodetool.data import (
    FromList,
    ToList,
    SelectColumn,
    ExtractColumn,
    AddColumn,
    Merge,
    Append,
    Aggregate,
    Pivot,
    Rename,
    FillNA,
)


@pytest.fixture
def mock_context():
    ctx = MagicMock(spec=ProcessingContext)
    ctx.dataframe_to_pandas = AsyncMock()
    ctx.dataframe_from_pandas = AsyncMock(return_value=DataframeRef())
    return ctx


@pytest.mark.asyncio
async def test_from_list_and_to_list(mock_context: ProcessingContext):
    node = FromList(values=[{"a": 1}, {"a": 2}])

    captured = {}

    async def df_from_pd(df, filename=None, parent_id=None):
        captured["df"] = df.copy()
        return DataframeRef()

    mock_context.dataframe_from_pandas = AsyncMock(side_effect=df_from_pd)

    df_ref = await node.process(mock_context)
    assert isinstance(df_ref, DataframeRef)
    assert list(captured["df"]["a"]) == [1, 2]

    # Now round-trip with ToList
    mock_context.dataframe_to_pandas = AsyncMock(return_value=captured["df"]) 
    list_node = ToList(dataframe=DataframeRef())
    out = await list_node.process(mock_context)
    assert out == [{"a": 1}, {"a": 2}]


@pytest.mark.asyncio
async def test_select_and_extract_and_add_column(mock_context: ProcessingContext):
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    mock_context.dataframe_to_pandas = AsyncMock(return_value=df)

    # Select columns
    captured = {}

    async def df_from_pd(df, filename=None, parent_id=None):
        captured["selected"] = df.copy()
        return DataframeRef()

    mock_context.dataframe_from_pandas = AsyncMock(side_effect=df_from_pd)
    sel = SelectColumn(dataframe=DataframeRef(), columns="a,b")
    await sel.process(mock_context)
    assert list(captured["selected"].columns) == ["a", "b"]

    # Extract column
    mock_context.dataframe_to_pandas = AsyncMock(return_value=df)
    ext = ExtractColumn(dataframe=DataframeRef(), column_name="b")
    vals = await ext.process(mock_context)
    assert vals == [3, 4]

    # Add column
    captured_add = {}

    async def df_from_pd_add(df, filename=None, parent_id=None):
        captured_add["df"] = df.copy()
        return DataframeRef()

    mock_context.dataframe_from_pandas = AsyncMock(side_effect=df_from_pd_add)
    add = AddColumn(dataframe=DataframeRef(), column_name="d", values=[9, 8])
    await add.process(mock_context)
    assert list(captured_add["df"]["d"]) == [9, 8]


@pytest.mark.asyncio
async def test_merge_and_append(mock_context: ProcessingContext):
    df_a = pd.DataFrame({"x": [1, 2]})
    df_b = pd.DataFrame({"y": [3, 4]})

    # Merge (concat columns)
    mock_context.dataframe_to_pandas = AsyncMock(side_effect=[df_a, df_b])
    captured_merge = {}

    async def df_from_pd_merge(df, filename=None, parent_id=None):
        captured_merge["df"] = df.copy()
        return DataframeRef()

    mock_context.dataframe_from_pandas = AsyncMock(side_effect=df_from_pd_merge)
    await Merge(dataframe_a=DataframeRef(), dataframe_b=DataframeRef()).process(
        mock_context
    )
    assert list(captured_merge["df"].columns) == ["x", "y"]

    # Append (concat rows)
    df_a2 = pd.DataFrame({"x": [1], "y": [3]})
    df_b2 = pd.DataFrame({"x": [2], "y": [4]})
    mock_context.dataframe_to_pandas = AsyncMock(side_effect=[df_a2, df_b2])
    captured_append = {}

    async def df_from_pd_append(df, filename=None, parent_id=None):
        captured_append["df"] = df.copy()
        return DataframeRef()

    mock_context.dataframe_from_pandas = AsyncMock(side_effect=df_from_pd_append)
    await Append(dataframe_a=DataframeRef(), dataframe_b=DataframeRef()).process(
        mock_context
    )
    assert len(captured_append["df"]) == 2


@pytest.mark.asyncio
async def test_aggregate_pivot_rename_fillna(mock_context: ProcessingContext):
    df = pd.DataFrame(
        {
            "grp": ["A", "A", "B"],
            "k": ["x", "y", "x"],
            "v": [1.0, None, 3.0],
        }
    )
    # Aggregate sum by grp
    mock_context.dataframe_to_pandas = AsyncMock(return_value=df)
    captured = {}

    async def df_from_pd(df, filename=None, parent_id=None):
        captured.setdefault("dfs", []).append(df.copy())
        return DataframeRef()

    mock_context.dataframe_from_pandas = AsyncMock(side_effect=df_from_pd)
    agg = Aggregate(dataframe=DataframeRef(), columns="grp", aggregation="sum")
    await agg.process(mock_context)
    out = captured["dfs"][0]
    assert list(out["grp"]) == ["A", "B"]
    # v sum, with None treated as NaN -> sum of A is 1.0
    assert list(out["v"]) == [1.0, 3.0]

    # Pivot k across grp with sum of v
    mock_context.dataframe_to_pandas = AsyncMock(return_value=df)
    piv = Pivot(
        dataframe=DataframeRef(), index="grp", columns="k", values="v", aggfunc="sum"
    )
    await piv.process(mock_context)
    out2 = captured["dfs"][1]
    assert set(out2.columns) >= {"grp", "x", "y"}

    # Rename columns
    mock_context.dataframe_to_pandas = AsyncMock(return_value=df)
    rn = Rename(dataframe=DataframeRef(), rename_map="grp:group,v:value")
    await rn.process(mock_context)
    out3 = captured["dfs"][2]
    assert "group" in out3.columns and "value" in out3.columns

    # FillNA
    mock_context.dataframe_to_pandas = AsyncMock(return_value=df)
    fn = FillNA(dataframe=DataframeRef(), value=0, method="value", columns="v")
    await fn.process(mock_context)
    out4 = captured["dfs"][3]
    assert list(out4["v"]) == [1.0, 0, 3.0]
