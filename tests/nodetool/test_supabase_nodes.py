"""
Tests for the Supabase database nodes.

The Supabase client is mocked to avoid requiring real credentials.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from nodetool.workflows.processing_context import ProcessingContext
from nodetool.nodes.lib.supabase import (
    Select,
    Insert,
    Update,
    Delete,
    Upsert,
    RPC,
    FilterOp,
    _qualify_table_name,
    _apply_filters,
)


@pytest.fixture
def context():
    return ProcessingContext(user_id="test", auth_token="test")


def _make_mock_client(rows=None):
    """
    Build a mock supabase async client whose chainable query methods
    ultimately return ``AsyncMock(execute=AsyncMock(return_value=response))``.
    """
    if rows is None:
        rows = []

    response = MagicMock()
    response.data = rows

    # A single query builder mock where every method returns itself so we can
    # chain .select().eq().order().limit().execute() etc.
    query = AsyncMock()
    query.select = MagicMock(return_value=query)
    query.eq = MagicMock(return_value=query)
    query.neq = MagicMock(return_value=query)
    query.gt = MagicMock(return_value=query)
    query.gte = MagicMock(return_value=query)
    query.lt = MagicMock(return_value=query)
    query.lte = MagicMock(return_value=query)
    query.in_ = MagicMock(return_value=query)
    query.like = MagicMock(return_value=query)
    query.contains = MagicMock(return_value=query)
    query.order = MagicMock(return_value=query)
    query.limit = MagicMock(return_value=query)
    query.insert = MagicMock(return_value=query)
    query.update = MagicMock(return_value=query)
    query.upsert = MagicMock(return_value=query)
    query.delete = MagicMock(return_value=query)
    query.execute = AsyncMock(return_value=response)

    client = MagicMock()
    client.table = MagicMock(return_value=query)
    rpc_query = AsyncMock()
    rpc_query.execute = AsyncMock(return_value=response)
    client.rpc = MagicMock(return_value=rpc_query)

    return client, query, response


# ---------------------------------------------------------------------------
# _qualify_table_name helper
# ---------------------------------------------------------------------------


def test_qualify_table_name_no_prefix():
    """Table name unchanged when no prefix is configured."""
    with patch(
        "nodetool.nodes.lib.supabase.Environment.get_node_supabase_table_prefix",
        return_value="",
    ):
        assert _qualify_table_name("users") == "users"


def test_qualify_table_name_with_prefix():
    """Prefix is prepended when configured."""
    with patch(
        "nodetool.nodes.lib.supabase.Environment.get_node_supabase_table_prefix",
        return_value="app_",
    ):
        assert _qualify_table_name("users") == "app_users"


def test_qualify_table_name_already_prefixed():
    """Prefix is not double-applied."""
    with patch(
        "nodetool.nodes.lib.supabase.Environment.get_node_supabase_table_prefix",
        return_value="app_",
    ):
        assert _qualify_table_name("app_users") == "app_users"


# ---------------------------------------------------------------------------
# _apply_filters helper
# ---------------------------------------------------------------------------


def test_apply_filters_eq():
    query = MagicMock()
    query.eq = MagicMock(return_value=query)
    _apply_filters(query, [("id", FilterOp.EQ, 1)])
    query.eq.assert_called_once_with("id", 1)


def test_apply_filters_all_ops():
    """Test each FilterOp is routed to the correct query method."""
    ops_and_methods = [
        (FilterOp.EQ, "eq"),
        (FilterOp.NE, "neq"),
        (FilterOp.GT, "gt"),
        (FilterOp.GTE, "gte"),
        (FilterOp.LT, "lt"),
        (FilterOp.LTE, "lte"),
        (FilterOp.IN, "in_"),
        (FilterOp.LIKE, "like"),
        (FilterOp.CONTAINS, "contains"),
    ]
    for op, method_name in ops_and_methods:
        query = MagicMock()
        setattr(query, method_name, MagicMock(return_value=query))
        _apply_filters(query, [("field", op, "value")])
        getattr(query, method_name).assert_called_once_with("field", "value")


def test_apply_filters_unknown_op():
    query = MagicMock()
    with pytest.raises(ValueError, match="Unsupported operator"):
        _apply_filters(query, [("id", "invalid_op", 1)])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Select node
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_select_raises_when_table_name_empty(context: ProcessingContext):
    node = Select(table_name="")
    with pytest.raises(ValueError, match="table_name cannot be empty"):
        await node.process(context)


@pytest.mark.asyncio
async def test_select_returns_rows(context: ProcessingContext):
    rows = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
    client, query, _ = _make_mock_client(rows)

    with patch(
        "nodetool.nodes.lib.supabase._get_supabase_client", AsyncMock(return_value=client)
    ), patch(
        "nodetool.nodes.lib.supabase._qualify_table_name", side_effect=lambda t: t
    ):
        node = Select(table_name="users")
        result = await node.process(context)

    assert result == rows
    client.table.assert_called_once_with("users")


@pytest.mark.asyncio
async def test_select_with_order_and_limit(context: ProcessingContext):
    client, query, _ = _make_mock_client([{"id": 1}])

    with patch(
        "nodetool.nodes.lib.supabase._get_supabase_client", AsyncMock(return_value=client)
    ), patch(
        "nodetool.nodes.lib.supabase._qualify_table_name", side_effect=lambda t: t
    ):
        node = Select(table_name="users", order_by="id", descending=True, limit=5)
        await node.process(context)

    query.order.assert_called_once_with("id", desc=True)
    query.limit.assert_called_once_with(5)


@pytest.mark.asyncio
async def test_select_with_filters(context: ProcessingContext):
    client, query, _ = _make_mock_client([{"id": 1}])

    with patch(
        "nodetool.nodes.lib.supabase._get_supabase_client", AsyncMock(return_value=client)
    ), patch(
        "nodetool.nodes.lib.supabase._qualify_table_name", side_effect=lambda t: t
    ):
        node = Select(
            table_name="users",
            filters=[("id", FilterOp.GT, 0)],
        )
        await node.process(context)

    query.gt.assert_called_once_with("id", 0)


# ---------------------------------------------------------------------------
# Insert node
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_insert_raises_when_table_name_empty(context: ProcessingContext):
    node = Insert(table_name="", records=[{"name": "Alice"}])
    with pytest.raises(ValueError, match="table_name cannot be empty"):
        await node.process(context)


@pytest.mark.asyncio
async def test_insert_single_record(context: ProcessingContext):
    rows = [{"id": 1, "name": "Alice"}]
    client, query, _ = _make_mock_client(rows)

    with patch(
        "nodetool.nodes.lib.supabase._get_supabase_client", AsyncMock(return_value=client)
    ), patch(
        "nodetool.nodes.lib.supabase._qualify_table_name", side_effect=lambda t: t
    ):
        node = Insert(table_name="users", records=[{"name": "Alice"}])
        result = await node.process(context)

    query.insert.assert_called_once_with([{"name": "Alice"}])
    assert result == rows


@pytest.mark.asyncio
async def test_insert_dict_record(context: ProcessingContext):
    """A single dict (not list) should be wrapped in a list."""
    rows = [{"id": 1, "name": "Bob"}]
    client, query, _ = _make_mock_client(rows)

    with patch(
        "nodetool.nodes.lib.supabase._get_supabase_client", AsyncMock(return_value=client)
    ), patch(
        "nodetool.nodes.lib.supabase._qualify_table_name", side_effect=lambda t: t
    ):
        node = Insert(table_name="users", records={"name": "Bob"})
        await node.process(context)

    query.insert.assert_called_once_with([{"name": "Bob"}])


# ---------------------------------------------------------------------------
# Update node
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_raises_when_table_name_empty(context: ProcessingContext):
    node = Update(table_name="", values={"name": "Charlie"})
    with pytest.raises(ValueError, match="table_name cannot be empty"):
        await node.process(context)


@pytest.mark.asyncio
async def test_update_raises_when_values_empty(context: ProcessingContext):
    node = Update(table_name="users", values={})
    with pytest.raises(ValueError, match="values cannot be empty"):
        await node.process(context)


@pytest.mark.asyncio
async def test_update_success(context: ProcessingContext):
    rows = [{"id": 1, "name": "Updated"}]
    client, query, _ = _make_mock_client(rows)

    with patch(
        "nodetool.nodes.lib.supabase._get_supabase_client", AsyncMock(return_value=client)
    ), patch(
        "nodetool.nodes.lib.supabase._qualify_table_name", side_effect=lambda t: t
    ):
        node = Update(
            table_name="users",
            values={"name": "Updated"},
            filters=[("id", FilterOp.EQ, 1)],
        )
        result = await node.process(context)

    query.update.assert_called_once_with({"name": "Updated"})
    query.eq.assert_called_once_with("id", 1)
    assert result == rows


# ---------------------------------------------------------------------------
# Delete node
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_raises_when_table_name_empty(context: ProcessingContext):
    node = Delete(table_name="", filters=[("id", FilterOp.EQ, 1)])
    with pytest.raises(ValueError, match="table_name cannot be empty"):
        await node.process(context)


@pytest.mark.asyncio
async def test_delete_raises_when_no_filters(context: ProcessingContext):
    """Delete without filters should be rejected for safety."""
    node = Delete(table_name="users", filters=[])
    with pytest.raises(ValueError, match="At least one filter is required"):
        await node.process(context)


@pytest.mark.asyncio
async def test_delete_success(context: ProcessingContext):
    client, query, _ = _make_mock_client([])

    with patch(
        "nodetool.nodes.lib.supabase._get_supabase_client", AsyncMock(return_value=client)
    ), patch(
        "nodetool.nodes.lib.supabase._qualify_table_name", side_effect=lambda t: t
    ):
        node = Delete(
            table_name="users",
            filters=[("id", FilterOp.EQ, 42)],
        )
        result = await node.process(context)

    assert result == {"deleted": True}
    query.delete.assert_called_once()
    query.eq.assert_called_once_with("id", 42)


# ---------------------------------------------------------------------------
# Upsert node
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_raises_when_table_name_empty(context: ProcessingContext):
    node = Upsert(table_name="", records=[{"id": 1}])
    with pytest.raises(ValueError, match="table_name cannot be empty"):
        await node.process(context)


@pytest.mark.asyncio
async def test_upsert_success(context: ProcessingContext):
    rows = [{"id": 1, "name": "Alice"}]
    client, query, _ = _make_mock_client(rows)

    with patch(
        "nodetool.nodes.lib.supabase._get_supabase_client", AsyncMock(return_value=client)
    ), patch(
        "nodetool.nodes.lib.supabase._qualify_table_name", side_effect=lambda t: t
    ):
        node = Upsert(table_name="users", records=[{"id": 1, "name": "Alice"}])
        result = await node.process(context)

    query.upsert.assert_called_once_with([{"id": 1, "name": "Alice"}])
    assert result == rows


# ---------------------------------------------------------------------------
# RPC node
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rpc_raises_when_function_empty(context: ProcessingContext):
    node = RPC(function="")
    with pytest.raises(ValueError, match="function cannot be empty"):
        await node.process(context)


@pytest.mark.asyncio
async def test_rpc_success(context: ProcessingContext):
    response_data = {"result": 42}
    client, _, _ = _make_mock_client()
    rpc_resp = MagicMock()
    rpc_resp.data = response_data
    rpc_query = AsyncMock()
    rpc_query.execute = AsyncMock(return_value=rpc_resp)
    client.rpc = MagicMock(return_value=rpc_query)

    with patch(
        "nodetool.nodes.lib.supabase._get_supabase_client", AsyncMock(return_value=client)
    ):
        node = RPC(function="my_func", params={"x": 1})
        result = await node.process(context)

    client.rpc.assert_called_once_with("my_func", {"x": 1})
    assert result == response_data


# ---------------------------------------------------------------------------
# _get_supabase_client raises when credentials are missing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_supabase_client_raises_when_no_credentials():
    with patch(
        "nodetool.nodes.lib.supabase.Environment.get_node_supabase_url",
        return_value=None,
    ), patch(
        "nodetool.nodes.lib.supabase.Environment.get_node_supabase_key",
        return_value=None,
    ):
        from nodetool.nodes.lib.supabase import _get_supabase_client

        with pytest.raises(ValueError, match="Supabase URL or key is not set"):
            await _get_supabase_client()
