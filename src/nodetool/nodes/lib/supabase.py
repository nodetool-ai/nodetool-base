"""
Supabase database nodes mirroring the SQLite nodes' features.

These nodes provide typed, safe access to Supabase tables for:
- Select (query with filters, order, limit)
- Insert (single or multiple records)
- Update (with optional filters)
- Delete (requires filters for safety)

Authentication and configuration are handled via nodetool-core Environment:
- NODE_SUPABASE_URL / NODE_SUPABASE_KEY (required for user-provided nodes)
- Optional NODE_SUPABASE_SCHEMA and NODE_SUPABASE_TABLE_PREFIX for namespacing

When Supabase is configured, existing asset save/load nodes will transparently
use Supabase Storage via Environment; these DB nodes focus purely on table CRUD.
"""

from typing import Any, ClassVar
from enum import Enum

from nodetool.config.environment import Environment
from pydantic import Field

from nodetool.metadata.types import RecordType
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext


class FilterOp(str, Enum):
    EQ = "eq"
    NE = "ne"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    IN = "in"
    LIKE = "like"
    CONTAINS = "contains"


# Use a tuple form for filters to keep metadata compatible:
# (field, operator, value)
Filter = tuple[str, FilterOp, Any]


def _qualify_table_name(table_name: str) -> str:
    """
    Apply NODE_SUPABASE_TABLE_PREFIX when set to avoid collisions with core tables.
    """
    prefix = Environment.get_node_supabase_table_prefix()
    if prefix and not table_name.startswith(prefix):
        return f"{prefix}{table_name}"
    return table_name


async def _get_supabase_client():
    """
    Build a Supabase client using node-specific credentials.

    NODE_SUPABASE_URL and NODE_SUPABASE_KEY are required; we do not fall back to
    core SUPABASE_* credentials to avoid cross-tenant data access.
    """
    supabase_url = Environment.get_node_supabase_url()
    supabase_key = Environment.get_node_supabase_key()
    schema = Environment.get_node_supabase_schema()

    if supabase_url is None or supabase_key is None:
        raise ValueError(
            "Supabase URL or key is not set. Configure NODE_SUPABASE_URL and NODE_SUPABASE_KEY."
        )

    from supabase import create_async_client

    client = await create_async_client(supabase_url, supabase_key)
    if schema:
        client = client.schema(schema)
    return client


def _apply_filters(query, filters: list[Filter]):
    for field, op, value in filters:
        if op == FilterOp.EQ:
            query = query.eq(field, value)
        elif op == FilterOp.NE:
            query = query.neq(field, value)
        elif op == FilterOp.GT:
            query = query.gt(field, value)
        elif op == FilterOp.GTE:
            query = query.gte(field, value)
        elif op == FilterOp.LT:
            query = query.lt(field, value)
        elif op == FilterOp.LTE:
            query = query.lte(field, value)
        elif op == FilterOp.IN:
            query = query.in_(field, value)
        elif op == FilterOp.LIKE:
            query = query.like(field, value)
        elif op == FilterOp.CONTAINS:
            query = query.contains(field, value)
        else:
            raise ValueError(f"Unsupported operator: {op}")
    return query


class Select(BaseNode):
    """
    Query records from a Supabase table.
    supabase, database, query, select
    """

    _expose_as_tool: ClassVar[bool] = True

    table_name: str = Field(default="", description="Table to query")
    columns: RecordType = Field(default=RecordType(), description="Columns to select")
    filters: list[Filter] = Field(
        default=[], description="List of typed filters to apply"
    )
    order_by: str = Field(default="", description="Column to order by")
    descending: bool = Field(default=False, description="Order direction")
    limit: int = Field(default=0, description="Max rows to return (0 = no limit)")
    to_dataframe: bool = Field(
        default=False, description="Return a DataframeRef instead of list of dicts"
    )

    async def process(self, context: ProcessingContext) -> Any:
        if not self.table_name:
            raise ValueError("table_name cannot be empty")

        client = await _get_supabase_client()
        table_name = _qualify_table_name(self.table_name)

        # Build select columns
        if not self.columns.columns:
            select_columns = "*"
        else:
            select_columns = ", ".join([c.name for c in self.columns.columns])

        query = client.table(table_name).select(select_columns)

        # Filters
        if self.filters:
            query = _apply_filters(query, self.filters)

        # Order + limit
        if self.order_by:
            query = query.order(self.order_by, desc=self.descending)
        if self.limit and self.limit > 0:
            query = query.limit(self.limit)

        resp = await query.execute()
        rows = getattr(resp, "data", None)
        if rows is None:
            # supabase-py might return list directly in some versions
            rows = resp

        if self.to_dataframe:
            import pandas as pd

            df = pd.DataFrame(rows or [])
            return await context.dataframe_from_pandas(df)
        return rows or []


class Insert(BaseNode):
    """
    Insert record(s) into a Supabase table.
    supabase, database, insert, add, record
    """

    _expose_as_tool: ClassVar[bool] = True

    table_name: str = Field(default="", description="Table to insert into")
    records: list[dict[str, Any]] | dict[str, Any] = Field(
        default=[], description="One or multiple rows to insert"
    )
    return_rows: bool = Field(
        default=True, description="Return inserted rows (uses select('*'))"
    )

    async def process(self, context: ProcessingContext) -> Any:
        if not self.table_name:
            raise ValueError("table_name cannot be empty")

        client = await _get_supabase_client()
        table_name = _qualify_table_name(self.table_name)
        data: list[dict[str, Any]]
        if isinstance(self.records, dict):
            data = [self.records]
        else:
            data = self.records

        q = client.table(table_name).insert(data)
        if self.return_rows:
            q = q.select("*") # type: ignore
        resp = await q.execute()
        rows = getattr(resp, "data", None)
        return rows if self.return_rows else {"inserted": len(data)}


class Update(BaseNode):
    """
    Update records in a Supabase table.
    supabase, database, update, modify, change
    """

    _expose_as_tool: ClassVar[bool] = True

    table_name: str = Field(default="", description="Table to update")
    values: dict[str, Any] = Field(default={}, description="New values")
    filters: list[Filter] = Field(
        default=[], description="Filters to select rows to update"
    )
    return_rows: bool = Field(
        default=True, description="Return updated rows (uses select('*'))"
    )

    async def process(self, context: ProcessingContext) -> Any:
        if not self.table_name:
            raise ValueError("table_name cannot be empty")
        if not self.values:
            raise ValueError("values cannot be empty")

        client = await _get_supabase_client()
        table_name = _qualify_table_name(self.table_name)
        q = client.table(table_name).update(self.values)
        if self.filters:
            q = _apply_filters(q, self.filters)
        if self.return_rows:
            q = q.select("*") # type: ignore
        resp = await q.execute()
        rows = getattr(resp, "data", None)
        return rows if self.return_rows else {"updated": True}


class Delete(BaseNode):
    """
    Delete records from a Supabase table.
    supabase, database, delete, remove
    """

    _expose_as_tool: ClassVar[bool] = True

    table_name: str = Field(default="", description="Table to delete from")
    filters: list[Filter] = Field(
        default=[],
        description="Filters to select rows to delete (required for safety)",
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        if not self.table_name:
            raise ValueError("table_name cannot be empty")
        if not self.filters:
            raise ValueError(
                "At least one filter is required for DELETE operations to prevent accidental data loss"
            )

        client = await _get_supabase_client()
        table_name = _qualify_table_name(self.table_name)
        q = client.table(table_name).delete()
        q = _apply_filters(q, self.filters)
        await q.execute()
        return {"deleted": True}


class Upsert(BaseNode):
    """
    Insert or update (upsert) records in a Supabase table.
    supabase, database, upsert, merge
    """

    _expose_as_tool: ClassVar[bool] = True

    table_name: str = Field(default="", description="Table to upsert into")
    records: list[dict[str, Any]] | dict[str, Any] = Field(
        default=[], description="One or multiple rows to upsert"
    )
    return_rows: bool = Field(
        default=True, description="Return upserted rows (uses select('*'))"
    )

    async def process(self, context: ProcessingContext) -> Any:
        if not self.table_name:
            raise ValueError("table_name cannot be empty")

        client = await _get_supabase_client()
        table_name = _qualify_table_name(self.table_name)
        data: list[dict[str, Any]]
        if isinstance(self.records, dict):
            data = [self.records]
        else:
            data = self.records

        q = client.table(table_name).upsert(data)

        if self.return_rows:
            q = q.select("*") # type: ignore
        resp = await q.execute()
        rows = getattr(resp, "data", None)
        return rows if self.return_rows else {"upserted": len(data)}


class RPC(BaseNode):
    """
    Call a PostgreSQL function via Supabase RPC.
    supabase, database, rpc, function
    """

    _expose_as_tool: ClassVar[bool] = True

    function: str = Field(default="", description="RPC function name")
    params: dict[str, Any] = Field(default={}, description="Function params")
    to_dataframe: bool = Field(
        default=False,
        description="Return DataframeRef if result is a list of records",
    )

    async def process(self, context: ProcessingContext) -> Any:
        if not self.function:
            raise ValueError("function cannot be empty")

        client = await _get_supabase_client()
        # supabase-py v2 typically requires .execute() after .rpc()
        resp = await client.rpc(self.function, self.params).execute()
        result = getattr(resp, "data", None)
        if result is None:
            result = resp

        if self.to_dataframe and isinstance(result, list):
            # Only convert list-of-dicts
            import pandas as pd

            df = pd.DataFrame(result)
            return await context.dataframe_from_pandas(df)
        return result
