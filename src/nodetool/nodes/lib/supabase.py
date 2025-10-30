"""
Supabase database nodes mirroring the SQLite nodes' features.

These nodes provide typed, safe access to Supabase tables for:
- Select (query with filters, order, limit)
- Insert (single or multiple records)
- Update (with optional filters)
- Delete (requires filters for safety)

Authentication and configuration are handled via nodetool-core Environment:
- SUPABASE_URL, SUPABASE_KEY

When Supabase is configured, existing asset save/load nodes will transparently
use Supabase Storage via Environment; these DB nodes focus purely on table CRUD.
"""

from typing import Any, ClassVar, Literal
from enum import Enum

import pandas as pd
from pydantic import BaseModel, Field

from nodetool.config.environment import Environment
from nodetool.metadata.types import DataframeRef, RecordType
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
        default_factory=list, description="List of typed filters to apply"
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

        client = await Environment.get_supabase_client()

        # Build select columns
        if not self.columns.columns:
            select_columns = "*"
        else:
            select_columns = ", ".join([c.name for c in self.columns.columns])

        query = client.table(self.table_name).select(select_columns)

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
        default_factory=list, description="One or multiple rows to insert"
    )
    return_rows: bool = Field(
        default=True, description="Return inserted rows (uses select('*'))"
    )

    async def process(self, context: ProcessingContext) -> Any:
        if not self.table_name:
            raise ValueError("table_name cannot be empty")

        client = await Environment.get_supabase_client()
        data: list[dict[str, Any]]
        if isinstance(self.records, dict):
            data = [self.records]
        else:
            data = self.records

        q = client.table(self.table_name).insert(data)
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
    values: dict[str, Any] = Field(default_factory=dict, description="New values")
    filters: list[Filter] = Field(
        default_factory=list, description="Filters to select rows to update"
    )
    return_rows: bool = Field(
        default=True, description="Return updated rows (uses select('*'))"
    )

    async def process(self, context: ProcessingContext) -> Any:
        if not self.table_name:
            raise ValueError("table_name cannot be empty")
        if not self.values:
            raise ValueError("values cannot be empty")

        client = await Environment.get_supabase_client()
        q = client.table(self.table_name).update(self.values)
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
        default_factory=list,
        description="Filters to select rows to delete (required for safety)",
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        if not self.table_name:
            raise ValueError("table_name cannot be empty")
        if not self.filters:
            raise ValueError(
                "At least one filter is required for DELETE operations to prevent accidental data loss"
            )

        client = await Environment.get_supabase_client()
        q = client.table(self.table_name).delete()
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
        default_factory=list, description="One or multiple rows to upsert"
    )
    on_conflict: str | None = Field(
        default=None,
        description="Optional column or comma-separated columns for ON CONFLICT",
    )
    return_rows: bool = Field(
        default=True, description="Return upserted rows (uses select('*'))"
    )

    async def process(self, context: ProcessingContext) -> Any:
        if not self.table_name:
            raise ValueError("table_name cannot be empty")

        client = await Environment.get_supabase_client()
        data: list[dict[str, Any]]
        if isinstance(self.records, dict):
            data = [self.records]
        else:
            data = self.records

        # Attempt to pass on_conflict if supported by the client
        if self.on_conflict is not None:
            q = client.table(self.table_name).upsert(data, on_conflict=self.on_conflict)
        else:
            q = client.table(self.table_name).upsert(data)

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
    params: dict[str, Any] = Field(default_factory=dict, description="Function params")
    to_dataframe: bool = Field(
        default=False,
        description="Return DataframeRef if result is a list of records",
    )

    async def process(self, context: ProcessingContext) -> Any:
        if not self.function:
            raise ValueError("function cannot be empty")

        client = await Environment.get_supabase_client()
        # supabase-py v2 typically requires .execute() after .rpc()
        resp = await client.rpc(self.function, self.params).execute()
        result = getattr(resp, "data", None)
        if result is None:
            result = resp

        if self.to_dataframe and isinstance(result, list):
            # Only convert list-of-dicts
            df = pd.DataFrame(result)
            return await context.dataframe_from_pandas(df)
        return result
