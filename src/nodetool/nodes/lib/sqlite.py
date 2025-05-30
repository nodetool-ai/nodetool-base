import sqlite3
import pandas as pd
from typing import Any
from pydantic import Field
from nodetool.metadata.types import DataframeRef, FilePath
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext


class SQLiteQuery(BaseNode):
    """
    Execute a SQL query on a SQLite database and return the results as a dataframe.
    sqlite, sql, query, database

    Use cases:
    - Run analytics on a local SQLite database
    - Load query results into a dataframe
    - Combine with other data processing nodes
    """

    db_path: FilePath = Field(
        default=FilePath(), description="Path to the SQLite database file"
    )
    query: str = Field(default="", description="SQL query to execute")
    params: list[Any] = Field(
        default_factory=list, description="Optional parameters for the query"
    )

    async def process(self, context: ProcessingContext) -> DataframeRef:
        conn = sqlite3.connect(self.db_path.path)
        df = pd.read_sql_query(self.query, conn, params=self.params)
        conn.close()
        return await context.dataframe_from_pandas(df)


class SQLiteExecute(BaseNode):
    """
    Execute a SQL statement on a SQLite database without returning rows.
    sqlite, sql, execute, database

    Use cases:
    - Create or modify tables
    - Insert or update records
    - Run maintenance commands
    """

    db_path: FilePath = Field(
        default=FilePath(), description="Path to the SQLite database file"
    )
    statement: str = Field(default="", description="SQL statement to execute")
    params: list[Any] = Field(
        default_factory=list, description="Optional parameters for the statement"
    )

    @classmethod
    def return_type(cls):
        return {"rowcount": int}

    async def process(self, context: ProcessingContext):
        conn = sqlite3.connect(self.db_path.path)
        cur = conn.cursor()
        cur.execute(self.statement, self.params)
        conn.commit()
        rowcount = cur.rowcount
        conn.close()
        return {"rowcount": rowcount}
