"""
SQLite database nodes for persistent storage and memory mechanisms.
"""

import sqlite3
import json
from typing import ClassVar, TypedDict, AsyncGenerator
from pathlib import Path
from typing import Any
from pydantic import Field
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.metadata.types import RecordType, DataframeRef
from nodetool.config.logging_config import get_logger


log = get_logger(__name__)


def column_type_to_sqlite(column_type: str) -> str:
    """Convert ColumnDef data_type to SQLite type"""
    mapping = {
        "int": "INTEGER",
        "float": "REAL",
        "datetime": "TEXT",
        "string": "TEXT",
        "object": "TEXT"
    }
    return mapping.get(column_type, "TEXT")


class CreateTable(BaseNode):
    """
    Create a new SQLite table with specified columns.
    sqlite, database, table, create, schema

    Use cases:
    - Initialize database schema for flashcards
    - Set up tables for persistent storage
    - Create memory structures for agents
    """

    database_name: str = Field(
        default="memory.db",
        description="Name of the SQLite database file"
    )
    table_name: str = Field(
        default="flashcards",
        description="Name of the table to create"
    )
    columns: RecordType = Field(
        default=RecordType(),
        description="Column definitions"
    )
    add_primary_key: bool = Field(
        default=True,
        description="Automatically make first integer column PRIMARY KEY AUTOINCREMENT"
    )
    if_not_exists: bool = Field(
        default=True,
        description="Only create table if it doesn't exist"
    )

    _expose_as_tool: ClassVar[bool] = True

    class OutputType(TypedDict):
        database_name: str
        table_name: str
        columns: RecordType

    async def process(self, context: ProcessingContext) -> OutputType:
        db_path = Path(context.workspace_dir) / self.database_name
        log.info(f"Creating table {self.table_name} in database {self.database_name} at {db_path}")

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        table_exists = conn.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{self.table_name}'").fetchone() is not None
        if table_exists:
            return {
                "database_name": self.database_name,
                "table_name": self.table_name,
                "columns": self.columns
            }
        try:
            cursor = conn.cursor()

            column_defs = []
            for i, col in enumerate(self.columns.columns):
                sqlite_type = column_type_to_sqlite(col.data_type)

                # Make first int column a primary key if requested
                if i == 0 and self.add_primary_key and col.data_type == "int":
                    column_defs.append(f"{col.name} INTEGER PRIMARY KEY AUTOINCREMENT")
                else:
                    column_defs.append(f"{col.name} {sqlite_type}")

            columns_sql = ", ".join(column_defs)
            if_not_exists_clause = "IF NOT EXISTS " if self.if_not_exists else ""
            sql = f"CREATE TABLE {if_not_exists_clause}{self.table_name} ({columns_sql})"

            cursor.execute(sql)
            conn.commit()

            return {
                "database_name": self.database_name,
                "table_name": self.table_name,
                "columns": self.columns
            }
        finally:
            conn.close()


class Insert(BaseNode):
    """
    Insert a record into a SQLite table.
    sqlite, database, insert, add, record

    Use cases:
    - Add new flashcards to database
    - Store agent observations
    - Persist workflow results
    """

    _expose_as_tool: ClassVar[bool] = True

    database_name: str = Field(
        default="memory.db",
        description="Name of the SQLite database file"
    )
    table_name: str = Field(
        default="flashcards",
        description="Name of the table to insert into"
    )
    data: dict[str, Any] = Field(
        default={"content": "example"},
        description="Data to insert as dict (column: value)"
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        db_path = Path(context.workspace_dir) / self.database_name

        conn = sqlite3.connect(db_path)
        log.info(f"Inserting record into table {self.table_name} in database {self.database_name} at {db_path}")
        try:
            cursor = conn.cursor()

            columns = ", ".join(self.data.keys())
            placeholders = ", ".join(["?" for _ in self.data.values()])
            sql = f"INSERT INTO {self.table_name} ({columns}) VALUES ({placeholders})"

            values = []
            for v in self.data.values():
                if isinstance(v, (dict, list)):
                    values.append(json.dumps(v))
                else:
                    values.append(v)

            cursor.execute(sql, values)
            conn.commit()

            return {
                "row_id": cursor.lastrowid,
                "rows_affected": cursor.rowcount,
                "message": f"Inserted record with ID {cursor.lastrowid}"
            }
        finally:
            conn.close()


class Query(BaseNode):
    """
    Query records from a SQLite table.
    sqlite, database, query, select, search, retrieve

    Use cases:
    - Retrieve flashcards for review
    - Search agent memory
    - Fetch stored data
    """

    _expose_as_tool: ClassVar[bool] = True

    database_name: str = Field(
        default="memory.db",
        description="Name of the SQLite database file"
    )
    table_name: str = Field(
        default="flashcards",
        description="Name of the table to query"
    )
    where: str = Field(
        default="",
        description="WHERE clause (without 'WHERE' keyword), e.g., 'id = 1'"
    )
    columns: RecordType = Field(
        default=RecordType(),
        description="Columns to select"
    )
    order_by: str = Field(
        default="",
        description="ORDER BY clause (without 'ORDER BY' keyword)"
    )
    limit: int = Field(
        default=0,
        description="Maximum number of rows to return (0 = no limit)"
    )

    async def process(self, context: ProcessingContext) -> list[dict[str, Any]]:
        db_path = Path(context.workspace_dir) / self.database_name
        log.info(f"Querying table {self.table_name} in database {self.database_name} at {db_path}")

        if not db_path.exists():
            return []

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.cursor()

            # Use SELECT * if no columns specified, otherwise use specified columns
            if not self.columns.columns:
                columns = "*"
            else:
                columns = ", ".join([f"{col.name}" for col in self.columns.columns])

            sql = f"SELECT {columns} FROM {self.table_name}"

            if self.where:
                sql += f" WHERE {self.where}"

            if self.order_by:
                sql += f" ORDER BY {self.order_by}"

            if self.limit > 0:
                sql += f" LIMIT {self.limit}"

            cursor.execute(sql)

            results = []
            for row in cursor.fetchall():
                row_dict = dict(row)
                # Try to parse JSON values
                for key, value in row_dict.items():
                    if isinstance(value, str):
                        try:
                            row_dict[key] = json.loads(value)
                        except (json.JSONDecodeError, TypeError):
                            pass
                results.append(row_dict)

            return results
        except Exception as e:
            import traceback
            log.error(traceback.print_stack())
            log.error(f"Error querying table {self.table_name} in database {self.database_name} at {db_path}: {e}")
            raise e
        finally:
            conn.close()


class Update(BaseNode):
    """
    Update records in a SQLite table.
    sqlite, database, update, modify, change

    Use cases:
    - Update flashcard content
    - Modify stored records
    - Change agent memory
    """

    _expose_as_tool: ClassVar[bool] = True

    database_name: str = Field(
        default="memory.db",
        description="Name of the SQLite database file"
    )
    table_name: str = Field(
        default="flashcards",
        description="Name of the table to update"
    )
    data: dict[str, Any] = Field(
        default={"content": "updated"},
        description="Data to update as dict (column: new_value)"
    )
    where: str = Field(
        default="",
        description="WHERE clause (without 'WHERE' keyword), e.g., 'id = 1'"
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        db_path = Path(context.workspace_dir) / self.database_name
        log.info(f"Updating table {self.table_name} in database {self.database_name} at {db_path}")

        conn = sqlite3.connect(db_path)
        try:
            cursor = conn.cursor()

            set_clause = ", ".join([f"{col} = ?" for col in self.data.keys()])
            sql = f"UPDATE {self.table_name} SET {set_clause}"

            if self.where:
                sql += f" WHERE {self.where}"

            values = []
            for v in self.data.values():
                if isinstance(v, (dict, list)):
                    values.append(json.dumps(v))
                else:
                    values.append(v)

            cursor.execute(sql, values)
            conn.commit()

            return {
                "rows_affected": cursor.rowcount,
                "message": f"Updated {cursor.rowcount} record(s)"
            }
        finally:
            conn.close()


class Delete(BaseNode):
    """
    Delete records from a SQLite table.
    sqlite, database, delete, remove, drop

    Use cases:
    - Remove flashcards
    - Delete agent memory
    - Clean up old data
    """

    _expose_as_tool: ClassVar[bool] = True

    database_name: str = Field(
        default="memory.db",
        description="Name of the SQLite database file"
    )
    table_name: str = Field(
        default="flashcards",
        description="Name of the table to delete from"
    )
    where: str = Field(
        default="",
        description="WHERE clause (without 'WHERE' keyword), e.g., 'id = 1'. REQUIRED for safety."
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        if not self.where:
            raise ValueError("WHERE clause is required for DELETE operations to prevent accidental data loss")

        db_path = Path(context.workspace_dir) / self.database_name

        conn = sqlite3.connect(db_path)
        try:
            cursor = conn.cursor()

            sql = f"DELETE FROM {self.table_name} WHERE {self.where}"

            cursor.execute(sql)
            conn.commit()

            return {
                "rows_affected": cursor.rowcount,
                "message": f"Deleted {cursor.rowcount} record(s)"
            }
        finally:
            conn.close()


class ExecuteSQL(BaseNode):
    """
    Execute arbitrary SQL statements for advanced operations.
    sqlite, database, sql, execute, custom

    Use cases:
    - Complex queries with joins
    - Aggregate functions (COUNT, SUM, AVG)
    - Custom SQL operations
    """

    _expose_as_tool: ClassVar[bool] = True

    database_name: str = Field(
        default="memory.db",
        description="Name of the SQLite database file"
    )
    sql: str = Field(
        default="SELECT * FROM flashcards",
        description="SQL statement to execute"
    )
    parameters: list[Any] = Field(
        default=[],
        description="Parameters for parameterized queries (use ? in SQL)"
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        db_path = Path(context.workspace_dir) / self.database_name

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.cursor()

            cursor.execute(self.sql, self.parameters)

            if self.sql.strip().upper().startswith(('INSERT', 'UPDATE', 'DELETE', 'CREATE', 'ALTER', 'DROP')):
                conn.commit()
                return {
                    "rows_affected": cursor.rowcount,
                    "last_row_id": cursor.lastrowid,
                    "message": "SQL executed successfully"
                }
            else:
                rows = cursor.fetchall()
                results = []
                for row in rows:
                    row_dict = dict(row)
                    for key, value in row_dict.items():
                        if isinstance(value, str):
                            try:
                                row_dict[key] = json.loads(value)
                            except (json.JSONDecodeError, TypeError):
                                pass
                    results.append(row_dict)

                return {
                    "rows": results,
                    "count": len(results),
                    "message": f"Query returned {len(results)} row(s)"
                }
        finally:
            conn.close()


class GetDatabasePath(BaseNode):
    """
    Get the full path to a SQLite database file.
    sqlite, database, path, location

    Use cases:
    - Reference database location
    - Verify database exists
    - Pass path to external tools
    """

    _expose_as_tool: ClassVar[bool] = True

    database_name: str = Field(
        default="memory.db",
        description="Name of the SQLite database file"
    )

    async def process(self, context: ProcessingContext) -> str:
        db_path = Path(context.workspace_dir) / self.database_name
        return str(db_path)
