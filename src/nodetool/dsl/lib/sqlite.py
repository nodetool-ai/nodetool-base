from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode


class SQLiteExecute(GraphNode):
    """
    Execute a SQL statement on a SQLite database without returning rows.
    sqlite, sql, execute, database

    Use cases:
    - Create or modify tables
    - Insert or update records
    - Run maintenance commands
    """

    db_path: types.FilePath | GraphNode | tuple[GraphNode, str] = Field(default=types.FilePath(type='file_path', path=''), description='Path to the SQLite database file')
    statement: str | GraphNode | tuple[GraphNode, str] = Field(default='', description='SQL statement to execute')
    params: list[Any] | GraphNode | tuple[GraphNode, str] = Field(default=PydanticUndefined, description='Optional parameters for the statement')

    @classmethod
    def get_node_type(cls): return "lib.sqlite.SQLiteExecute"



class SQLiteQuery(GraphNode):
    """
    Execute a SQL query on a SQLite database and return the results as a dataframe.
    sqlite, sql, query, database

    Use cases:
    - Run analytics on a local SQLite database
    - Load query results into a dataframe
    - Combine with other data processing nodes
    """

    db_path: types.FilePath | GraphNode | tuple[GraphNode, str] = Field(default=types.FilePath(type='file_path', path=''), description='Path to the SQLite database file')
    query: str | GraphNode | tuple[GraphNode, str] = Field(default='', description='SQL query to execute')
    params: list[Any] | GraphNode | tuple[GraphNode, str] = Field(default=PydanticUndefined, description='Optional parameters for the query')

    @classmethod
    def get_node_type(cls): return "lib.sqlite.SQLiteQuery"


