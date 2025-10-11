import pytest
from pathlib import Path
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.metadata.types import ColumnDef, RecordType
from nodetool.nodes.lib.sqlite import (
    CreateTable,
    Insert,
    Query,
    Update,
    Delete,
    ExecuteSQL,
    GetDatabasePath,
)


@pytest.fixture
def context():
    return ProcessingContext(user_id="test", auth_token="test")


@pytest.mark.asyncio
async def test_create_table(context: ProcessingContext):
    """Test creating a basic table"""
    node = CreateTable(
        database_name="test.db",
        table_name="flashcards",
        columns=RecordType(columns=[
            ColumnDef(name="id", data_type="int"),
            ColumnDef(name="question", data_type="string"),
            ColumnDef(name="answer", data_type="string"),
        ]),
    )
    result = await node.process(context)
    assert "created successfully" in result
    assert "flashcards" in result


@pytest.mark.asyncio
async def test_insert_and_query(context: ProcessingContext):
    """Test inserting and querying data"""
    # Create table
    create = CreateTable(
        database_name="test.db",
        table_name="flashcards",
        columns=RecordType(columns=[
            ColumnDef(name="id", data_type="int"),
            ColumnDef(name="question", data_type="string"),
            ColumnDef(name="answer", data_type="string"),
        ]),
    )
    await create.process(context)

    # Insert data
    insert = Insert(
        database_name="test.db",
        table_name="flashcards",
        data={"question": "What is 2+2?", "answer": "4"},
    )
    insert_result = await insert.process(context)
    assert insert_result["row_id"] > 0
    assert insert_result["rows_affected"] == 1

    # Query data
    query = Query(
        database_name="test.db",
        table_name="flashcards",
    )
    query_result = await query.process(context)
    assert len(query_result) == 1
    assert query_result[0]["question"] == "What is 2+2?"
    # JSON parsing converts numeric strings, so "4" becomes 4
    assert str(query_result[0]["answer"]) == "4"


@pytest.mark.asyncio
async def test_query_with_where(context: ProcessingContext):
    """Test querying with WHERE clause"""
    # Setup
    create = CreateTable(
        database_name="test.db",
        table_name="items",
        columns=RecordType(columns=[
            ColumnDef(name="id", data_type="int"),
            ColumnDef(name="name", data_type="string"),
            ColumnDef(name="value", data_type="int"),
        ]),
    )
    await create.process(context)

    # Insert multiple items
    for i, name in enumerate(["apple", "banana", "cherry"], 1):
        insert = Insert(
            database_name="test.db",
            table_name="items",
            data={"name": name, "value": i * 10},
        )
        await insert.process(context)

    # Query with WHERE
    query = Query(
        database_name="test.db",
        table_name="items",
        where="value > 10",
    )
    result = await query.process(context)
    assert len(result) == 2
    assert all(r["value"] > 10 for r in result)


@pytest.mark.asyncio
async def test_query_with_order_and_limit(context: ProcessingContext):
    """Test querying with ORDER BY and LIMIT"""
    # Setup
    create = CreateTable(
        database_name="test.db",
        table_name="scores",
        columns=RecordType(columns=[
            ColumnDef(name="id", data_type="int"),
            ColumnDef(name="score", data_type="int"),
        ]),
    )
    await create.process(context)

    # Insert scores
    for score in [100, 50, 75, 90]:
        insert = Insert(
            database_name="test.db",
            table_name="scores",
            data={"score": score},
        )
        await insert.process(context)

    # Query with ORDER BY and LIMIT
    query = Query(
        database_name="test.db",
        table_name="scores",
        order_by="score DESC",
        limit=2,
    )
    result = await query.process(context)
    assert len(result) == 2
    assert result[0]["score"] == 100
    assert result[1]["score"] == 90


@pytest.mark.asyncio
async def test_update(context: ProcessingContext):
    """Test updating records"""
    # Setup
    create = CreateTable(
        database_name="test.db",
        table_name="users",
        columns=RecordType(columns=[
            ColumnDef(name="id", data_type="int"),
            ColumnDef(name="name", data_type="string"),
        ]),
    )
    await create.process(context)

    insert = Insert(
        database_name="test.db",
        table_name="users",
        data={"name": "Alice"},
    )
    insert_result = await insert.process(context)
    row_id = insert_result["row_id"]

    # Update
    update = Update(
        database_name="test.db",
        table_name="users",
        data={"name": "Bob"},
        where=f"id = {row_id}",
    )
    update_result = await update.process(context)
    assert update_result["rows_affected"] == 1

    # Verify update
    query = Query(
        database_name="test.db",
        table_name="users",
        where=f"id = {row_id}",
    )
    result = await query.process(context)
    assert result[0]["name"] == "Bob"


@pytest.mark.asyncio
async def test_delete(context: ProcessingContext):
    """Test deleting records"""
    # Setup
    create = CreateTable(
        database_name="test.db",
        table_name="temp",
        columns=RecordType(columns=[
            ColumnDef(name="id", data_type="int"),
            ColumnDef(name="value", data_type="string"),
        ]),
    )
    await create.process(context)

    # Insert data
    insert = Insert(
        database_name="test.db",
        table_name="temp",
        data={"value": "delete me"},
    )
    insert_result = await insert.process(context)
    row_id = insert_result["row_id"]

    # Delete
    delete = Delete(
        database_name="test.db",
        table_name="temp",
        where=f"id = {row_id}",
    )
    delete_result = await delete.process(context)
    assert delete_result["rows_affected"] == 1

    # Verify deletion
    query = Query(
        database_name="test.db",
        table_name="temp",
    )
    result = await query.process(context)
    assert len(result) == 0


@pytest.mark.asyncio
async def test_delete_requires_where(context: ProcessingContext):
    """Test that DELETE requires WHERE clause for safety"""
    delete = Delete(
        database_name="test.db",
        table_name="temp",
        where="",
    )
    with pytest.raises(ValueError, match="WHERE clause is required"):
        await delete.process(context)


@pytest.mark.asyncio
async def test_json_serialization(context: ProcessingContext):
    """Test that dict and list values are JSON serialized"""
    # Setup
    create = CreateTable(
        database_name="test.db",
        table_name="json_test",
        columns=RecordType(columns=[
            ColumnDef(name="id", data_type="int"),
            ColumnDef(name="data", data_type="object"),
        ]),
    )
    await create.process(context)

    # Insert with dict
    insert = Insert(
        database_name="test.db",
        table_name="json_test",
        data={"data": {"nested": {"key": "value"}, "list": [1, 2, 3]}},
    )
    await insert.process(context)

    # Query back
    query = Query(
        database_name="test.db",
        table_name="json_test",
    )
    result = await query.process(context)
    assert result[0]["data"] == {"nested": {"key": "value"}, "list": [1, 2, 3]}


@pytest.mark.asyncio
async def test_execute_sql_select(context: ProcessingContext):
    """Test ExecuteSQL for SELECT queries"""
    # Setup
    create = CreateTable(
        database_name="test.db",
        table_name="numbers",
        columns=RecordType(columns=[
            ColumnDef(name="id", data_type="int"),
            ColumnDef(name="value", data_type="int"),
        ]),
    )
    await create.process(context)

    # Insert data
    for i in range(1, 6):
        insert = Insert(
            database_name="test.db",
            table_name="numbers",
            data={"value": i * 10},
        )
        await insert.process(context)

    # Execute custom SQL
    execute = ExecuteSQL(
        database_name="test.db",
        sql="SELECT COUNT(*) as count, SUM(value) as total FROM numbers",
    )
    result = await execute.process(context)
    # Result includes both count field and rows array
    assert len(result["rows"]) > 0
    assert result["rows"][0]["count"] == 5
    assert result["rows"][0]["total"] == 150


@pytest.mark.asyncio
async def test_execute_sql_insert(context: ProcessingContext):
    """Test ExecuteSQL for INSERT operations"""
    # Setup
    create = CreateTable(
        database_name="test.db",
        table_name="test_table",
        columns=RecordType(columns=[
            ColumnDef(name="id", data_type="int"),
            ColumnDef(name="name", data_type="string"),
        ]),
    )
    await create.process(context)

    # Execute INSERT
    execute = ExecuteSQL(
        database_name="test.db",
        sql="INSERT INTO test_table (name) VALUES (?)",
        parameters=["test"],
    )
    result = await execute.process(context)
    assert result["rows_affected"] == 1
    assert result["last_row_id"] > 0


@pytest.mark.asyncio
async def test_get_database_path(context: ProcessingContext):
    """Test getting database path"""
    node = GetDatabasePath(database_name="test.db")
    path = await node.process(context)
    assert isinstance(path, str)
    assert "test.db" in path


@pytest.mark.asyncio
async def test_query_nonexistent_database(context: ProcessingContext):
    """Test querying non-existent database returns empty list"""
    query = Query(
        database_name="nonexistent.db",
        table_name="any_table",
    )
    result = await query.process(context)
    assert result == []
