"""
Supabase CRUD DSL Example

Demonstrates Insert, Update, Delete, and a final Select.

Note:
- Adjust table name and filters to your schema.
- Ensure SUPABASE_URL and SUPABASE_KEY are set.
"""

from nodetool.dsl.graph import create_graph, run_graph
from nodetool.dsl.nodetool.output import DictionaryOutput, DataframeOutput
from nodetool.dsl.lib.supabase import Insert, Update, Delete, Select
from nodetool.nodes.lib.supabase import FilterOp
from nodetool.metadata.types import RecordType, ColumnDef


# Example schema display (optional in Select)
columns = RecordType(
    columns=[
        ColumnDef(name="id", data_type="int"),
        ColumnDef(name="title", data_type="string"),
        ColumnDef(name="done", data_type="int"),
    ]
)

insert = Insert(
    table_name="todos",
    records={"title": "Walk the dog", "done": False},
    return_rows=True,
)

update = Update(
    table_name="todos",
    values={"done": True},
    filters=[("title", FilterOp.EQ, "Walk the dog")],
    return_rows=True,
)

delete = Delete(
    table_name="todos",
    filters=[("title", FilterOp.EQ, "Walk the dog")],
)

select = Select(
    table_name="todos",
    columns=columns,
    order_by="id",
    limit=10,
    to_dataframe=True,
)

out1 = DictionaryOutput(name="insert_result", value=insert.output)
out2 = DictionaryOutput(name="update_result", value=update.output)
out3 = DictionaryOutput(name="delete_result", value=delete.output)
out4 = DataframeOutput(name="todos_df", value=select.output)

graph = create_graph(out1, out2, out3, out4)


if __name__ == "__main__":
    result = run_graph(graph)
    print("Insert result:", result["insert_result"])  # list of rows
    print("Update result:", result["update_result"])  # list of rows
    print("Delete result:", result["delete_result"])  # {deleted: True}
    print("Todos:", result["todos_df"])  # DataframeRef

