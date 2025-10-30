"""
CSV Data Processor DSL Example

Ingest, clean, and export sales pipeline data stored in CSV files.

Workflow:
1. **Seed Workspace** – Write a sample sales CSV file into the NodeTool workspace
2. **Filter Records** – Keep only completed deals above a revenue threshold
3. **Map Columns** – Rename and select the most relevant columns for reporting
4. **Export Outputs** – Save the refined table and expose a priority client list
"""

from nodetool.dsl.graph import create_graph, run_graph
from nodetool.dsl.nodetool.workspace import WriteTextFile
from nodetool.dsl.nodetool.data import (
    LoadCSVFile,
    Filter,
    Rename,
    SelectColumn,
    SaveCSVDataframeFile,
    ToList,
)
from nodetool.dsl.nodetool.list import MapField
from nodetool.dsl.nodetool.output import DataframeOutput, ListOutput


# --- Workspace setup ---------------------------------------------------------
sales_seed_file = WriteTextFile(
    path="sales_pipeline.csv",
    content=(
        "company,region,status,revenue\n"
        "Acme Rockets,North America,Completed,12500\n"
        "Beacon Analytics,Europe,Prospecting,4200\n"
        "Cascade Systems,North America,Completed,9800\n"
        "Delta Freight,Asia Pacific,Completed,4700\n"
        "Evergreen Labs,Europe,Completed,15750\n"
        "Futura Robotics,Latin America,Prospecting,3100\n"
        "Glide Solar,North America,Completed,6200\n"
    ),
)


# --- Dataframe transformations -----------------------------------------------
raw_sales = LoadCSVFile(file_path=sales_seed_file.output)
qualified_sales = Filter(
    df=raw_sales.output,
    condition="status == 'Completed' and revenue >= 6000",
)
renamed_sales = Rename(
    dataframe=qualified_sales.output,
    rename_map="company:client_name,revenue:total_revenue",
)
selected_columns = SelectColumn(
    dataframe=renamed_sales.output,
    columns="client_name,region,total_revenue",
)

export_csv = SaveCSVDataframeFile(
    dataframe=selected_columns.output,
    filename="qualified_sales.csv",
)

sales_records = ToList(dataframe=selected_columns.output)
priority_clients = MapField(values=sales_records.output, field="client_name")


# --- Workflow outputs --------------------------------------------------------
clean_table_output = DataframeOutput(
    name="qualified_sales_table",
    value=export_csv.output,
)
client_list_output = ListOutput(
    name="priority_clients",
    value=priority_clients.output,
)


graph = create_graph(clean_table_output, client_list_output)


if __name__ == "__main__":
    result = run_graph(graph)
    print(
        "Qualified sales table rows:",
        result["qualified_sales_table"],
    )
    print("Priority clients:", result["priority_clients"])
