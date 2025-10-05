from datetime import datetime
from io import StringIO
from typing import AsyncGenerator, ClassVar, TypedDict
import json
import os
import pandas as pd
from typing import Any, ClassVar
from pydantic import Field
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.metadata.types import ColumnDef, DataframeRef, FolderRef, RecordType
from nodetool.config.environment import Environment


class Schema(BaseNode):
    """
    Define a schema for a dataframe.
    schema, dataframe, create
    """
    
    columns: RecordType = Field(
        default=RecordType(),
        description="The columns to use in the dataframe.",
    )

    async def process(self, context: ProcessingContext) -> RecordType:
        return self.columns
    

class Filter(BaseNode):
    """
    Filter dataframe based on condition.
    filter, query, condition

    Example conditions:
    age > 30
    age > 30 and salary < 50000
    name == 'John Doe'
    100 <= price <= 200
    status in ['Active', 'Pending']
    not (age < 18)

    Use cases:
    - Extract subset of data meeting specific criteria
    - Remove outliers or invalid data points
    - Focus analysis on relevant data segments
    """

    _expose_as_tool: ClassVar[bool] = True

    df: DataframeRef = Field(
        default=DataframeRef(), description="The DataFrame to filter."
    )
    condition: str = Field(
        default="",
        description="The filtering condition to be applied to the DataFrame, e.g. column_name > 5.",
    )

    async def process(self, context: ProcessingContext) -> DataframeRef:
        df = await context.dataframe_to_pandas(self.df)
        res = df.query(self.condition)
        return await context.dataframe_from_pandas(res)


class Slice(BaseNode):
    """
    Slice a dataframe by rows using start and end indices.
    slice, subset, rows

    Use cases:
    - Extract a specific range of rows from a large dataset
    - Create training and testing subsets for machine learning
    - Analyze data in smaller chunks
    """

    _expose_as_tool: ClassVar[bool] = True

    dataframe: DataframeRef = Field(
        default=DataframeRef(), description="The input dataframe to be sliced."
    )
    start_index: int = Field(
        default=0, description="The starting index of the slice (inclusive)."
    )
    end_index: int = Field(
        default=-1,
        description="The ending index of the slice (exclusive). Use -1 for the last row.",
    )

    async def process(self, context: ProcessingContext) -> DataframeRef:
        df = await context.dataframe_to_pandas(self.dataframe)

        if self.end_index == -1:
            self.end_index = len(df)

        sliced_df = df.iloc[self.start_index : self.end_index]
        return await context.dataframe_from_pandas(sliced_df)


class SaveDataframe(BaseNode):
    """
    Save dataframe in specified folder.
    csv, folder, save

    Use cases:
    - Export processed data for external use
    - Create backups of dataframes
    """

    _expose_as_tool: ClassVar[bool] = True
    df: DataframeRef = DataframeRef()
    folder: FolderRef = Field(
        default=FolderRef(), description="Name of the output folder."
    )
    name: str = Field(
        default="output.csv",
        description="""
        Name of the output file.
        You can use time and date variables to create unique names:
        %Y - Year
        %m - Month
        %d - Day
        %H - Hour
        %M - Minute
        %S - Second
        """,
    )

    def required_inputs(self):
        return ["df"]

    async def process(self, context: ProcessingContext) -> DataframeRef:
        df = await context.dataframe_to_pandas(self.df)
        parent_id = self.folder.asset_id if self.folder.is_set() else None
        filename = datetime.now().strftime(self.name)
        return await context.dataframe_from_pandas(df, filename, parent_id)


class ImportCSV(BaseNode):
    """
    Convert CSV string to dataframe.
    csv, dataframe, import

    Use cases:
    - Import CSV data from string input
    - Convert CSV responses from APIs to dataframe
    """

    _expose_as_tool: ClassVar[bool] = True
    csv_data: str = Field(
        default="", title="CSV Data", description="String input of CSV formatted text."
    )

    async def process(self, context: ProcessingContext) -> DataframeRef:
        df = pd.read_csv(StringIO(self.csv_data))
        return await context.dataframe_from_pandas(df)


class LoadCSVURL(BaseNode):
    """
    Load CSV file from URL.
    csv, dataframe, import
    """

    _expose_as_tool: ClassVar[bool] = True
    url: str = Field(default="", description="The URL of the CSV file to load.")

    async def process(self, context: ProcessingContext) -> DataframeRef:
        df = pd.read_csv(self.url)
        return await context.dataframe_from_pandas(df)


class LoadCSVFile(BaseNode):
    """
    Load CSV file from file path.
    csv, dataframe, import
    """

    _expose_as_tool: ClassVar[bool] = True
    file_path: str = Field(default="", description="The path to the CSV file to load.")

    async def process(self, context: ProcessingContext) -> DataframeRef:
        if not self.file_path:
            raise ValueError("file_path cannot be empty")
        df = pd.read_csv(self.file_path)
        return await context.dataframe_from_pandas(df)


class FromList(BaseNode):
    """
    Convert list of dicts to dataframe.
    list, dataframe, convert

    Use cases:
    - Transform list data into structured dataframe
    - Prepare list data for analysis or visualization
    - Convert API responses to dataframe format
    """

    _expose_as_tool: ClassVar[bool] = True
    values: list[Any] = Field(
        title="Values",
        default=[],
        description="List of values to be converted, each value will be a row.",
    )

    async def process(self, context: ProcessingContext) -> DataframeRef:
        rows = []
        for value in self.values:
            if not isinstance(value, dict):
                raise ValueError("List must contain dicts.")
            row = {}
            for k, v in value.items():
                if isinstance(v, dict):
                    row[k] = v["value"]
                elif isinstance(v, (int, float, str, bool)):
                    row[k] = v
                else:
                    row[k] = str(v)
            rows.append(row)
        df = pd.DataFrame(rows)
        return await context.dataframe_from_pandas(df)


class JSONToDataframe(BaseNode):
    """
    Transforms a JSON string into a pandas DataFrame.
    json, dataframe, conversion

    Use cases:
    - Converting API responses to tabular format
    - Preparing JSON data for analysis or visualization
    - Structuring unstructured JSON data for further processing
    """

    _expose_as_tool: ClassVar[bool] = True
    text: str = Field(title="JSON", default="")

    @classmethod
    def get_title(cls):
        return "Convert JSON to DataFrame"

    async def process(self, context: ProcessingContext) -> DataframeRef:
        rows = json.loads(self.text)
        df = pd.DataFrame(rows)
        return await context.dataframe_from_pandas(df)


class ToList(BaseNode):
    """
    Convert dataframe to list of dictionaries.
    dataframe, list, convert

    Use cases:
    - Convert dataframe data for API consumption
    - Transform data for JSON serialization
    - Prepare data for document-based storage
    """

    _expose_as_tool: ClassVar[bool] = True
    dataframe: DataframeRef = Field(
        default=DataframeRef(), description="The input dataframe to convert."
    )

    async def process(self, context: ProcessingContext) -> list[dict]:
        df = await context.dataframe_to_pandas(self.dataframe)
        return df.to_dict("records")


class SelectColumn(BaseNode):
    """
    Select specific columns from dataframe.
    dataframe, columns, filter

    Use cases:
    - Extract relevant features for analysis
    - Reduce dataframe size by removing unnecessary columns
    - Prepare data for specific visualizations or models
    """

    _expose_as_tool: ClassVar[bool] = True
    dataframe: DataframeRef = Field(
        default=DataframeRef(),
        description="a dataframe from which columns are to be selected",
    )
    columns: str = Field("", description="comma separated list of column names")

    async def process(self, context: ProcessingContext) -> DataframeRef:
        columns = self.columns.split(",")
        df = await context.dataframe_to_pandas(self.dataframe)
        return await context.dataframe_from_pandas(df[columns])  # type: ignore


class ExtractColumn(BaseNode):
    """
    Convert dataframe column to list.
    dataframe, column, list

    Use cases:
    - Extract data for use in other processing steps
    - Prepare column data for plotting or analysis
    - Convert categorical data to list for encoding
    """

    _expose_as_tool: ClassVar[bool] = True
    dataframe: DataframeRef = Field(
        default=DataframeRef(), description="The input dataframe."
    )
    column_name: str = Field(
        default="", description="The name of the column to be converted to a list."
    )

    async def process(self, context: ProcessingContext) -> list[Any]:
        df = await context.dataframe_to_pandas(self.dataframe)
        return df[self.column_name].tolist()


class AddColumn(BaseNode):
    """
    Add list of values as new column to dataframe.
    dataframe, column, list

    Use cases:
    - Incorporate external data into existing dataframe
    - Add calculated results as new column
    - Augment dataframe with additional features
    """

    _expose_as_tool: ClassVar[bool] = True
    dataframe: DataframeRef = Field(
        default=DataframeRef(),
        description="Dataframe object to add a new column to.",
    )
    column_name: str = Field(
        default="",
        description="The name of the new column to be added to the dataframe.",
    )
    values: list[Any] = Field(
        default=[],
        description="A list of any type of elements which will be the new column's values.",
    )

    async def process(self, context: ProcessingContext) -> DataframeRef:
        df = await context.dataframe_to_pandas(self.dataframe)
        df[self.column_name] = self.values
        return await context.dataframe_from_pandas(df)


class Merge(BaseNode):
    """
    Merge two dataframes along columns.
    merge, concat, columns

    Use cases:
    - Combine data from multiple sources
    - Add new features to existing dataframe
    - Merge time series data from different periods
    """

    _expose_as_tool: ClassVar[bool] = True
    dataframe_a: DataframeRef = Field(
        default=DataframeRef(), description="First DataFrame to be merged."
    )
    dataframe_b: DataframeRef = Field(
        default=DataframeRef(), description="Second DataFrame to be merged."
    )

    async def process(self, context: ProcessingContext) -> DataframeRef:
        df_a = await context.dataframe_to_pandas(self.dataframe_a)
        df_b = await context.dataframe_to_pandas(self.dataframe_b)
        df = pd.concat([df_a, df_b], axis=1)
        return await context.dataframe_from_pandas(df)


class Append(BaseNode):
    """
    Append two dataframes along rows.
    append, concat, rows

    Use cases:
    - Combine data from multiple time periods
    - Merge datasets with same structure
    - Aggregate data from different sources
    """

    _expose_as_tool: ClassVar[bool] = True
    dataframe_a: DataframeRef = Field(
        default=DataframeRef(), description="First DataFrame to be appended."
    )
    dataframe_b: DataframeRef = Field(
        default=DataframeRef(), description="Second DataFrame to be appended."
    )

    async def process(self, context: ProcessingContext) -> DataframeRef:
        df_a = await context.dataframe_to_pandas(self.dataframe_a)
        df_b = await context.dataframe_to_pandas(self.dataframe_b)

        # Handle empty dataframes
        if df_a.empty:
            return await context.dataframe_from_pandas(df_b)
        if df_b.empty:
            return await context.dataframe_from_pandas(df_a)

        # Check column compatibility only if both dataframes are non-empty
        if not df_a.columns.equals(df_b.columns):
            raise ValueError(
                f"Columns in dataframe A ({df_a.columns}) do not match columns in dataframe B ({df_b.columns})"
            )

        df = pd.concat([df_a, df_b], axis=0)
        return await context.dataframe_from_pandas(df)


class Join(BaseNode):
    """
    Join two dataframes on specified column.
    join, merge, column

    Use cases:
    - Combine data from related tables
    - Enrich dataset with additional information
    - Link data based on common identifiers
    """

    _expose_as_tool: ClassVar[bool] = True
    dataframe_a: DataframeRef = Field(
        default=DataframeRef(), description="First DataFrame to be merged."
    )
    dataframe_b: DataframeRef = Field(
        default=DataframeRef(), description="Second DataFrame to be merged."
    )
    join_on: str = Field(
        default="",
        description="The column name on which to join the two dataframes.",
    )

    async def process(self, context: ProcessingContext) -> DataframeRef:
        df_a = await context.dataframe_to_pandas(self.dataframe_a)
        df_b = await context.dataframe_to_pandas(self.dataframe_b)
        if not df_a.columns.equals(df_b.columns):
            raise ValueError(
                f"Columns in dataframe A ({df_a.columns}) do not match columns in dataframe B ({df_b.columns})"
            )
        df = pd.merge(df_a, df_b, on=self.join_on)
        return await context.dataframe_from_pandas(df)


class RowIterator(BaseNode):
    """
    Iterate over rows of a dataframe.
    """

    dataframe: DataframeRef = Field(
        default=DataframeRef(), description="The input dataframe."
    )

    @classmethod
    def get_title(cls):
        return "Row Iterator"

    class OutputType(TypedDict):
        dict: dict
        index: Any

    async def gen_process(
        self, context: ProcessingContext
    ) -> AsyncGenerator[OutputType, None]:
        df = await context.dataframe_to_pandas(self.dataframe)
        for index, row in df.iterrows():
            yield {"dict": row.to_dict(), "index": index}


class FindRow(BaseNode):
    """
    Find the first row in a dataframe that matches a given condition.
    filter, query, condition, single row

    Example conditions:
    age > 30
    age > 30 and salary < 50000
    name == 'John Doe'
    100 <= price <= 200
    status in ['Active', 'Pending']
    not (age < 18)

    Use cases:
    - Retrieve specific record based on criteria
    - Find first occurrence of a particular condition
    - Extract single data point for further analysis
    """

    _expose_as_tool: ClassVar[bool] = True
    df: DataframeRef = Field(
        default=DataframeRef(), description="The DataFrame to search."
    )
    condition: str = Field(
        default="",
        description="The condition to filter the DataFrame, e.g. 'column_name == value'.",
    )

    async def process(self, context: ProcessingContext) -> DataframeRef:
        df = await context.dataframe_to_pandas(self.df)
        result = df.query(self.condition).head(1)
        return await context.dataframe_from_pandas(result)


class SortByColumn(BaseNode):
    """
    Sort dataframe by specified column.
    sort, order, column

    Use cases:
    - Arrange data in ascending or descending order
    - Identify top or bottom values in dataset
    - Prepare data for rank-based analysis
    """

    df: DataframeRef = Field(default=DataframeRef())
    column: str = Field(default="", description="The column to sort the DataFrame by.")

    async def process(self, context: ProcessingContext) -> DataframeRef:
        df = await context.dataframe_to_pandas(self.df)
        res = df.sort_values(self.column)
        return await context.dataframe_from_pandas(res)


class DropDuplicates(BaseNode):
    """
    Remove duplicate rows from dataframe.
    duplicates, unique, clean

    Use cases:
    - Clean dataset by removing redundant entries
    - Ensure data integrity in analysis
    - Prepare data for unique value operations
    """

    _expose_as_tool: ClassVar[bool] = True
    df: DataframeRef = Field(default=DataframeRef(), description="The input DataFrame.")

    async def process(self, context: ProcessingContext) -> DataframeRef:
        df = await context.dataframe_to_pandas(self.df)
        res = df.drop_duplicates()
        return await context.dataframe_from_pandas(res)


class DropNA(BaseNode):
    """
    Remove rows with NA values from dataframe.
    na, missing, clean

    Use cases:
    - Clean dataset by removing incomplete entries
    - Prepare data for analysis requiring complete cases
    - Improve data quality for modeling
    """

    _expose_as_tool: ClassVar[bool] = True
    df: DataframeRef = Field(default=DataframeRef(), description="The input DataFrame.")

    async def process(self, context: ProcessingContext) -> DataframeRef:
        df = await context.dataframe_to_pandas(self.df)
        res = df.dropna()
        return await context.dataframe_from_pandas(res)


class LoadCSVAssets(BaseNode):
    """
    Load dataframes from an asset folder.
    load, dataframe, file, import

    Use cases:
    - Load multiple dataframes from a folder
    - Process multiple datasets in sequence
    - Batch import of data files
    """

    _expose_as_tool: ClassVar[bool] = True
    folder: FolderRef = Field(
        default=FolderRef(), description="The asset folder to load the dataframes from."
    )

    @classmethod
    def get_title(cls):
        return "Load CSV Assets"

    class OutputType(TypedDict):
        dataframe: DataframeRef
        name: str

    async def gen_process(
        self, context: ProcessingContext
    ) -> AsyncGenerator[OutputType, None]:
        if self.folder.is_empty():
            raise ValueError("Please select an asset folder.")

        parent_id = self.folder.asset_id
        list_assets, _ = await context.list_assets(
            parent_id=parent_id, content_type="text/csv"
        )

        for asset in list_assets:
            bytes_io = await context.download_asset(asset.id)
            df = pd.read_csv(bytes_io)
            yield {
                "name": asset.name,
                "dataframe": await context.dataframe_from_pandas(df),
            }


class Aggregate(BaseNode):
    """
    Aggregate dataframe by one or more columns.
    aggregate, groupby, group, sum, mean, count, min, max, std, var, median, first, last

    Use cases:
    - Prepare data for aggregation operations
    - Analyze data by categories
    - Create summary statistics by groups
    """

    _expose_as_tool: ClassVar[bool] = True
    dataframe: DataframeRef = Field(
        default=DataframeRef(), description="The DataFrame to group."
    )
    columns: str = Field(
        default="",
        description="Comma-separated column names to group by.",
    )
    aggregation: str = Field(
        default="sum",
        description="Aggregation function: sum, mean, count, min, max, std, var, median, first, last",
    )

    async def process(self, context: ProcessingContext) -> DataframeRef:
        df = await context.dataframe_to_pandas(self.dataframe)
        group_columns = [col.strip() for col in self.columns.split(",")]

        # Apply aggregation
        if self.aggregation == "sum":
            result = df.groupby(group_columns).sum()
        elif self.aggregation == "mean":
            result = df.groupby(group_columns).mean()
        elif self.aggregation == "count":
            result = df.groupby(group_columns).count()
        elif self.aggregation == "min":
            result = df.groupby(group_columns).min()
        elif self.aggregation == "max":
            result = df.groupby(group_columns).max()
        elif self.aggregation == "std":
            result = df.groupby(group_columns).std()
        elif self.aggregation == "var":
            result = df.groupby(group_columns).var()
        elif self.aggregation == "median":
            result = df.groupby(group_columns).median()
        elif self.aggregation == "first":
            result = df.groupby(group_columns).first()
        elif self.aggregation == "last":
            result = df.groupby(group_columns).last()
        else:
            raise ValueError(f"Unknown aggregation function: {self.aggregation}")

        # Reset index to convert group columns back to regular columns
        result = result.reset_index()  # type: ignore
        return await context.dataframe_from_pandas(result)


class Pivot(BaseNode):
    """
    Pivot dataframe to reshape data.
    pivot, reshape, transform

    Use cases:
    - Transform long data to wide format
    - Create cross-tabulation tables
    - Reorganize data for visualization
    """

    _expose_as_tool: ClassVar[bool] = True
    dataframe: DataframeRef = Field(
        default=DataframeRef(), description="The DataFrame to pivot."
    )
    index: str = Field(
        default="",
        description="Column name to use as index (rows).",
    )
    columns: str = Field(
        default="",
        description="Column name to use as columns.",
    )
    values: str = Field(
        default="",
        description="Column name to use as values.",
    )
    aggfunc: str = Field(
        default="sum",
        description="Aggregation function: sum, mean, count, min, max, first, last",
    )

    async def process(self, context: ProcessingContext) -> DataframeRef:
        df = await context.dataframe_to_pandas(self.dataframe)

        # Map string aggfunc to pandas function
        agg_map = {
            "sum": "sum",
            "mean": "mean",
            "count": "count",
            "min": "min",
            "max": "max",
            "first": "first",
            "last": "last",
        }

        if self.aggfunc not in agg_map:
            raise ValueError(f"Unknown aggregation function: {self.aggfunc}")

        result = df.pivot_table(
            index=self.index,
            columns=self.columns,
            values=self.values,
            aggfunc=agg_map[self.aggfunc],
        )

        # Flatten column names if multi-level
        if isinstance(result.columns, pd.MultiIndex):
            result.columns = [
                "_".join(map(str, col)).strip() for col in result.columns.values
            ]

        result = result.reset_index()
        return await context.dataframe_from_pandas(result)


class Rename(BaseNode):
    """
    Rename columns in dataframe.
    rename, columns, names

    Use cases:
    - Standardize column names
    - Make column names more descriptive
    - Prepare data for specific requirements
    """

    _expose_as_tool: ClassVar[bool] = True
    dataframe: DataframeRef = Field(
        default=DataframeRef(), description="The DataFrame to rename columns."
    )
    rename_map: str = Field(
        default="",
        description="Column rename mapping in format: old1:new1,old2:new2",
    )

    async def process(self, context: ProcessingContext) -> DataframeRef:
        df = await context.dataframe_to_pandas(self.dataframe)

        # Parse rename mapping
        rename_dict = {}
        for mapping in self.rename_map.split(","):
            if ":" in mapping:
                old_name, new_name = mapping.strip().split(":", 1)
                rename_dict[old_name.strip()] = new_name.strip()

        if rename_dict:
            df = df.rename(columns=rename_dict)

        return await context.dataframe_from_pandas(df)


class FillNA(BaseNode):
    """
    Fill missing values in dataframe.
    fillna, missing, impute

    Use cases:
    - Handle missing data
    - Prepare data for analysis
    - Improve data quality
    """

    _expose_as_tool: ClassVar[bool] = True
    dataframe: DataframeRef = Field(
        default=DataframeRef(), description="The DataFrame with missing values."
    )
    value: Any = Field(
        default=0,
        description="Value to use for filling missing values.",
    )
    method: str = Field(
        default="value",
        description="Method for filling: value, forward, backward, mean, median",
    )
    columns: str = Field(
        default="",
        description="Comma-separated column names to fill. Leave empty for all columns.",
    )

    async def process(self, context: ProcessingContext) -> DataframeRef:
        df = await context.dataframe_to_pandas(self.dataframe)

        if self.columns:
            cols = [col.strip() for col in self.columns.split(",")]
        else:
            cols = df.columns.tolist()

        if self.method == "value":
            df[cols] = df[cols].fillna(self.value)
        elif self.method == "forward":
            df[cols] = df[cols].fillna(method="ffill")  # type: ignore
        elif self.method == "backward":
            df[cols] = df[cols].fillna(method="bfill")  # type: ignore
        elif self.method == "mean":
            for col in cols:
                if df[col].dtype in ["float64", "int64"]:
                    df[col] = df[col].fillna(df[col].mean())
        elif self.method == "median":
            for col in cols:
                if df[col].dtype in ["float64", "int64"]:
                    df[col] = df[col].fillna(df[col].median())
        else:
            raise ValueError(f"Unknown fill method: {self.method}")

        return await context.dataframe_from_pandas(df)


class SaveCSVDataframeFile(BaseNode):
    """
    Write a pandas DataFrame to a CSV file.
    files, csv, write, output, save, file

    The filename can include time and date variables:
    %Y - Year, %m - Month, %d - Day
    %H - Hour, %M - Minute, %S - Second
    """

    dataframe: DataframeRef = Field(
        default=DataframeRef(), description="DataFrame to write to CSV"
    )
    folder: str = Field(default="", description="Folder where the file will be saved")
    filename: str = Field(
        default="",
        description="Name of the CSV file to save. Supports strftime format codes.",
    )

    async def process(self, context: ProcessingContext):
        if Environment.is_production():
            raise ValueError("This node is not available in production")
        if not self.folder:
            raise ValueError("folder cannot be empty")
        if not self.filename:
            raise ValueError("filename cannot be empty")

        expanded_folder = os.path.expanduser(self.folder)
        if not os.path.exists(expanded_folder):
            raise ValueError(f"Folder does not exist: {expanded_folder}")

        filename = datetime.now().strftime(self.filename)
        expanded_path = os.path.join(expanded_folder, filename)
        df = await context.dataframe_to_pandas(self.dataframe)
        df.to_csv(expanded_path, index=False)
