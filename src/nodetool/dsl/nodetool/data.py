from pydantic import Field
from typing import Any
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode


class AddColumn(GraphNode):
    """
    Add list of values as new column to dataframe.
    dataframe, column, list

    Use cases:
    - Incorporate external data into existing dataframe
    - Add calculated results as new column
    - Augment dataframe with additional features
    """

    dataframe: types.DataframeRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.DataframeRef(
            type="dataframe", uri="", asset_id=None, data=None, columns=None
        ),
        description="Dataframe object to add a new column to.",
    )
    column_name: str | GraphNode | tuple[GraphNode, str] = Field(
        default="",
        description="The name of the new column to be added to the dataframe.",
    )
    values: list[Any] | GraphNode | tuple[GraphNode, str] = Field(
        default=[],
        description="A list of any type of elements which will be the new column's values.",
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.data.AddColumn"


class Append(GraphNode):
    """
    Append two dataframes along rows.
    append, concat, rows

    Use cases:
    - Combine data from multiple time periods
    - Merge datasets with same structure
    - Aggregate data from different sources
    """

    dataframe_a: types.DataframeRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.DataframeRef(
            type="dataframe", uri="", asset_id=None, data=None, columns=None
        ),
        description="First DataFrame to be appended.",
    )
    dataframe_b: types.DataframeRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.DataframeRef(
            type="dataframe", uri="", asset_id=None, data=None, columns=None
        ),
        description="Second DataFrame to be appended.",
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.data.Append"


class DropDuplicates(GraphNode):
    """
    Remove duplicate rows from dataframe.
    duplicates, unique, clean

    Use cases:
    - Clean dataset by removing redundant entries
    - Ensure data integrity in analysis
    - Prepare data for unique value operations
    """

    df: types.DataframeRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.DataframeRef(
            type="dataframe", uri="", asset_id=None, data=None, columns=None
        ),
        description="The input DataFrame.",
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.data.DropDuplicates"


class DropNA(GraphNode):
    """
    Remove rows with NA values from dataframe.
    na, missing, clean

    Use cases:
    - Clean dataset by removing incomplete entries
    - Prepare data for analysis requiring complete cases
    - Improve data quality for modeling
    """

    df: types.DataframeRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.DataframeRef(
            type="dataframe", uri="", asset_id=None, data=None, columns=None
        ),
        description="The input DataFrame.",
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.data.DropNA"


class ExtractColumn(GraphNode):
    """
    Convert dataframe column to list.
    dataframe, column, list

    Use cases:
    - Extract data for use in other processing steps
    - Prepare column data for plotting or analysis
    - Convert categorical data to list for encoding
    """

    dataframe: types.DataframeRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.DataframeRef(
            type="dataframe", uri="", asset_id=None, data=None, columns=None
        ),
        description="The input dataframe.",
    )
    column_name: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The name of the column to be converted to a list."
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.data.ExtractColumn"


class Filter(GraphNode):
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

    df: types.DataframeRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.DataframeRef(
            type="dataframe", uri="", asset_id=None, data=None, columns=None
        ),
        description="The DataFrame to filter.",
    )
    condition: str | GraphNode | tuple[GraphNode, str] = Field(
        default="",
        description="The filtering condition to be applied to the DataFrame, e.g. column_name > 5.",
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.data.Filter"


class FindRow(GraphNode):
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

    df: types.DataframeRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.DataframeRef(
            type="dataframe", uri="", asset_id=None, data=None, columns=None
        ),
        description="The DataFrame to search.",
    )
    condition: str | GraphNode | tuple[GraphNode, str] = Field(
        default="",
        description="The condition to filter the DataFrame, e.g. 'column_name == value'.",
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.data.FindRow"


class FromList(GraphNode):
    """
    Convert list of dicts to dataframe.
    list, dataframe, convert

    Use cases:
    - Transform list data into structured dataframe
    - Prepare list data for analysis or visualization
    - Convert API responses to dataframe format
    """

    values: list[Any] | GraphNode | tuple[GraphNode, str] = Field(
        default=[],
        description="List of values to be converted, each value will be a row.",
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.data.FromList"


class ImportCSV(GraphNode):
    """
    Convert CSV string to dataframe.
    csv, dataframe, import

    Use cases:
    - Import CSV data from string input
    - Convert CSV responses from APIs to dataframe
    """

    csv_data: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="String input of CSV formatted text."
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.data.ImportCSV"


class JSONToDataframe(GraphNode):
    """
    Transforms a JSON string into a pandas DataFrame.
    json, dataframe, conversion

    Use cases:
    - Converting API responses to tabular format
    - Preparing JSON data for analysis or visualization
    - Structuring unstructured JSON data for further processing
    """

    text: str | GraphNode | tuple[GraphNode, str] = Field(default="", description=None)

    @classmethod
    def get_node_type(cls):
        return "nodetool.data.JSONToDataframe"


class Join(GraphNode):
    """
    Join two dataframes on specified column.
    join, merge, column

    Use cases:
    - Combine data from related tables
    - Enrich dataset with additional information
    - Link data based on common identifiers
    """

    dataframe_a: types.DataframeRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.DataframeRef(
            type="dataframe", uri="", asset_id=None, data=None, columns=None
        ),
        description="First DataFrame to be merged.",
    )
    dataframe_b: types.DataframeRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.DataframeRef(
            type="dataframe", uri="", asset_id=None, data=None, columns=None
        ),
        description="Second DataFrame to be merged.",
    )
    join_on: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The column name on which to join the two dataframes."
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.data.Join"


class LoadCSVAssets(GraphNode):
    """
    Load dataframes from an asset folder.
    load, dataframe, file, import

    Use cases:
    - Load multiple dataframes from a folder
    - Process multiple datasets in sequence
    - Batch import of data files
    """

    folder: types.FolderRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.FolderRef(type="folder", uri="", asset_id=None, data=None),
        description="The asset folder to load the dataframes from.",
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.data.LoadCSVAssets"


class Merge(GraphNode):
    """
    Merge two dataframes along columns.
    merge, concat, columns

    Use cases:
    - Combine data from multiple sources
    - Add new features to existing dataframe
    - Merge time series data from different periods
    """

    dataframe_a: types.DataframeRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.DataframeRef(
            type="dataframe", uri="", asset_id=None, data=None, columns=None
        ),
        description="First DataFrame to be merged.",
    )
    dataframe_b: types.DataframeRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.DataframeRef(
            type="dataframe", uri="", asset_id=None, data=None, columns=None
        ),
        description="Second DataFrame to be merged.",
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.data.Merge"


class RowIterator(GraphNode):
    """
    Iterate over rows of a dataframe.
    """

    dataframe: types.DataframeRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.DataframeRef(
            type="dataframe", uri="", asset_id=None, data=None, columns=None
        ),
        description="The input dataframe.",
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.data.RowIterator"


class SaveDataframe(GraphNode):
    """
    Save dataframe in specified folder.
    csv, folder, save

    Use cases:
    - Export processed data for external use
    - Create backups of dataframes
    """

    df: types.DataframeRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.DataframeRef(
            type="dataframe", uri="", asset_id=None, data=None, columns=None
        ),
        description=None,
    )
    folder: types.FolderRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.FolderRef(type="folder", uri="", asset_id=None, data=None),
        description="Name of the output folder.",
    )
    name: str | GraphNode | tuple[GraphNode, str] = Field(
        default="output.csv",
        description="\n        Name of the output file.\n        You can use time and date variables to create unique names:\n        %Y - Year\n        %m - Month\n        %d - Day\n        %H - Hour\n        %M - Minute\n        %S - Second\n        ",
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.data.SaveDataframe"


class SelectColumn(GraphNode):
    """
    Select specific columns from dataframe.
    dataframe, columns, filter

    Use cases:
    - Extract relevant features for analysis
    - Reduce dataframe size by removing unnecessary columns
    - Prepare data for specific visualizations or models
    """

    dataframe: types.DataframeRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.DataframeRef(
            type="dataframe", uri="", asset_id=None, data=None, columns=None
        ),
        description="a dataframe from which columns are to be selected",
    )
    columns: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="comma separated list of column names"
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.data.SelectColumn"


class Slice(GraphNode):
    """
    Slice a dataframe by rows using start and end indices.
    slice, subset, rows

    Use cases:
    - Extract a specific range of rows from a large dataset
    - Create training and testing subsets for machine learning
    - Analyze data in smaller chunks
    """

    dataframe: types.DataframeRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.DataframeRef(
            type="dataframe", uri="", asset_id=None, data=None, columns=None
        ),
        description="The input dataframe to be sliced.",
    )
    start_index: int | GraphNode | tuple[GraphNode, str] = Field(
        default=0, description="The starting index of the slice (inclusive)."
    )
    end_index: int | GraphNode | tuple[GraphNode, str] = Field(
        default=-1,
        description="The ending index of the slice (exclusive). Use -1 for the last row.",
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.data.Slice"


class SortByColumn(GraphNode):
    """
    Sort dataframe by specified column.
    sort, order, column

    Use cases:
    - Arrange data in ascending or descending order
    - Identify top or bottom values in dataset
    - Prepare data for rank-based analysis
    """

    df: types.DataframeRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.DataframeRef(
            type="dataframe", uri="", asset_id=None, data=None, columns=None
        ),
        description=None,
    )
    column: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The column to sort the DataFrame by."
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.data.SortByColumn"


class ToList(GraphNode):
    """
    Convert dataframe to list of dictionaries.
    dataframe, list, convert

    Use cases:
    - Convert dataframe data for API consumption
    - Transform data for JSON serialization
    - Prepare data for document-based storage
    """

    dataframe: types.DataframeRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.DataframeRef(
            type="dataframe", uri="", asset_id=None, data=None, columns=None
        ),
        description="The input dataframe to convert.",
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.data.ToList"
