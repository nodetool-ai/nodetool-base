from datetime import datetime
from io import StringIO
import json
import pandas as pd
from typing import Any
from pydantic import Field
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.metadata.types import DataframeRef, FolderRef


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

    csv_data: str = Field(
        default="", title="CSV Data", description="String input of CSV formatted text."
    )

    async def process(self, context: ProcessingContext) -> DataframeRef:
        df = pd.read_csv(StringIO(self.csv_data))
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
                if type(v) == dict:
                    row[k] = v["value"]
                elif type(v) in [int, float, str, bool]:
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

    dataframe: DataframeRef = Field(
        default=DataframeRef(), description="The input dataframe to convert."
    )

    async def process(self, context: ProcessingContext) -> list[dict]:
        df = await context.dataframe_to_pandas(self.dataframe)
        return df.to_dict("records")
