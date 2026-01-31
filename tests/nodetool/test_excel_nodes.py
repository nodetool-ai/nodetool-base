import pytest
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.metadata.types import ExcelRef, DataframeRef
from nodetool.nodes.lib.excel import (
    CreateWorkbook,
    DataFrameToExcel,
    ExcelToDataFrame,
    FormatCells,
    AutoFitColumns,
)


@pytest.fixture
def context():
    return ProcessingContext(user_id="test", auth_token="test")


class TestCreateWorkbook:
    """Tests for CreateWorkbook node."""

    @pytest.mark.asyncio
    async def test_create_default_workbook(self, context: ProcessingContext):
        node = CreateWorkbook()
        result = await node.process(context)
        assert isinstance(result, ExcelRef)
        assert result.data is not None
        # Default sheet name should be Sheet1
        assert "Sheet1" in result.data.sheetnames

    @pytest.mark.asyncio
    async def test_create_workbook_custom_sheet_name(self, context: ProcessingContext):
        node = CreateWorkbook(sheet_name="CustomSheet")
        result = await node.process(context)
        assert isinstance(result, ExcelRef)
        assert "CustomSheet" in result.data.sheetnames


class TestDataFrameToExcel:
    """Tests for DataFrameToExcel node."""

    @pytest.mark.asyncio
    async def test_dataframe_to_excel(self, context: ProcessingContext):
        # Create a workbook
        wb_node = CreateWorkbook()
        workbook = await wb_node.process(context)

        # Create a simple dataframe
        import pandas as pd
        df = pd.DataFrame({
            "Name": ["Alice", "Bob", "Charlie"],
            "Age": [25, 30, 35],
            "Score": [85.5, 90.0, 78.5],
        })
        df_ref = await context.dataframe_from_pandas(df)

        # Write to Excel
        node = DataFrameToExcel(
            workbook=workbook,
            dataframe=df_ref,
            sheet_name="Sheet1",
            include_header=True,
        )
        result = await node.process(context)

        assert isinstance(result, ExcelRef)
        ws = result.data["Sheet1"]
        # Check headers
        assert ws.cell(1, 1).value == "Name"
        assert ws.cell(1, 2).value == "Age"
        assert ws.cell(1, 3).value == "Score"
        # Check data
        assert ws.cell(2, 1).value == "Alice"
        assert ws.cell(2, 2).value == 25

    @pytest.mark.asyncio
    async def test_dataframe_to_new_sheet(self, context: ProcessingContext):
        wb_node = CreateWorkbook()
        workbook = await wb_node.process(context)

        import pandas as pd
        df = pd.DataFrame({"Col1": [1, 2, 3]})
        df_ref = await context.dataframe_from_pandas(df)

        node = DataFrameToExcel(
            workbook=workbook,
            dataframe=df_ref,
            sheet_name="NewSheet",
            include_header=True,
        )
        result = await node.process(context)

        assert "NewSheet" in result.data.sheetnames

    @pytest.mark.asyncio
    async def test_dataframe_no_header(self, context: ProcessingContext):
        wb_node = CreateWorkbook()
        workbook = await wb_node.process(context)

        import pandas as pd
        df = pd.DataFrame({"Col1": [10, 20], "Col2": [30, 40]})
        df_ref = await context.dataframe_from_pandas(df)

        node = DataFrameToExcel(
            workbook=workbook,
            dataframe=df_ref,
            sheet_name="Sheet1",
            include_header=False,
        )
        result = await node.process(context)

        ws = result.data["Sheet1"]
        # First row should be data, not header
        assert ws.cell(1, 1).value == 10


class TestExcelToDataFrame:
    """Tests for ExcelToDataFrame node."""

    @pytest.mark.asyncio
    async def test_excel_to_dataframe(self, context: ProcessingContext):
        # Create workbook and add data
        wb_node = CreateWorkbook()
        workbook = await wb_node.process(context)

        import pandas as pd
        df = pd.DataFrame({
            "Product": ["Widget", "Gadget"],
            "Price": [10.99, 24.99],
        })
        df_ref = await context.dataframe_from_pandas(df)

        write_node = DataFrameToExcel(
            workbook=workbook,
            dataframe=df_ref,
            sheet_name="Sheet1",
            include_header=True,
        )
        workbook = await write_node.process(context)

        # Read back as dataframe
        read_node = ExcelToDataFrame(
            workbook=workbook,
            sheet_name="Sheet1",
            has_header=True,
        )
        result = await read_node.process(context)

        assert isinstance(result, DataframeRef)
        result_df = await context.dataframe_to_pandas(result)
        assert list(result_df.columns) == ["Product", "Price"]
        assert len(result_df) == 2


class TestFormatCells:
    """Tests for FormatCells node."""

    @pytest.mark.asyncio
    async def test_format_cells(self, context: ProcessingContext):
        # Create workbook with some data
        wb_node = CreateWorkbook()
        workbook = await wb_node.process(context)

        import pandas as pd
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        df_ref = await context.dataframe_from_pandas(df)

        write_node = DataFrameToExcel(workbook=workbook, dataframe=df_ref)
        workbook = await write_node.process(context)

        # Format cells
        format_node = FormatCells(
            workbook=workbook,
            sheet_name="Sheet1",
            cell_range="A1:B2",
            bold=True,
            background_color="FFFF00",
            text_color="FF0000",
        )
        result = await format_node.process(context)

        assert isinstance(result, ExcelRef)


class TestAutoFitColumns:
    """Tests for AutoFitColumns node."""

    @pytest.mark.asyncio
    async def test_autofit_columns(self, context: ProcessingContext):
        wb_node = CreateWorkbook()
        workbook = await wb_node.process(context)

        import pandas as pd
        df = pd.DataFrame({
            "Short": [1, 2],
            "LongColumnNameHere": [100, 200],
        })
        df_ref = await context.dataframe_from_pandas(df)

        write_node = DataFrameToExcel(workbook=workbook, dataframe=df_ref)
        workbook = await write_node.process(context)

        # Auto-fit columns
        fit_node = AutoFitColumns(workbook=workbook, sheet_name="Sheet1")
        result = await fit_node.process(context)

        assert isinstance(result, ExcelRef)
        ws = result.data["Sheet1"]
        # Column widths should be adjusted (just verify it doesn't crash)
        assert ws is not None


class TestExcelWorkflow:
    """Integration tests for Excel workflows."""

    @pytest.mark.asyncio
    async def test_complete_excel_workflow(self, context: ProcessingContext):
        """Test creating, writing, formatting, and reading Excel data."""
        import pandas as pd

        # Create workbook
        wb_node = CreateWorkbook(sheet_name="Sales")
        workbook = await wb_node.process(context)

        # Create sales data
        sales_df = pd.DataFrame({
            "Product": ["Widget A", "Widget B", "Gadget X"],
            "Q1": [1000, 1500, 800],
            "Q2": [1200, 1600, 900],
            "Q3": [1100, 1400, 1100],
            "Q4": [1300, 1700, 1200],
        })
        sales_ref = await context.dataframe_from_pandas(sales_df)

        # Write to Excel
        write_node = DataFrameToExcel(
            workbook=workbook,
            dataframe=sales_ref,
            sheet_name="Sales",
            include_header=True,
        )
        workbook = await write_node.process(context)

        # Format header row
        format_node = FormatCells(
            workbook=workbook,
            sheet_name="Sales",
            cell_range="A1:E1",
            bold=True,
            background_color="4472C4",
            text_color="FFFFFF",
        )
        workbook = await format_node.process(context)

        # Auto-fit columns
        fit_node = AutoFitColumns(workbook=workbook, sheet_name="Sales")
        workbook = await fit_node.process(context)

        # Read back and verify
        read_node = ExcelToDataFrame(
            workbook=workbook,
            sheet_name="Sales",
            has_header=True,
        )
        result = await read_node.process(context)
        result_df = await context.dataframe_to_pandas(result)

        assert len(result_df) == 3
        assert list(result_df.columns) == ["Product", "Q1", "Q2", "Q3", "Q4"]
