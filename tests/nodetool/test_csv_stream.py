import pytest
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.metadata.types import FilePath
from nodetool.nodes.nodetool.data import CSVRowIterator
from nodetool.nodes.nodetool.os import LoadCSVFileStream


@pytest.fixture
def context():
    return ProcessingContext(user_id="test", auth_token="test")


@pytest.mark.asyncio
async def test_csv_row_iterator(context: ProcessingContext):
    csv_text = "a,b\n1,2\n3,4\n"
    node = CSVRowIterator(csv_data=csv_text)
    rows = []
    async for name, value in node.gen_process(context):
        if name == "dict":
            rows.append(value)
    assert rows == [{"a": "1", "b": "2"}, {"a": "3", "b": "4"}]


@pytest.mark.asyncio
async def test_load_csv_file_stream(context: ProcessingContext, tmp_path):
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("a,b\n5,6\n7,8\n")
    node = LoadCSVFileStream(path=FilePath(path=str(csv_file)))
    rows = []
    async for name, value in node.gen_process(context):
        if name == "dict":
            rows.append(value)
    assert rows == [{"a": "5", "b": "6"}, {"a": "7", "b": "8"}]
