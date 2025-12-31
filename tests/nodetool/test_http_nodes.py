import pytest
from unittest.mock import AsyncMock, MagicMock
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.nodes.lib.http import (
    GetRequest,
    PostRequest,
    PutRequest,
    DeleteRequest,
    HeadRequest,
    GetRequestBinary,
    GetRequestDocument,
    PostRequestBinary,
    HTTPBaseNode,
    DownloadDataframe,
)
from nodetool.metadata.types import ColumnDef, RecordType


class DummyResponse:
    def __init__(self, text: str = "", headers: dict | None = None, encoding: str = "utf-8", json_data: dict | None = None):
        self.content = text.encode(encoding)
        self.encoding = encoding
        self.headers = headers or {}
        self._json_data = json_data

    def json(self):
        if self._json_data is not None:
            return self._json_data
        import json
        return json.loads(self.content.decode(self.encoding))


@pytest.fixture
def mock_context():
    ctx = MagicMock(spec=ProcessingContext)
    ctx.http_get = AsyncMock(return_value=DummyResponse("ok-get"))
    ctx.http_post = AsyncMock(return_value=DummyResponse("ok-post"))
    ctx.http_put = AsyncMock(return_value=DummyResponse("ok-put"))
    ctx.http_delete = AsyncMock(return_value=DummyResponse("ok-delete"))
    ctx.http_head = AsyncMock(return_value=DummyResponse(headers={"X": "1"}))
    ctx.http_patch = AsyncMock(return_value=DummyResponse('{"status": "patched"}', json_data={"status": "patched"}))
    return ctx


@pytest.mark.asyncio
async def test_http_basic_requests(mock_context: ProcessingContext):
    assert await GetRequest(url="http://x").process(mock_context) == "ok-get"
    assert (
        await PostRequest(url="http://x", data="d").process(mock_context) == "ok-post"
    )
    assert await PutRequest(url="http://x", data="d").process(mock_context) == "ok-put"
    assert await DeleteRequest(url="http://x").process(mock_context) == "ok-delete"


@pytest.mark.asyncio
async def test_http_head(mock_context: ProcessingContext):
    result = await HeadRequest(url="http://x").process(mock_context)
    assert result == {"X": "1"}


@pytest.mark.asyncio
async def test_http_get_binary(mock_context: ProcessingContext):
    """Test GetRequestBinary returns raw bytes."""
    mock_context.http_get = AsyncMock(return_value=DummyResponse("binary-data"))
    result = await GetRequestBinary(url="http://x/file.bin").process(mock_context)
    assert result == b"binary-data"
    assert isinstance(result, bytes)


@pytest.mark.asyncio
async def test_http_get_document(mock_context: ProcessingContext):
    """Test GetRequestDocument returns DocumentRef."""
    mock_context.http_get = AsyncMock(return_value=DummyResponse("document-content"))
    result = await GetRequestDocument(url="http://x/doc.pdf").process(mock_context)
    assert result.data == b"document-content"


@pytest.mark.asyncio
async def test_http_post_binary(mock_context: ProcessingContext):
    """Test PostRequestBinary returns raw bytes."""
    mock_context.http_post = AsyncMock(return_value=DummyResponse("binary-response"))
    result = await PostRequestBinary(url="http://x/upload", data="input-data").process(mock_context)
    assert result == b"binary-response"
    assert isinstance(result, bytes)


@pytest.mark.asyncio
async def test_http_base_node_visibility():
    """Test HTTPBaseNode is not visible but subclasses are."""
    assert HTTPBaseNode.is_visible() is False
    assert GetRequest.is_visible() is True
    assert PostRequest.is_visible() is True


@pytest.mark.asyncio
async def test_http_get_request_kwargs():
    """Test get_request_kwargs method."""
    node = GetRequest(url="http://x")
    assert node.get_request_kwargs() == {}


@pytest.mark.asyncio
async def test_http_get_basic_fields():
    """Test get_basic_fields classmethod."""
    assert HTTPBaseNode.get_basic_fields() == ["url"]


@pytest.mark.asyncio
async def test_http_get_titles():
    """Test get_title for various HTTP request nodes."""
    assert GetRequest.get_title() == "GET Request"
    assert PostRequest.get_title() == "POST Request"
    assert PutRequest.get_title() == "PUT Request"
    assert DeleteRequest.get_title() == "DELETE Request"
    assert HeadRequest.get_title() == "HEAD Request"
    assert GetRequestBinary.get_title() == "GET Binary"
    assert GetRequestDocument.get_title() == "GET Document"
    assert PostRequestBinary.get_title() == "POST Binary"


@pytest.mark.asyncio
async def test_download_dataframe_csv(mock_context: ProcessingContext):
    """Test DownloadDataframe with CSV format."""
    csv_content = "name,age,score\nAlice,30,85\nBob,25,90"
    mock_context.http_get = AsyncMock(return_value=DummyResponse(csv_content))
    
    columns = RecordType(columns=[
        ColumnDef(name="name", data_type="string"),
        ColumnDef(name="age", data_type="int"),
        ColumnDef(name="score", data_type="int"),
    ])
    
    node = DownloadDataframe(
        url="http://x/data.csv",
        file_format=DownloadDataframe.FileFormat.CSV,
        columns=columns,
    )
    result = await node.process(mock_context)
    
    assert len(result.columns) == 3
    assert result.data == [["Alice", 30, 85], ["Bob", 25, 90]]


@pytest.mark.asyncio
async def test_download_dataframe_json(mock_context: ProcessingContext):
    """Test DownloadDataframe with JSON format."""
    import json
    json_content = json.dumps([
        {"name": "Alice", "age": 30},
        {"name": "Bob", "age": 25},
    ])
    mock_context.http_get = AsyncMock(return_value=DummyResponse(json_content))
    
    columns = RecordType(columns=[
        ColumnDef(name="name", data_type="string"),
        ColumnDef(name="age", data_type="int"),
    ])
    
    node = DownloadDataframe(
        url="http://x/data.json",
        file_format=DownloadDataframe.FileFormat.JSON,
        columns=columns,
    )
    result = await node.process(mock_context)
    
    assert len(result.columns) == 2
    assert result.data == [["Alice", 30], ["Bob", 25]]


@pytest.mark.asyncio
async def test_download_dataframe_empty_columns(mock_context: ProcessingContext):
    """Test DownloadDataframe returns empty dataframe when no columns defined."""
    csv_content = "name,age\nAlice,30"
    mock_context.http_get = AsyncMock(return_value=DummyResponse(csv_content))
    
    node = DownloadDataframe(
        url="http://x/data.csv",
        file_format=DownloadDataframe.FileFormat.CSV,
        columns=RecordType(columns=[]),
    )
    result = await node.process(mock_context)
    
    assert result.columns == []
    assert result.data == []


@pytest.mark.asyncio
async def test_download_dataframe_cast_float(mock_context: ProcessingContext):
    """Test DownloadDataframe casts float values correctly."""
    csv_content = "name,price\nItem1,19.99\nItem2,29.50"
    mock_context.http_get = AsyncMock(return_value=DummyResponse(csv_content))
    
    columns = RecordType(columns=[
        ColumnDef(name="name", data_type="string"),
        ColumnDef(name="price", data_type="float"),
    ])
    
    node = DownloadDataframe(
        url="http://x/data.csv",
        file_format=DownloadDataframe.FileFormat.CSV,
        columns=columns,
    )
    result = await node.process(mock_context)
    
    assert result.data[0][1] == 19.99
    assert result.data[1][1] == 29.50
