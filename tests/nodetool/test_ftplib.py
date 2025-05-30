import pytest
from unittest.mock import patch, MagicMock
from nodetool.nodes.lib.ftplib import FTPDownloadFile, FTPUploadFile, FTPListDirectory
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.metadata.types import DocumentRef


@pytest.fixture
def context():
    return ProcessingContext(user_id="test", auth_token="test")


@pytest.mark.asyncio
@patch("ftplib.FTP")
async def test_download_file(mock_ftp_cls, context):
    mock_ftp = MagicMock()
    mock_ftp_cls.return_value = mock_ftp
    data = b"hello"

    def retrbinary(cmd, callback):
        assert cmd == "RETR remote.txt"
        callback(data)

    mock_ftp.retrbinary.side_effect = retrbinary

    node = FTPDownloadFile(
        host="ftp.example.com", username="u", password="p", remote_path="remote.txt"
    )
    result = await node.process(context)

    assert isinstance(result, DocumentRef)
    assert result.data == data
    mock_ftp.login.assert_called_once_with("u", "p")
    mock_ftp.retrbinary.assert_called()


@pytest.mark.asyncio
@patch("ftplib.FTP")
async def test_upload_file(mock_ftp_cls, context):
    mock_ftp = MagicMock()
    mock_ftp_cls.return_value = mock_ftp

    node = FTPUploadFile(
        host="ftp.example.com",
        username="u",
        password="p",
        remote_path="upload.txt",
        document=DocumentRef(data=b"hello"),
    )

    await node.process(context)

    mock_ftp.login.assert_called_once_with("u", "p")
    assert mock_ftp.storbinary.call_args[0][0] == "STOR upload.txt"


@pytest.mark.asyncio
@patch("ftplib.FTP")
async def test_list_directory(mock_ftp_cls, context):
    mock_ftp = MagicMock()
    mock_ftp_cls.return_value = mock_ftp
    mock_ftp.nlst.return_value = ["a.txt", "b.txt"]

    node = FTPListDirectory(
        host="ftp.example.com",
        username="u",
        password="p",
        directory="data",
    )

    files = await node.process(context)

    mock_ftp.login.assert_called_once_with("u", "p")
    mock_ftp.cwd.assert_called_once_with("data")
    assert files == ["a.txt", "b.txt"]
