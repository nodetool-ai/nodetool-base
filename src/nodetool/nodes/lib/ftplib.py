import asyncio
import io
import ftplib
from pydantic import Field
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.metadata.types import DocumentRef


class FTPBaseNode(BaseNode):
    """Base node for FTP operations.
    ftp, network, transfer

    Use cases:
    - Provide shared connection parameters
    - Reuse login logic across FTP nodes
    - Hide base class from UI
    """

    host: str = Field(default="", description="FTP server host")
    username: str = Field(default="", description="Username for authentication")
    password: str = Field(default="", description="Password for authentication")

    @classmethod
    def is_visible(cls) -> bool:
        return cls is not FTPBaseNode

    def _connect(self) -> ftplib.FTP:
        ftp = ftplib.FTP(self.host)
        ftp.login(self.username, self.password)
        return ftp


class FTPDownloadFile(FTPBaseNode):
    """Download a file from an FTP server.
    ftp, download, file

    Use cases:
    - Retrieve remote files for processing
    - Backup data from an FTP server
    - Integrate legacy FTP systems
    """

    remote_path: str = Field(default="", description="Remote file path to download")

    @classmethod
    def get_title(cls):
        return "Download File"

    async def process(self, context: ProcessingContext) -> DocumentRef:
        if not self.remote_path:
            raise ValueError("remote_path cannot be empty")

        def _download() -> bytes:
            ftp = self._connect()
            try:
                buffer = io.BytesIO()
                ftp.retrbinary(f"RETR {self.remote_path}", buffer.write)
                return buffer.getvalue()
            finally:
                try:
                    ftp.quit()
                except Exception:
                    ftp.close()

        data = await asyncio.to_thread(_download)
        return DocumentRef(data=data)


class FTPUploadFile(FTPBaseNode):
    """Upload a file to an FTP server.
    ftp, upload, file

    Use cases:
    - Transfer files to an FTP server
    - Automate backups to a remote system
    - Integrate with legacy FTP workflows
    """

    remote_path: str = Field(default="", description="Remote file path to upload to")
    document: DocumentRef = Field(
        default=DocumentRef(), description="Document to upload"
    )

    @classmethod
    def get_title(cls):
        return "Upload File"

    async def process(self, context: ProcessingContext) -> None:
        if not self.remote_path:
            raise ValueError("remote_path cannot be empty")

        data = await context.asset_to_bytes(self.document)

        def _upload() -> None:
            ftp = self._connect()
            try:
                ftp.storbinary(f"STOR {self.remote_path}", io.BytesIO(data))
            finally:
                try:
                    ftp.quit()
                except Exception:
                    ftp.close()

        await asyncio.to_thread(_upload)
        return None


class FTPListDirectory(FTPBaseNode):
    """List files in a directory on an FTP server.
    ftp, list, directory

    Use cases:
    - Browse remote directories
    - Check available files before download
    - Monitor FTP server contents
    """

    directory: str = Field(default="", description="Remote directory to list")

    @classmethod
    def get_title(cls):
        return "List Directory"

    async def process(self, context: ProcessingContext) -> list[str]:
        def _list() -> list[str]:
            ftp = self._connect()
            try:
                if self.directory:
                    ftp.cwd(self.directory)
                return ftp.nlst()
            finally:
                try:
                    ftp.quit()
                except Exception:
                    ftp.close()

        return await asyncio.to_thread(_list)
