from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode


class FTPDownloadFile(GraphNode):
    """Download a file from an FTP server.
    ftp, download, file

    Use cases:
    - Retrieve remote files for processing
    - Backup data from an FTP server
    - Integrate legacy FTP systems
    """

    host: str | GraphNode | tuple[GraphNode, str] = Field(default='', description='FTP server host')
    username: str | GraphNode | tuple[GraphNode, str] = Field(default='', description='Username for authentication')
    password: str | GraphNode | tuple[GraphNode, str] = Field(default='', description='Password for authentication')
    remote_path: str | GraphNode | tuple[GraphNode, str] = Field(default='', description='Remote file path to download')

    @classmethod
    def get_node_type(cls): return "lib.ftplib.FTPDownloadFile"



class FTPListDirectory(GraphNode):
    """List files in a directory on an FTP server.
    ftp, list, directory

    Use cases:
    - Browse remote directories
    - Check available files before download
    - Monitor FTP server contents
    """

    host: str | GraphNode | tuple[GraphNode, str] = Field(default='', description='FTP server host')
    username: str | GraphNode | tuple[GraphNode, str] = Field(default='', description='Username for authentication')
    password: str | GraphNode | tuple[GraphNode, str] = Field(default='', description='Password for authentication')
    directory: str | GraphNode | tuple[GraphNode, str] = Field(default='', description='Remote directory to list')

    @classmethod
    def get_node_type(cls): return "lib.ftplib.FTPListDirectory"



class FTPUploadFile(GraphNode):
    """Upload a file to an FTP server.
    ftp, upload, file

    Use cases:
    - Transfer files to an FTP server
    - Automate backups to a remote system
    - Integrate with legacy FTP workflows
    """

    host: str | GraphNode | tuple[GraphNode, str] = Field(default='', description='FTP server host')
    username: str | GraphNode | tuple[GraphNode, str] = Field(default='', description='Username for authentication')
    password: str | GraphNode | tuple[GraphNode, str] = Field(default='', description='Password for authentication')
    remote_path: str | GraphNode | tuple[GraphNode, str] = Field(default='', description='Remote file path to upload to')
    document: types.DocumentRef | GraphNode | tuple[GraphNode, str] = Field(default=types.DocumentRef(type='document', uri='', asset_id=None, data=None), description='Document to upload')

    @classmethod
    def get_node_type(cls): return "lib.ftplib.FTPUploadFile"


