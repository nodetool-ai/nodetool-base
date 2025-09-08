from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode


class CreateTar(GraphNode):
    """
    Create a tar archive from a directory.
    files, tar, create

    Use cases:
    - Package multiple files into a single archive
    - Backup directories
    - Prepare archives for distribution
    """

    source_folder: types.FilePath | GraphNode | tuple[GraphNode, str] = Field(
        default=types.FilePath(type="file_path", path=""),
        description="Folder to archive",
    )
    tar_path: types.FilePath | GraphNode | tuple[GraphNode, str] = Field(
        default=types.FilePath(type="file_path", path=""),
        description="Output tar file path",
    )
    gzip: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=False, description="Use gzip compression"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.tar.CreateTar"


class ExtractTar(GraphNode):
    """
    Extract a tar archive to a folder.
    files, tar, extract

    Use cases:
    - Unpack archived data
    - Restore backups
    - Retrieve files for processing
    """

    tar_path: types.FilePath | GraphNode | tuple[GraphNode, str] = Field(
        default=types.FilePath(type="file_path", path=""),
        description="Tar archive to extract",
    )
    output_folder: types.FilePath | GraphNode | tuple[GraphNode, str] = Field(
        default=types.FilePath(type="file_path", path=""),
        description="Folder to extract into",
    )

    @classmethod
    def get_node_type(cls):
        return "lib.tar.ExtractTar"


class ListTar(GraphNode):
    """
    List contents of a tar archive.
    files, tar, list

    Use cases:
    - Inspect archives without extracting
    - Preview tar contents
    - Verify archive contents
    """

    tar_path: types.FilePath | GraphNode | tuple[GraphNode, str] = Field(
        default=types.FilePath(type="file_path", path=""),
        description="Tar archive to inspect",
    )

    @classmethod
    def get_node_type(cls):
        return "lib.tar.ListTar"
