from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode


class LoadBytesFile(GraphNode):
    """
    Read raw bytes from a file on disk.
    files, bytes, read, input, load, file

    Use cases:
    - Load binary data for processing
    - Read binary files for a workflow
    """

    path: types.FilePath | GraphNode | tuple[GraphNode, str] = Field(
        default=types.FilePath(type="file_path", path=""),
        description="Path to the file to read",
    )

    @classmethod
    def get_node_type(cls):
        return "lib.bytes.LoadBytesFile"


class SaveBytesFile(GraphNode):
    """
    Write raw bytes to a file on disk.
    files, bytes, save, output

    The filename can include time and date variables:
    %Y - Year, %m - Month, %d - Day
    %H - Hour, %M - Minute, %S - Second
    """

    data: bytes | None | GraphNode | tuple[GraphNode, str] = Field(
        default=None, description="The bytes to write to file"
    )
    folder: types.FolderPath | GraphNode | tuple[GraphNode, str] = Field(
        default=types.FolderPath(type="folder_path", path=""),
        description="Folder where the file will be saved",
    )
    filename: str | GraphNode | tuple[GraphNode, str] = Field(
        default="",
        description="Name of the file to save. Supports strftime format codes.",
    )

    @classmethod
    def get_node_type(cls):
        return "lib.bytes.SaveBytesFile"
