from nodetool.config.environment import Environment
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from pydantic import Field
import os
import datetime


class LoadBytesFile(BaseNode):
    """
    Read raw bytes from a file on disk.
    files, bytes, read, input, load, file

    Use cases:
    - Load binary data for processing
    - Read binary files for a workflow
    """

    path: str = Field(default="", description="Path to the file to read")

    async def process(self, context: ProcessingContext) -> bytes:
        if Environment.is_production():
            raise ValueError("This node is not available in production")
        if not self.path:
            raise ValueError("path cannot be empty")
        expanded_path = os.path.expanduser(self.path)
        with open(expanded_path, "rb") as f:
            return f.read()


class SaveBytesFile(BaseNode):
    """
    Write raw bytes to a file on disk.
    files, bytes, save, output

    The filename can include time and date variables:
    %Y - Year, %m - Month, %d - Day
    %H - Hour, %M - Minute, %S - Second
    """

    data: bytes | None = Field(default=None, description="The bytes to write to file")
    folder: str = Field(default="", description="Folder where the file will be saved")
    filename: str = Field(
        default="",
        description="Name of the file to save. Supports strftime format codes.",
    )

    async def process(self, context: ProcessingContext):
        if Environment.is_production():
            raise ValueError("This node is not available in production")
        if not self.data:
            raise ValueError("data cannot be empty")
        if not self.folder:
            raise ValueError("folder cannot be empty")
        if not self.filename:
            raise ValueError("filename cannot be empty")

        expanded_folder = os.path.expanduser(self.folder)
        if not os.path.exists(expanded_folder):
            raise ValueError(f"Folder does not exist: {expanded_folder}")

        filename = datetime.datetime.now().strftime(self.filename)
        expanded_path = os.path.join(expanded_folder, filename)
        os.makedirs(os.path.dirname(expanded_path), exist_ok=True)
        with open(expanded_path, "wb") as f:
            f.write(self.data)
