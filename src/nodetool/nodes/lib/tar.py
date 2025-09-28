from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.config.environment import Environment
from pydantic import Field
import os
import tarfile


class CreateTar(BaseNode):
    """
    Create a tar archive from a directory.
    files, tar, create

    Use cases:
    - Package multiple files into a single archive
    - Backup directories
    - Prepare archives for distribution
    """

    source_folder: str = Field(default="", description="Folder to archive")
    tar_path: str = Field(default="", description="Output tar file path")
    gzip: bool = Field(default=False, description="Use gzip compression")

    async def process(self, context: ProcessingContext) -> str:
        if Environment.is_production():
            raise ValueError("This node is not available in production")
        if not self.source_folder:
            raise ValueError("source_folder cannot be empty")
        if not self.tar_path:
            raise ValueError("tar_path cannot be empty")

        expanded_folder = os.path.expanduser(self.source_folder)
        expanded_tar = os.path.expanduser(self.tar_path)
        mode = "w:gz" if self.gzip else "w"
        with tarfile.open(expanded_tar, mode) as tar:
            tar.add(expanded_folder, arcname=os.path.basename(expanded_folder))
        return expanded_tar


class ExtractTar(BaseNode):
    """
    Extract a tar archive to a folder.
    files, tar, extract

    Use cases:
    - Unpack archived data
    - Restore backups
    - Retrieve files for processing
    """

    tar_path: str = Field(default="", description="Tar archive to extract")
    output_folder: str = Field(default="", description="Folder to extract into")

    async def process(self, context: ProcessingContext) -> str:
        if Environment.is_production():
            raise ValueError("This node is not available in production")
        if not self.tar_path:
            raise ValueError("tar_path cannot be empty")
        if not self.output_folder:
            raise ValueError("output_folder cannot be empty")

        expanded_tar = os.path.expanduser(self.tar_path)
        expanded_folder = os.path.expanduser(self.output_folder)
        os.makedirs(expanded_folder, exist_ok=True)
        with tarfile.open(expanded_tar, "r:*") as tar:
            tar.extractall(expanded_folder)
        return expanded_folder


class ListTar(BaseNode):
    """
    List contents of a tar archive.
    files, tar, list

    Use cases:
    - Inspect archives without extracting
    - Preview tar contents
    - Verify archive contents
    """

    tar_path: str = Field(default="", description="Tar archive to inspect")

    async def process(self, context: ProcessingContext) -> list[str]:
        if Environment.is_production():
            raise ValueError("This node is not available in production")
        if not self.tar_path:
            raise ValueError("tar_path cannot be empty")

        expanded_tar = os.path.expanduser(self.tar_path)
        with tarfile.open(expanded_tar, "r:*") as tar:
            return tar.getnames()
