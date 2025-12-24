import tarfile
import os
from pydantic import Field
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.base_node import BaseNode


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
        if not self.source_folder:
            raise ValueError("source_folder cannot be empty")
        if not self.tar_path:
            raise ValueError("tar_path cannot be empty")

        mode = "w:gz" if self.gzip else "w"
        with tarfile.open(self.tar_path, mode) as tar:
            tar.add(self.source_folder, arcname=os.path.basename(self.source_folder))

        return self.tar_path


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
        if not self.tar_path:
            raise ValueError("tar_path cannot be empty")
        if not self.output_folder:
            raise ValueError("output_folder cannot be empty")

        output_folder = os.path.abspath(self.output_folder)

        with tarfile.open(self.tar_path, "r:*") as tar:
            # Validate all members to prevent path traversal attacks
            for member in tar.getmembers():
                member_path = os.path.abspath(os.path.join(output_folder, member.name))
                if not member_path.startswith(output_folder):
                    raise ValueError(
                        f"Attempted path traversal in tar file: {member.name}"
                    )
            tar.extractall(path=output_folder)

        return self.output_folder


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
        if not self.tar_path:
            raise ValueError("tar_path cannot be empty")

        with tarfile.open(self.tar_path, "r:*") as tar:
            return tar.getnames()
