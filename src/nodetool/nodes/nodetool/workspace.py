"""
Workspace nodes for file operations within the current workflow workspace.

All nodes in this module operate exclusively within the current workspace directory
for security and isolation. They automatically use context.workspace_dir as the base path.
"""

import os
from typing import AsyncGenerator, TypedDict
from pydantic import Field
from nodetool.metadata.types import ImageRef, VideoRef
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import SaveUpdate
from nodetool.io.uri_utils import create_file_uri
from datetime import datetime
import datetime as dt


def _validate_workspace_path(workspace_dir: str, relative_path: str) -> str:
    """
    Validate that a relative path stays within the workspace.
    Returns the absolute path if valid, raises ValueError if invalid.
    """
    if not relative_path:
        raise ValueError("Path cannot be empty")

    # Prevent absolute paths
    if os.path.isabs(relative_path):
        raise ValueError(
            "Absolute paths are not allowed. Use relative paths within workspace."
        )

    # Prevent parent directory traversal
    if ".." in relative_path.split(os.sep):
        raise ValueError("Parent directory traversal (..) is not allowed")

    # Construct full path and verify it's within workspace
    full_path = os.path.abspath(os.path.join(workspace_dir, relative_path))
    if not full_path.startswith(os.path.abspath(workspace_dir)):
        raise ValueError("Path must be within workspace directory")

    return full_path


class GetWorkspaceDir(BaseNode):
    """
    Get the current workspace directory path.
    workspace, directory, path
    """

    async def process(self, context: ProcessingContext) -> str:
        return context.workspace_dir


class ListWorkspaceFiles(BaseNode):
    """
    List files in the workspace directory matching a pattern.
    workspace, files, list, directory
    """

    @classmethod
    def is_cacheable(cls) -> bool:
        return False

    path: str = Field(
        default=".",
        description="Relative path within workspace (use . for workspace root)",
    )
    pattern: str = Field(
        default="*", description="File pattern to match (e.g. *.txt, *.json)"
    )
    recursive: bool = Field(
        default=False, description="Search subdirectories recursively"
    )

    class OutputType(TypedDict):
        file: str

    async def gen_process(
        self, context: ProcessingContext
    ) -> AsyncGenerator[OutputType, None]:
        import glob

        workspace_dir = context.workspace_dir
        full_path = _validate_workspace_path(workspace_dir, self.path)

        if self.recursive:
            pattern_path = os.path.join(full_path, "**", self.pattern)
            paths = glob.glob(pattern_path, recursive=True)
        else:
            pattern_path = os.path.join(full_path, self.pattern)
            paths = glob.glob(pattern_path)

        # Return relative paths from workspace root
        for p in paths:
            rel_path = os.path.relpath(p, workspace_dir)
            yield {"file": rel_path}


class ReadTextFile(BaseNode):
    """
    Read a text file from the workspace.
    workspace, file, read, text
    """

    path: str = Field(default="", description="Relative path to file within workspace")
    encoding: str = Field(
        default="utf-8", description="Text encoding (utf-8, ascii, etc.)"
    )

    async def process(self, context: ProcessingContext) -> str:
        full_path = _validate_workspace_path(context.workspace_dir, self.path)

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"File not found: {self.path}")

        if not os.path.isfile(full_path):
            raise ValueError(f"Path is not a file: {self.path}")

        with open(full_path, "r", encoding=self.encoding) as f:
            return f.read()


class WriteTextFile(BaseNode):
    """
    Write text to a file in the workspace.
    workspace, file, write, text, save
    """

    path: str = Field(default="", description="Relative path to file within workspace")
    content: str = Field(default="", description="Text content to write")
    encoding: str = Field(
        default="utf-8", description="Text encoding (utf-8, ascii, etc.)"
    )
    append: bool = Field(
        default=False, description="Append to file instead of overwriting"
    )

    async def process(self, context: ProcessingContext) -> str:
        full_path = _validate_workspace_path(context.workspace_dir, self.path)

        # Create parent directories if needed
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        mode = "a" if self.append else "w"
        with open(full_path, mode, encoding=self.encoding) as f:
            f.write(self.content)

        return self.path


class ReadBinaryFile(BaseNode):
    """
    Read a binary file from the workspace as base64-encoded string.
    workspace, file, read, binary
    """

    path: str = Field(default="", description="Relative path to file within workspace")

    async def process(self, context: ProcessingContext) -> str:
        import base64

        full_path = _validate_workspace_path(context.workspace_dir, self.path)

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"File not found: {self.path}")

        if not os.path.isfile(full_path):
            raise ValueError(f"Path is not a file: {self.path}")

        with open(full_path, "rb") as f:
            data = f.read()
            return base64.b64encode(data).decode("ascii")


class WriteBinaryFile(BaseNode):
    """
    Write binary data (base64-encoded) to a file in the workspace.
    workspace, file, write, binary, save
    """

    path: str = Field(default="", description="Relative path to file within workspace")
    content: str = Field(
        default="", description="Base64-encoded binary content to write"
    )

    async def process(self, context: ProcessingContext) -> str:
        import base64

        full_path = _validate_workspace_path(context.workspace_dir, self.path)

        # Create parent directories if needed
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        # Decode base64 and write
        data = base64.b64decode(self.content)
        with open(full_path, "wb") as f:
            f.write(data)

        return self.path


class DeleteWorkspaceFile(BaseNode):
    """
    Delete a file or directory from the workspace.
    workspace, file, delete, remove
    """

    @classmethod
    def is_cacheable(cls) -> bool:
        return False

    path: str = Field(
        default="", description="Relative path to file or directory within workspace"
    )
    recursive: bool = Field(default=False, description="Delete directories recursively")

    async def process(self, context: ProcessingContext) -> None:
        import shutil

        full_path = _validate_workspace_path(context.workspace_dir, self.path)

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Path not found: {self.path}")

        if os.path.isdir(full_path):
            if not self.recursive:
                raise ValueError(
                    f"Path is a directory. Set recursive=True to delete: {self.path}"
                )
            shutil.rmtree(full_path)
        else:
            os.remove(full_path)


class CreateWorkspaceDirectory(BaseNode):
    """
    Create a directory in the workspace.
    workspace, directory, create, folder
    """

    path: str = Field(
        default="", description="Relative path to directory within workspace"
    )

    async def process(self, context: ProcessingContext) -> str:
        full_path = _validate_workspace_path(context.workspace_dir, self.path)
        os.makedirs(full_path, exist_ok=True)
        return self.path


class WorkspaceFileExists(BaseNode):
    """
    Check if a file or directory exists in the workspace.
    workspace, file, exists, check
    """

    @classmethod
    def is_cacheable(cls) -> bool:
        return False

    path: str = Field(default="", description="Relative path within workspace to check")

    async def process(self, context: ProcessingContext) -> bool:
        full_path = _validate_workspace_path(context.workspace_dir, self.path)
        return os.path.exists(full_path)


class GetWorkspaceFileInfo(BaseNode):
    """
    Get information about a file in the workspace.
    workspace, file, info, metadata
    """

    @classmethod
    def is_cacheable(cls) -> bool:
        return False

    path: str = Field(default="", description="Relative path to file within workspace")

    async def process(self, context: ProcessingContext) -> dict:
        full_path = _validate_workspace_path(context.workspace_dir, self.path)

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Path not found: {self.path}")

        stats = os.stat(full_path)

        return {
            "path": self.path,
            "name": os.path.basename(self.path),
            "size": stats.st_size,
            "is_file": os.path.isfile(full_path),
            "is_directory": os.path.isdir(full_path),
            "created": datetime.fromtimestamp(stats.st_ctime).isoformat(),
            "modified": datetime.fromtimestamp(stats.st_mtime).isoformat(),
            "accessed": datetime.fromtimestamp(stats.st_atime).isoformat(),
        }


class CopyWorkspaceFile(BaseNode):
    """
    Copy a file within the workspace.
    workspace, file, copy, duplicate
    """

    source: str = Field(default="", description="Relative source path within workspace")
    destination: str = Field(
        default="", description="Relative destination path within workspace"
    )

    async def process(self, context: ProcessingContext) -> str:
        import shutil

        source_path = _validate_workspace_path(context.workspace_dir, self.source)
        dest_path = _validate_workspace_path(context.workspace_dir, self.destination)

        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Source file not found: {self.source}")

        # Create parent directories if needed
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        if os.path.isdir(source_path):
            shutil.copytree(source_path, dest_path)
        else:
            shutil.copy2(source_path, dest_path)

        return self.destination


class MoveWorkspaceFile(BaseNode):
    """
    Move or rename a file within the workspace.
    workspace, file, move, rename
    """

    source: str = Field(default="", description="Relative source path within workspace")
    destination: str = Field(
        default="", description="Relative destination path within workspace"
    )

    async def process(self, context: ProcessingContext) -> str:
        import shutil

        source_path = _validate_workspace_path(context.workspace_dir, self.source)
        dest_path = _validate_workspace_path(context.workspace_dir, self.destination)

        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Source file not found: {self.source}")

        # Create parent directories if needed
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        shutil.move(source_path, dest_path)

        return self.destination


class GetWorkspaceFileSize(BaseNode):
    """
    Get file size in bytes for a workspace file.
    workspace, file, size, bytes
    """

    @classmethod
    def is_cacheable(cls) -> bool:
        return False

    path: str = Field(default="", description="Relative path to file within workspace")

    async def process(self, context: ProcessingContext) -> int:
        full_path = _validate_workspace_path(context.workspace_dir, self.path)

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"File not found: {self.path}")

        if not os.path.isfile(full_path):
            raise ValueError(f"Path is not a file: {self.path}")

        return os.path.getsize(full_path)


class IsWorkspaceFile(BaseNode):
    """
    Check if a path in the workspace is a file.
    workspace, file, check, type
    """

    @classmethod
    def is_cacheable(cls) -> bool:
        return False

    path: str = Field(default="", description="Relative path within workspace to check")

    async def process(self, context: ProcessingContext) -> bool:
        full_path = _validate_workspace_path(context.workspace_dir, self.path)
        return os.path.isfile(full_path)


class IsWorkspaceDirectory(BaseNode):
    """
    Check if a path in the workspace is a directory.
    workspace, directory, check, type
    """

    @classmethod
    def is_cacheable(cls) -> bool:
        return False

    path: str = Field(default="", description="Relative path within workspace to check")

    async def process(self, context: ProcessingContext) -> bool:
        full_path = _validate_workspace_path(context.workspace_dir, self.path)
        return os.path.isdir(full_path)


class JoinWorkspacePaths(BaseNode):
    """
    Join path components relative to workspace.
    workspace, path, join, combine
    """

    paths: list[str] = Field(
        default=[], description="Path components to join (relative to workspace)"
    )

    async def process(self, context: ProcessingContext) -> str:
        if not self.paths:
            raise ValueError("paths cannot be empty")

        # Join paths and validate result
        joined = os.path.join(*self.paths)
        _validate_workspace_path(context.workspace_dir, joined)

        return joined


class SaveImageFile(BaseNode):
    """
    Save an image to a file in the workspace.
    workspace, image, save, file, output
    """

    image: ImageRef = Field(default=ImageRef(), description="The image to save")
    folder: str = Field(
        default=".",
        description="Relative folder path within workspace (use . for workspace root)",
    )
    filename: str = Field(
        default="image.png",
        description="""
        The name of the image file.
        You can use time and date variables to create unique names:
        %Y - Year
        %m - Month
        %d - Day
        %H - Hour
        %M - Minute
        %S - Second
        """,
    )
    overwrite: bool = Field(
        default=False,
        description="Overwrite the file if it already exists, otherwise file will be renamed",
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        if not self.filename:
            raise ValueError("filename cannot be empty")

        # Format filename with current date/time
        filename = dt.datetime.now().strftime(self.filename)

        # Build and validate the full path
        relative_path = os.path.join(self.folder, filename)
        full_path = _validate_workspace_path(context.workspace_dir, relative_path)

        # Create parent directories if needed
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        # Handle filename conflicts if not overwriting
        if not self.overwrite:
            count = 1
            while os.path.exists(full_path):
                fname, ext = os.path.splitext(filename)
                new_filename = f"{fname}_{count}{ext}"
                relative_path = os.path.join(self.folder, new_filename)
                full_path = _validate_workspace_path(
                    context.workspace_dir, relative_path
                )
                count += 1
                filename = new_filename

        # Save the image
        image = await context.image_to_pil(self.image)
        image.save(full_path)

        result = ImageRef(uri=create_file_uri(full_path), data=image.tobytes())

        # Emit SaveUpdate event
        context.post_message(
            SaveUpdate(
                node_id=self.id, name=filename, value=result, output_type="image"
            )
        )

        return result


class SaveVideoFile(BaseNode):
    """
    Save a video file to the workspace.
    workspace, video, save, file, output

    The filename can include time and date variables:
    %Y - Year, %m - Month, %d - Day
    %H - Hour, %M - Minute, %S - Second
    """

    video: VideoRef = Field(default=VideoRef(), description="The video to save")
    folder: str = Field(
        default=".",
        description="Relative folder path within workspace (use . for workspace root)",
    )
    filename: str = Field(
        default="video.mp4",
        description="""
        Name of the file to save.
        You can use time and date variables to create unique names:
        %Y - Year
        %m - Month
        %d - Day
        %H - Hour
        %M - Minute
        %S - Second
        """,
    )
    overwrite: bool = Field(
        default=False,
        description="Overwrite the file if it already exists, otherwise file will be renamed",
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        if not self.filename:
            raise ValueError("filename cannot be empty")

        # Format filename with current date/time
        filename = dt.datetime.now().strftime(self.filename)

        # Build and validate the full path
        relative_path = os.path.join(self.folder, filename)
        full_path = _validate_workspace_path(context.workspace_dir, relative_path)

        # Create parent directories if needed
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        # Handle filename conflicts if not overwriting
        if not self.overwrite:
            count = 1
            while os.path.exists(full_path):
                fname, ext = os.path.splitext(filename)
                new_filename = f"{fname}_{count}{ext}"
                relative_path = os.path.join(self.folder, new_filename)
                full_path = _validate_workspace_path(
                    context.workspace_dir, relative_path
                )
                count += 1
                filename = new_filename

        # Save the video
        video_io = await context.asset_to_io(self.video)
        video_data = video_io.read()

        with open(full_path, "wb") as f:
            f.write(video_data)

        result = VideoRef(uri=create_file_uri(full_path), data=video_data)

        # Emit SaveUpdate event
        context.post_message(
            SaveUpdate(
                node_id=self.id, name=filename, value=result, output_type="video"
            )
        )

        return result
