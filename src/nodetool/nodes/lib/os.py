import csv
import os
import shutil
import glob
import tarfile
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, TypedDict
import pandas as pd
from pydantic import Field
from nodetool.config.environment import Environment
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.metadata.types import Datetime

import subprocess
from nodetool.io.uri_utils import create_file_uri


class WorkspaceDirectory(BaseNode):
    """
    Get the workspace directory.
    files, workspace, directory
    """

    async def process(self, context: ProcessingContext) -> str:
        return context.workspace_dir


class OpenWorkspaceDirectory(BaseNode):
    """
    Open the workspace directory.
    files, workspace, directory
    """

    async def process(self, context: ProcessingContext) -> None:
        if Environment.is_production():
            raise ValueError("This node is not available in production")
        subprocess.run(["open", context.workspace_dir])



class FileExists(BaseNode):
    """
    Check if a file or directory exists at the specified path.
    files, check, exists

    Use cases:
    - Validate file presence before processing
    - Implement conditional logic based on file existence
    """

    @classmethod
    def is_cacheable(cls) -> bool:
        return False

    path: str = Field(default="", description="Path to check for existence")

    async def process(self, context: ProcessingContext) -> bool:
        if Environment.is_production():
            raise ValueError("This node is not available in production")
        if not self.path:
            raise ValueError("'path' field cannot be empty")
        expanded_path = os.path.expanduser(self.path)
        return os.path.exists(expanded_path)


class ListFiles(BaseNode):
    """
    list files in a directory matching a pattern.
    files, list, directory

    Use cases:
    - Get files for batch processing
    - Filter files by extension or pattern
    """

    @classmethod
    def is_cacheable(cls) -> bool:
        return False

    folder: str = Field(default="~", description="Directory to scan")
    pattern: str = Field(default="*", description="File pattern to match (e.g. *.txt)")
    include_subdirectories: bool = Field(
        default=False, description="Search subdirectories"
    )

    class OutputType(TypedDict):
        file: str

    async def gen_process(
        self, context: ProcessingContext
    ) -> AsyncGenerator[OutputType, None]:
        if Environment.is_production():
            raise ValueError("This node is not available in production")
        if not self.folder:
            raise ValueError("directory cannot be empty")
        expanded_directory = os.path.expanduser(self.folder)

        if self.include_subdirectories:
            pattern = os.path.join(expanded_directory, "**", self.pattern)
            paths = glob.glob(pattern, recursive=True)
        else:
            pattern = os.path.join(expanded_directory, self.pattern)
            paths = glob.glob(pattern)

        for p in paths:
            yield {"file": p}


class CopyFile(BaseNode):
    """
    Copy a file from source to destination path.
    files, copy, manage

    Use cases:
    - Create file backups
    - Duplicate files for processing
    - Copy files to new locations
    """

    source_path: str = Field(default="", description="Source file path")
    destination_path: str = Field(default="", description="Destination file path")

    async def process(self, context: ProcessingContext):
        if Environment.is_production():
            raise ValueError("This node is not available in production")
        if not self.source_path:
            raise ValueError("'source_path' field cannot be empty")
        if not self.destination_path:
            raise ValueError("'destination_path' field cannot be empty")
        expanded_source = os.path.expanduser(self.source_path)
        expanded_dest = os.path.expanduser(self.destination_path)

        shutil.copy2(expanded_source, expanded_dest)


class MoveFile(BaseNode):
    """
    Move a file from source to destination path.
    files, move, manage

    Use cases:
    - Organize files into directories
    - Process and archive files
    - Relocate completed files
    """

    source_path: str = Field(default="", description="Source file path")
    destination_path: str = Field(default="", description="Destination file path")

    async def process(self, context: ProcessingContext):
        if Environment.is_production():
            raise ValueError("This node is not available in production")
        expanded_source = os.path.expanduser(self.source_path)
        expanded_dest = os.path.expanduser(self.destination_path)

        shutil.move(expanded_source, expanded_dest)


class CreateDirectory(BaseNode):
    """
    Create a new directory at specified path.
    files, directory, create

    Use cases:
    - Set up directory structure for file organization
    - Create output directories for processed files
    """

    path: str = Field(default="", description="Directory path to create")
    exist_ok: bool = Field(
        default=True, description="Don't error if directory already exists"
    )

    async def process(self, context: ProcessingContext):
        if Environment.is_production():
            raise ValueError("This node is not available in production")
        if not self.path:
            raise ValueError("'path' field cannot be empty")
        expanded_path = os.path.expanduser(self.path)
        os.makedirs(expanded_path, exist_ok=self.exist_ok)


class GetFileSize(BaseNode):
    """
    Get file size in bytes.
    files, metadata, size
    """

    @classmethod
    def is_cacheable(cls) -> bool:
        return False

    path: str = Field(default="", description="Path to file")

    async def process(self, context: ProcessingContext) -> int:
        if Environment.is_production():
            raise ValueError("This node is not available in production")
        if not self.path:
            raise ValueError("'path' field cannot be empty")
        expanded_path = os.path.expanduser(self.path)
        stats = os.stat(expanded_path)
        return stats.st_size


class CreatedTime(BaseNode):
    """
    Get file creation timestamp.
    files, metadata, created, time
    """

    @classmethod
    def is_cacheable(cls) -> bool:
        return False

    path: str = Field(default="", description="Path to file")

    async def process(self, context: ProcessingContext) -> Datetime:
        if Environment.is_production():
            raise ValueError("This node is not available in production")
        if not self.path:
            raise ValueError("'path' field cannot be empty")
        expanded_path = os.path.expanduser(self.path)
        stats = os.stat(expanded_path)
        return Datetime.from_datetime(datetime.fromtimestamp(stats.st_ctime))


class ModifiedTime(BaseNode):
    """
    Get file last modified timestamp.
    files, metadata, modified, time
    """

    @classmethod
    def is_cacheable(cls) -> bool:
        return False

    path: str = Field(default="", description="Path to file")

    async def process(self, context: ProcessingContext) -> Datetime:
        if Environment.is_production():
            raise ValueError("This node is not available in production")
        if not self.path:
            raise ValueError("'path' field cannot be empty")
        expanded_path = os.path.expanduser(self.path)
        stats = os.stat(expanded_path)
        return Datetime.from_datetime(datetime.fromtimestamp(stats.st_mtime))


class AccessedTime(BaseNode):
    """
    Get file last accessed timestamp.
    files, metadata, accessed, time
    """

    @classmethod
    def is_cacheable(cls) -> bool:
        return False

    path: str = Field(default="", description="Path to file")

    async def process(self, context: ProcessingContext) -> Datetime:
        if Environment.is_production():
            raise ValueError("This node is not available in production")
        if not self.path:
            raise ValueError("'path' field cannot be empty")
        expanded_path = os.path.expanduser(self.path)
        stats = os.stat(expanded_path)
        return Datetime.from_datetime(datetime.fromtimestamp(stats.st_atime))


class IsFile(BaseNode):
    """
    Check if path is a file.
    files, metadata, type
    """

    @classmethod
    def is_cacheable(cls) -> bool:
        return False

    path: str = Field(default="", description="Path to check")

    async def process(self, context: ProcessingContext) -> bool:
        if Environment.is_production():
            raise ValueError("This node is not available in production")
        if not self.path:
            raise ValueError("'path' field cannot be empty")
        expanded_path = os.path.expanduser(self.path)
        return os.path.isfile(expanded_path)


class IsDirectory(BaseNode):
    """
    Check if path is a directory.
    files, metadata, type
    """

    @classmethod
    def is_cacheable(cls) -> bool:
        return False

    path: str = Field(default="", description="Path to check")

    async def process(self, context: ProcessingContext) -> bool:
        if Environment.is_production():
            raise ValueError("This node is not available in production")
        if not self.path:
            raise ValueError("'path' field cannot be empty")
        expanded_path = os.path.expanduser(self.path)
        return os.path.isdir(expanded_path)


class FileExtension(BaseNode):
    """
    Get file extension.
    files, metadata, extension
    """

    path: str = Field(default="", description="Path to file")

    async def process(self, context: ProcessingContext) -> str:
        if Environment.is_production():
            raise ValueError("This node is not available in production")
        if not self.path:
            raise ValueError("'path' field cannot be empty")
        expanded_path = os.path.expanduser(self.path)
        return os.path.splitext(expanded_path)[1]


class FileName(BaseNode):
    """
    Get file name without path.
    files, metadata, name
    """

    path: str = Field(default="", description="Path to file")

    async def process(self, context: ProcessingContext) -> str:
        if Environment.is_production():
            raise ValueError("This node is not available in production")
        if not self.path:
            raise ValueError("'path' field cannot be empty")
        expanded_path = os.path.expanduser(self.path)
        return os.path.basename(expanded_path)


class GetDirectory(BaseNode):
    """
    Get directory containing the file.
    files, metadata, directory
    """

    path: str = Field(default="", description="Path to file")

    async def process(self, context: ProcessingContext) -> str:
        if Environment.is_production():
            raise ValueError("This node is not available in production")
        if not self.path:
            raise ValueError("'path' field cannot be empty")
        expanded_path = os.path.expanduser(self.path)
        return os.path.dirname(expanded_path)


class FileNameMatch(BaseNode):
    """
    Match a filename against a pattern using Unix shell-style wildcards.
    files, pattern, match, filter

    Use cases:
    - Filter files by name pattern
    - Validate file naming conventions
    - Match file extensions
    """

    filename: str = Field(default="", description="Filename to check")
    pattern: str = Field(
        default="*", description="Pattern to match against (e.g. *.txt, data_*.csv)"
    )
    case_sensitive: bool = Field(
        default=True,
        description="Whether the pattern matching should be case-sensitive",
    )

    async def process(self, context: ProcessingContext) -> bool:
        if Environment.is_production():
            raise ValueError("This node is not available in production")
        import fnmatch

        if self.case_sensitive:
            return fnmatch.fnmatch(self.filename, self.pattern)
        return fnmatch.fnmatchcase(self.filename, self.pattern)


class FilterFileNames(BaseNode):
    """
    Filter a list of filenames using Unix shell-style wildcards.
    files, pattern, filter, list

    Use cases:
    - Filter multiple files by pattern
    - Batch process files matching criteria
    - Select files by extension
    """

    filenames: list[str] = Field(default=[], description="list of filenames to filter")
    pattern: str = Field(
        default="*", description="Pattern to filter by (e.g. *.txt, data_*.csv)"
    )
    case_sensitive: bool = Field(
        default=True,
        description="Whether the pattern matching should be case-sensitive",
    )

    async def process(self, context: ProcessingContext) -> list[str]:
        if Environment.is_production():
            raise ValueError("This node is not available in production")
        import fnmatch

        if self.case_sensitive:
            return fnmatch.filter(self.filenames, self.pattern)
        return [
            name
            for name in self.filenames
            if fnmatch.fnmatchcase(name.lower(), self.pattern.lower())
        ]


class Basename(BaseNode):
    """
    Get the base name component of a file path.
    files, path, basename

    Use cases:
    - Extract filename from full path
    - Get file name without directory
    - Process file names independently
    """

    path: str = Field(default="", description="File path to get basename from")
    remove_extension: bool = Field(
        default=False, description="Remove file extension from basename"
    )

    async def process(self, context: ProcessingContext) -> str:
        if Environment.is_production():
            raise ValueError("This node is not available in production")
        if self.path.strip() == "":
            raise ValueError("path is empty")
        expanded_path = os.path.expanduser(self.path)
        basename = os.path.basename(expanded_path)
        if self.remove_extension:
            return os.path.splitext(basename)[0]
        return basename


class Dirname(BaseNode):
    """
    Get the directory name component of a file path.
    files, path, dirname

    Use cases:
    - Extract directory path from full path
    - Get parent directory
    - Process directory paths
    """

    path: str = Field(default="", description="File path to get dirname from")

    async def process(self, context: ProcessingContext) -> str:
        if Environment.is_production():
            raise ValueError("This node is not available in production")
        expanded_path = os.path.expanduser(self.path)
        return os.path.dirname(expanded_path)


class JoinPaths(BaseNode):
    """
    Joins path components.
    path, join, combine

    Use cases:
    - Build file paths
    - Create cross-platform paths
    """

    paths: list[str] = Field(default=[], description="Path components to join")

    async def process(self, context: ProcessingContext) -> str:
        if Environment.is_production():
            raise ValueError("This node is not available in production")
        if not self.paths:
            raise ValueError("paths cannot be empty")
        return os.path.join(*self.paths)


class NormalizePath(BaseNode):
    """
    Normalizes a path.
    path, normalize, clean

    Use cases:
    - Standardize paths
    - Remove redundant separators
    """

    path: str = Field(default="", description="Path to normalize")

    async def process(self, context: ProcessingContext) -> str:
        if Environment.is_production():
            raise ValueError("This node is not available in production")
        if not self.path:
            raise ValueError("path cannot be empty")
        return os.path.normpath(self.path)


class GetPathInfo(BaseNode):
    """
    Gets information about a path.
    path, info, metadata

    Use cases:
    - Extract path components
    - Parse file paths
    """

    path: str = Field(default="", description="Path to analyze")

    async def process(self, context: ProcessingContext) -> dict:
        if Environment.is_production():
            raise ValueError("This node is not available in production")
        return {
            "dirname": os.path.dirname(self.path),
            "basename": os.path.basename(self.path),
            "extension": os.path.splitext(self.path)[1],
            "absolute": os.path.abspath(self.path),
            "exists": os.path.exists(self.path),
            "is_file": os.path.isfile(self.path),
            "is_dir": os.path.isdir(self.path),
            "is_symlink": os.path.islink(self.path),
        }


class AbsolutePath(BaseNode):
    """
    Return the absolute path of a file or directory.
    files, path, absolute

    Use cases:
    - Convert relative paths to absolute
    - Get full system path
    - Resolve path references
    """

    path: str = Field(default="", description="Path to convert to absolute")

    async def process(self, context: ProcessingContext) -> str:
        if Environment.is_production():
            raise ValueError("This node is not available in production")
        expanded_path = os.path.expanduser(self.path)
        if not expanded_path:
            raise ValueError("path cannot be empty")
        return os.path.abspath(expanded_path)


class SplitPath(BaseNode):
    """
    Split a path into directory and file components.
    files, path, split

    Use cases:
    - Separate directory from filename
    - Process path components separately
    - Extract path parts
    """

    path: str = Field(default="", description="Path to split")

    async def process(self, context: ProcessingContext) -> dict:
        if Environment.is_production():
            raise ValueError("This node is not available in production")
        expanded_path = os.path.expanduser(self.path)
        dirname, basename = os.path.split(expanded_path)
        return {"dirname": dirname, "basename": basename}


class SplitExtension(BaseNode):
    """
    Split a path into root and extension components.
    files, path, extension, split

    Use cases:
    - Extract file extension
    - Process filename without extension
    - Handle file types
    """

    path: str = Field(default="", description="Path to split")

    async def process(self, context: ProcessingContext) -> dict:
        if Environment.is_production():
            raise ValueError("This node is not available in production")
        expanded_path = os.path.expanduser(self.path)
        root, ext = os.path.splitext(expanded_path)
        return {"root": root, "extension": ext}


class RelativePath(BaseNode):
    """
    Return a relative path to a target from a start directory.
    files, path, relative

    Use cases:
    - Create relative path references
    - Generate portable paths
    - Compare file locations
    """

    target_path: str = Field(
        default="", description="Target path to convert to relative"
    )
    start_path: str = Field(
        default=".", description="Start path for relative conversion"
    )

    async def process(self, context: ProcessingContext) -> str:
        if Environment.is_production():
            raise ValueError("This node is not available in production")
        expanded_target = os.path.expanduser(self.target_path)
        expanded_start = os.path.expanduser(self.start_path)
        if not expanded_target:
            raise ValueError("target_path cannot be empty")
        return os.path.relpath(expanded_target, expanded_start)


class PathToString(BaseNode):
    """
    Convert a FilePath object to a string.
    files, path, string, convert

    Use cases:
    - Get raw string path from FilePath object
    - Convert FilePath for string operations
    - Extract path string for external use
    """

    file_path: str = Field(default="", description="File path to convert to string")

    async def process(self, context: ProcessingContext) -> str:
        if Environment.is_production():
            raise ValueError("This node is not available in production")
        if not self.file_path:
            raise ValueError("file_path cannot be empty")
        return self.file_path


class ShowNotification(BaseNode):
    """
    Shows a system notification.
    notification, system, alert

    Use cases:
    - Alert user of completed tasks
    - Show process status
    - Display important messages
    """

    title: str = Field(default="", description="Title of the notification")
    message: str = Field(default="", description="Content of the notification")
    timeout: int = Field(
        default=10,
        description="How long the notification should stay visible (in seconds)",
    )

    @classmethod
    def is_cacheable(cls) -> bool:
        return False

    async def process(self, context: ProcessingContext) -> None:
        if Environment.is_production():
            raise ValueError("This node is not available in production")
        if not self.title:
            raise ValueError("title cannot be empty")
        if not self.message:
            raise ValueError("message cannot be empty")

        if os.name == "posix" and "darwin" in os.uname().sysname.lower():  # macOS
            # Escape single quotes in the title and message
            escaped_title = self.title.replace("'", "'\"'\"'")
            escaped_message = self.message.replace("'", "'\"'\"'")

            cmd = [
                "osascript",
                "-e",
                f'display notification "{escaped_message}" with title "{escaped_title}"',
            ]

            try:
                subprocess.run(cmd, check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Failed to show notification: {e.stderr.decode()}")

        else:  # Windows and Linux
            from plyer import notification

            notification.notify(
                title=self.title, message=self.message, timeout=self.timeout
            )  # type: ignore
