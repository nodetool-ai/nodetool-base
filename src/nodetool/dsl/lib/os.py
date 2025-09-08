from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode


class AbsolutePath(GraphNode):
    """
    Return the absolute path of a file or directory.
    files, path, absolute

    Use cases:
    - Convert relative paths to absolute
    - Get full system path
    - Resolve path references
    """

    path: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="Path to convert to absolute"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.os.AbsolutePath"


class AccessedTime(GraphNode):
    """
    Get file last accessed timestamp.
    files, metadata, accessed, time
    """

    path: types.FilePath | GraphNode | tuple[GraphNode, str] = Field(
        default=types.FilePath(type="file_path", path=""), description="Path to file"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.os.AccessedTime"


class Basename(GraphNode):
    """
    Get the base name component of a file path.
    files, path, basename

    Use cases:
    - Extract filename from full path
    - Get file name without directory
    - Process file names independently
    """

    path: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="File path to get basename from"
    )
    remove_extension: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=False, description="Remove file extension from basename"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.os.Basename"


class CopyFile(GraphNode):
    """
    Copy a file from source to destination path.
    files, copy, manage

    Use cases:
    - Create file backups
    - Duplicate files for processing
    - Copy files to new locations
    """

    source_path: types.FilePath | GraphNode | tuple[GraphNode, str] = Field(
        default=types.FilePath(type="file_path", path=""),
        description="Source file path",
    )
    destination_path: types.FilePath | GraphNode | tuple[GraphNode, str] = Field(
        default=types.FilePath(type="file_path", path=""),
        description="Destination file path",
    )

    @classmethod
    def get_node_type(cls):
        return "lib.os.CopyFile"


class CreateDirectory(GraphNode):
    """
    Create a new directory at specified path.
    files, directory, create

    Use cases:
    - Set up directory structure for file organization
    - Create output directories for processed files
    """

    path: types.FilePath | GraphNode | tuple[GraphNode, str] = Field(
        default=types.FilePath(type="file_path", path=""),
        description="Directory path to create",
    )
    exist_ok: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=True, description="Don't error if directory already exists"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.os.CreateDirectory"


class CreatedTime(GraphNode):
    """
    Get file creation timestamp.
    files, metadata, created, time
    """

    path: types.FilePath | GraphNode | tuple[GraphNode, str] = Field(
        default=types.FilePath(type="file_path", path=""), description="Path to file"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.os.CreatedTime"


class Dirname(GraphNode):
    """
    Get the directory name component of a file path.
    files, path, dirname

    Use cases:
    - Extract directory path from full path
    - Get parent directory
    - Process directory paths
    """

    path: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="File path to get dirname from"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.os.Dirname"


class FileExists(GraphNode):
    """
    Check if a file or directory exists at the specified path.
    files, check, exists

    Use cases:
    - Validate file presence before processing
    - Implement conditional logic based on file existence
    """

    path: types.FilePath | GraphNode | tuple[GraphNode, str] = Field(
        default=types.FilePath(type="file_path", path=""),
        description="Path to check for existence",
    )

    @classmethod
    def get_node_type(cls):
        return "lib.os.FileExists"


class FileExtension(GraphNode):
    """
    Get file extension.
    files, metadata, extension
    """

    path: types.FilePath | GraphNode | tuple[GraphNode, str] = Field(
        default=types.FilePath(type="file_path", path=""), description="Path to file"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.os.FileExtension"


class FileName(GraphNode):
    """
    Get file name without path.
    files, metadata, name
    """

    path: types.FilePath | GraphNode | tuple[GraphNode, str] = Field(
        default=types.FilePath(type="file_path", path=""), description="Path to file"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.os.FileName"


class FileNameMatch(GraphNode):
    """
    Match a filename against a pattern using Unix shell-style wildcards.
    files, pattern, match, filter

    Use cases:
    - Filter files by name pattern
    - Validate file naming conventions
    - Match file extensions
    """

    filename: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="Filename to check"
    )
    pattern: str | GraphNode | tuple[GraphNode, str] = Field(
        default="*", description="Pattern to match against (e.g. *.txt, data_*.csv)"
    )
    case_sensitive: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=True,
        description="Whether the pattern matching should be case-sensitive",
    )

    @classmethod
    def get_node_type(cls):
        return "lib.os.FileNameMatch"


class FilterFileNames(GraphNode):
    """
    Filter a list of filenames using Unix shell-style wildcards.
    files, pattern, filter, list

    Use cases:
    - Filter multiple files by pattern
    - Batch process files matching criteria
    - Select files by extension
    """

    filenames: list[str] | GraphNode | tuple[GraphNode, str] = Field(
        default=[], description="list of filenames to filter"
    )
    pattern: str | GraphNode | tuple[GraphNode, str] = Field(
        default="*", description="Pattern to filter by (e.g. *.txt, data_*.csv)"
    )
    case_sensitive: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=True,
        description="Whether the pattern matching should be case-sensitive",
    )

    @classmethod
    def get_node_type(cls):
        return "lib.os.FilterFileNames"


class GetDirectory(GraphNode):
    """
    Get directory containing the file.
    files, metadata, directory
    """

    path: types.FilePath | GraphNode | tuple[GraphNode, str] = Field(
        default=types.FilePath(type="file_path", path=""), description="Path to file"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.os.GetDirectory"


class GetEnvironmentVariable(GraphNode):
    """
    Gets an environment variable value.
    environment, variable, system

    Use cases:
    - Access configuration
    - Get system settings
    """

    name: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="Environment variable name"
    )
    default: str | None | GraphNode | tuple[GraphNode, str] = Field(
        default=None, description="Default value if not found"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.os.GetEnvironmentVariable"


class GetFileSize(GraphNode):
    """
    Get file size in bytes.
    files, metadata, size
    """

    path: types.FilePath | GraphNode | tuple[GraphNode, str] = Field(
        default=types.FilePath(type="file_path", path=""), description="Path to file"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.os.GetFileSize"


class GetPathInfo(GraphNode):
    """
    Gets information about a path.
    path, info, metadata

    Use cases:
    - Extract path components
    - Parse file paths
    """

    path: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="Path to analyze"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.os.GetPathInfo"


class GetSystemInfo(GraphNode):
    """
    Gets system information.
    system, info, platform

    Use cases:
    - Check system compatibility
    - Platform-specific logic
    """

    @classmethod
    def get_node_type(cls):
        return "lib.os.GetSystemInfo"


class IsDirectory(GraphNode):
    """
    Check if path is a directory.
    files, metadata, type
    """

    path: types.FilePath | GraphNode | tuple[GraphNode, str] = Field(
        default=types.FilePath(type="file_path", path=""), description="Path to check"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.os.IsDirectory"


class IsFile(GraphNode):
    """
    Check if path is a file.
    files, metadata, type
    """

    path: types.FilePath | GraphNode | tuple[GraphNode, str] = Field(
        default=types.FilePath(type="file_path", path=""), description="Path to check"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.os.IsFile"


class JoinPaths(GraphNode):
    """
    Joins path components.
    path, join, combine

    Use cases:
    - Build file paths
    - Create cross-platform paths
    """

    paths: list[str] | GraphNode | tuple[GraphNode, str] = Field(
        default=[], description="Path components to join"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.os.JoinPaths"


class ListFiles(GraphNode):
    """
    list files in a directory matching a pattern.
    files, list, directory

    Use cases:
    - Get files for batch processing
    - Filter files by extension or pattern
    """

    folder: types.FolderPath | GraphNode | tuple[GraphNode, str] = Field(
        default=types.FolderPath(type="folder_path", path="~"),
        description="Directory to scan",
    )
    pattern: str | GraphNode | tuple[GraphNode, str] = Field(
        default="*", description="File pattern to match (e.g. *.txt)"
    )
    recursive: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=False, description="Search subdirectories"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.os.ListFiles"


class ModifiedTime(GraphNode):
    """
    Get file last modified timestamp.
    files, metadata, modified, time
    """

    path: types.FilePath | GraphNode | tuple[GraphNode, str] = Field(
        default=types.FilePath(type="file_path", path=""), description="Path to file"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.os.ModifiedTime"


class MoveFile(GraphNode):
    """
    Move a file from source to destination path.
    files, move, manage

    Use cases:
    - Organize files into directories
    - Process and archive files
    - Relocate completed files
    """

    source_path: types.FilePath | GraphNode | tuple[GraphNode, str] = Field(
        default=types.FilePath(type="file_path", path=""),
        description="Source file path",
    )
    destination_path: types.FilePath | GraphNode | tuple[GraphNode, str] = Field(
        default=types.FilePath(type="file_path", path=""),
        description="Destination file path",
    )

    @classmethod
    def get_node_type(cls):
        return "lib.os.MoveFile"


class NormalizePath(GraphNode):
    """
    Normalizes a path.
    path, normalize, clean

    Use cases:
    - Standardize paths
    - Remove redundant separators
    """

    path: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="Path to normalize"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.os.NormalizePath"


class OpenWorkspaceDirectory(GraphNode):
    """
    Open the workspace directory.
    files, workspace, directory
    """

    @classmethod
    def get_node_type(cls):
        return "lib.os.OpenWorkspaceDirectory"


class PathToString(GraphNode):
    """
    Convert a FilePath object to a string.
    files, path, string, convert

    Use cases:
    - Get raw string path from FilePath object
    - Convert FilePath for string operations
    - Extract path string for external use
    """

    file_path: types.FilePath | GraphNode | tuple[GraphNode, str] = Field(
        default=types.FilePath(type="file_path", path=""),
        description="FilePath object to convert to string",
    )

    @classmethod
    def get_node_type(cls):
        return "lib.os.PathToString"


class RelativePath(GraphNode):
    """
    Return a relative path to a target from a start directory.
    files, path, relative

    Use cases:
    - Create relative path references
    - Generate portable paths
    - Compare file locations
    """

    target_path: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="Target path to convert to relative"
    )
    start_path: str | GraphNode | tuple[GraphNode, str] = Field(
        default=".", description="Start path for relative conversion"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.os.RelativePath"


class SetEnvironmentVariable(GraphNode):
    """
    Sets an environment variable.
    environment, variable, system

    Use cases:
    - Configure runtime settings
    - Set up process environment
    """

    name: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="Environment variable name"
    )
    value: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="Environment variable value"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.os.SetEnvironmentVariable"


class ShowNotification(GraphNode):
    """
    Shows a system notification.
    notification, system, alert

    Use cases:
    - Alert user of completed tasks
    - Show process status
    - Display important messages
    """

    title: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="Title of the notification"
    )
    message: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="Content of the notification"
    )
    timeout: int | GraphNode | tuple[GraphNode, str] = Field(
        default=10,
        description="How long the notification should stay visible (in seconds)",
    )

    @classmethod
    def get_node_type(cls):
        return "lib.os.ShowNotification"


class SplitExtension(GraphNode):
    """
    Split a path into root and extension components.
    files, path, extension, split

    Use cases:
    - Extract file extension
    - Process filename without extension
    - Handle file types
    """

    path: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="Path to split"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.os.SplitExtension"


class SplitPath(GraphNode):
    """
    Split a path into directory and file components.
    files, path, split

    Use cases:
    - Separate directory from filename
    - Process path components separately
    - Extract path parts
    """

    path: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="Path to split"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.os.SplitPath"


class WorkspaceDirectory(GraphNode):
    """
    Get the workspace directory.
    files, workspace, directory
    """

    @classmethod
    def get_node_type(cls):
        return "lib.os.WorkspaceDirectory"
