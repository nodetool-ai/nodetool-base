import pytest
import os
import tempfile
from unittest.mock import patch, MagicMock
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.nodes.lib.os import (
    FileExists,
    ListFiles,
    CopyFile,
    MoveFile,
    CreateDirectory,
    GetFileSize,
    IsFile,
    IsDirectory,
    FileExtension,
    FileName,
    GetDirectory,
    FileNameMatch,
    FilterFileNames,
    Basename,
    Dirname,
    JoinPaths,
    NormalizePath,
    GetPathInfo,
    AbsolutePath,
    SplitPath,
    SplitExtension,
    RelativePath,
    PathToString,
)


@pytest.fixture
def context():
    return ProcessingContext(user_id="test", auth_token="test")


@pytest.fixture
def temp_file():
    """Create a temporary file for testing."""
    fd, path = tempfile.mkstemp(suffix=".txt")
    os.write(fd, b"test content here")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    path = tempfile.mkdtemp()
    yield path
    if os.path.exists(path):
        import shutil
        shutil.rmtree(path)


class TestFileExists:
    """Tests for FileExists node."""

    @pytest.mark.asyncio
    async def test_existing_file(self, context: ProcessingContext, temp_file):
        with patch("nodetool.config.environment.Environment.is_production", return_value=False):
            node = FileExists(path=temp_file)
            result = await node.process(context)
            assert result is True

    @pytest.mark.asyncio
    async def test_nonexistent_file(self, context: ProcessingContext):
        with patch("nodetool.config.environment.Environment.is_production", return_value=False):
            node = FileExists(path="/nonexistent/path/file.txt")
            result = await node.process(context)
            assert result is False

    @pytest.mark.asyncio
    async def test_empty_path_raises(self, context: ProcessingContext):
        with patch("nodetool.config.environment.Environment.is_production", return_value=False):
            node = FileExists(path="")
            with pytest.raises(ValueError, match="cannot be empty"):
                await node.process(context)

    @pytest.mark.asyncio
    async def test_production_raises(self, context: ProcessingContext):
        with patch("nodetool.config.environment.Environment.is_production", return_value=True):
            node = FileExists(path="/some/path")
            with pytest.raises(ValueError, match="not available in production"):
                await node.process(context)


class TestGetFileSize:
    """Tests for GetFileSize node."""

    @pytest.mark.asyncio
    async def test_get_file_size(self, context: ProcessingContext, temp_file):
        with patch("nodetool.config.environment.Environment.is_production", return_value=False):
            node = GetFileSize(path=temp_file)
            result = await node.process(context)
            assert result == 17  # "test content here" is 17 bytes

    @pytest.mark.asyncio
    async def test_empty_path_raises(self, context: ProcessingContext):
        with patch("nodetool.config.environment.Environment.is_production", return_value=False):
            node = GetFileSize(path="")
            with pytest.raises(ValueError, match="cannot be empty"):
                await node.process(context)


class TestIsFile:
    """Tests for IsFile node."""

    @pytest.mark.asyncio
    async def test_is_file_true(self, context: ProcessingContext, temp_file):
        with patch("nodetool.config.environment.Environment.is_production", return_value=False):
            node = IsFile(path=temp_file)
            result = await node.process(context)
            assert result is True

    @pytest.mark.asyncio
    async def test_is_file_false_for_dir(self, context: ProcessingContext, temp_dir):
        with patch("nodetool.config.environment.Environment.is_production", return_value=False):
            node = IsFile(path=temp_dir)
            result = await node.process(context)
            assert result is False


class TestIsDirectory:
    """Tests for IsDirectory node."""

    @pytest.mark.asyncio
    async def test_is_directory_true(self, context: ProcessingContext, temp_dir):
        with patch("nodetool.config.environment.Environment.is_production", return_value=False):
            node = IsDirectory(path=temp_dir)
            result = await node.process(context)
            assert result is True

    @pytest.mark.asyncio
    async def test_is_directory_false_for_file(self, context: ProcessingContext, temp_file):
        with patch("nodetool.config.environment.Environment.is_production", return_value=False):
            node = IsDirectory(path=temp_file)
            result = await node.process(context)
            assert result is False


class TestFileExtension:
    """Tests for FileExtension node."""

    @pytest.mark.asyncio
    async def test_get_extension(self, context: ProcessingContext):
        with patch("nodetool.config.environment.Environment.is_production", return_value=False):
            node = FileExtension(path="/path/to/file.txt")
            result = await node.process(context)
            assert result == ".txt"

    @pytest.mark.asyncio
    async def test_no_extension(self, context: ProcessingContext):
        with patch("nodetool.config.environment.Environment.is_production", return_value=False):
            node = FileExtension(path="/path/to/file")
            result = await node.process(context)
            assert result == ""


class TestFileName:
    """Tests for FileName node."""

    @pytest.mark.asyncio
    async def test_get_filename(self, context: ProcessingContext):
        with patch("nodetool.config.environment.Environment.is_production", return_value=False):
            node = FileName(path="/path/to/myfile.txt")
            result = await node.process(context)
            assert result == "myfile.txt"


class TestGetDirectory:
    """Tests for GetDirectory node."""

    @pytest.mark.asyncio
    async def test_get_directory(self, context: ProcessingContext):
        with patch("nodetool.config.environment.Environment.is_production", return_value=False):
            node = GetDirectory(path="/path/to/myfile.txt")
            result = await node.process(context)
            assert result == "/path/to"


class TestFileNameMatch:
    """Tests for FileNameMatch node."""

    @pytest.mark.asyncio
    async def test_pattern_match_txt(self, context: ProcessingContext):
        with patch("nodetool.config.environment.Environment.is_production", return_value=False):
            node = FileNameMatch(filename="document.txt", pattern="*.txt")
            result = await node.process(context)
            assert result is True

    @pytest.mark.asyncio
    async def test_pattern_no_match(self, context: ProcessingContext):
        with patch("nodetool.config.environment.Environment.is_production", return_value=False):
            node = FileNameMatch(filename="document.pdf", pattern="*.txt")
            result = await node.process(context)
            assert result is False

    @pytest.mark.asyncio
    async def test_case_sensitive(self, context: ProcessingContext):
        with patch("nodetool.config.environment.Environment.is_production", return_value=False):
            node = FileNameMatch(filename="Document.TXT", pattern="*.txt", case_sensitive=True)
            result = await node.process(context)
            # In fnmatch, *.txt won't match .TXT case-sensitively
            # Actually fnmatch matches *.txt with .TXT on most systems; let's be conservative
            # The code uses fnmatchcase when case_sensitive=True which should be stricter
            assert result is False


class TestFilterFileNames:
    """Tests for FilterFileNames node."""

    @pytest.mark.asyncio
    async def test_filter_by_extension(self, context: ProcessingContext):
        with patch("nodetool.config.environment.Environment.is_production", return_value=False):
            files = ["a.txt", "b.pdf", "c.txt", "d.doc"]
            node = FilterFileNames(filenames=files, pattern="*.txt")
            result = await node.process(context)
            assert result == ["a.txt", "c.txt"]

    @pytest.mark.asyncio
    async def test_filter_with_prefix(self, context: ProcessingContext):
        with patch("nodetool.config.environment.Environment.is_production", return_value=False):
            files = ["data_001.csv", "data_002.csv", "report.csv"]
            node = FilterFileNames(filenames=files, pattern="data_*.csv")
            result = await node.process(context)
            assert result == ["data_001.csv", "data_002.csv"]


class TestBasename:
    """Tests for Basename node."""

    @pytest.mark.asyncio
    async def test_basename_with_extension(self, context: ProcessingContext):
        with patch("nodetool.config.environment.Environment.is_production", return_value=False):
            node = Basename(path="/path/to/file.txt")
            result = await node.process(context)
            assert result == "file.txt"

    @pytest.mark.asyncio
    async def test_basename_remove_extension(self, context: ProcessingContext):
        with patch("nodetool.config.environment.Environment.is_production", return_value=False):
            node = Basename(path="/path/to/file.txt", remove_extension=True)
            result = await node.process(context)
            assert result == "file"

    @pytest.mark.asyncio
    async def test_empty_path_raises(self, context: ProcessingContext):
        with patch("nodetool.config.environment.Environment.is_production", return_value=False):
            node = Basename(path="")
            with pytest.raises(ValueError, match="empty"):
                await node.process(context)


class TestDirname:
    """Tests for Dirname node."""

    @pytest.mark.asyncio
    async def test_dirname(self, context: ProcessingContext):
        with patch("nodetool.config.environment.Environment.is_production", return_value=False):
            node = Dirname(path="/path/to/file.txt")
            result = await node.process(context)
            assert result == "/path/to"


class TestJoinPaths:
    """Tests for JoinPaths node."""

    @pytest.mark.asyncio
    async def test_join_paths(self, context: ProcessingContext):
        with patch("nodetool.config.environment.Environment.is_production", return_value=False):
            node = JoinPaths(paths=["path", "to", "file.txt"])
            result = await node.process(context)
            assert result == "path/to/file.txt"

    @pytest.mark.asyncio
    async def test_empty_paths_raises(self, context: ProcessingContext):
        with patch("nodetool.config.environment.Environment.is_production", return_value=False):
            node = JoinPaths(paths=[])
            with pytest.raises(ValueError, match="cannot be empty"):
                await node.process(context)


class TestNormalizePath:
    """Tests for NormalizePath node."""

    @pytest.mark.asyncio
    async def test_normalize(self, context: ProcessingContext):
        with patch("nodetool.config.environment.Environment.is_production", return_value=False):
            node = NormalizePath(path="/path//to/../to/file.txt")
            result = await node.process(context)
            assert result == "/path/to/file.txt"


class TestGetPathInfo:
    """Tests for GetPathInfo node."""

    @pytest.mark.asyncio
    async def test_path_info(self, context: ProcessingContext, temp_file):
        with patch("nodetool.config.environment.Environment.is_production", return_value=False):
            node = GetPathInfo(path=temp_file)
            result = await node.process(context)
            assert "dirname" in result
            assert "basename" in result
            assert "extension" in result
            assert result["extension"] == ".txt"
            assert result["is_file"] is True


class TestAbsolutePath:
    """Tests for AbsolutePath node."""

    @pytest.mark.asyncio
    async def test_absolute_path(self, context: ProcessingContext):
        with patch("nodetool.config.environment.Environment.is_production", return_value=False):
            node = AbsolutePath(path="relative/path")
            result = await node.process(context)
            assert os.path.isabs(result)


class TestSplitPath:
    """Tests for SplitPath node."""

    @pytest.mark.asyncio
    async def test_split_path(self, context: ProcessingContext):
        with patch("nodetool.config.environment.Environment.is_production", return_value=False):
            node = SplitPath(path="/path/to/file.txt")
            result = await node.process(context)
            assert result["dirname"] == "/path/to"
            assert result["basename"] == "file.txt"


class TestSplitExtension:
    """Tests for SplitExtension node."""

    @pytest.mark.asyncio
    async def test_split_extension(self, context: ProcessingContext):
        with patch("nodetool.config.environment.Environment.is_production", return_value=False):
            node = SplitExtension(path="/path/to/file.txt")
            result = await node.process(context)
            assert result["root"] == "/path/to/file"
            assert result["extension"] == ".txt"


class TestRelativePath:
    """Tests for RelativePath node."""

    @pytest.mark.asyncio
    async def test_relative_path(self, context: ProcessingContext):
        with patch("nodetool.config.environment.Environment.is_production", return_value=False):
            node = RelativePath(target_path="/a/b/c/file.txt", start_path="/a/b")
            result = await node.process(context)
            assert result == "c/file.txt"


class TestPathToString:
    """Tests for PathToString node."""

    @pytest.mark.asyncio
    async def test_path_to_string(self, context: ProcessingContext):
        with patch("nodetool.config.environment.Environment.is_production", return_value=False):
            node = PathToString(file_path="/path/to/file.txt")
            result = await node.process(context)
            assert result == "/path/to/file.txt"

    @pytest.mark.asyncio
    async def test_empty_path_raises(self, context: ProcessingContext):
        with patch("nodetool.config.environment.Environment.is_production", return_value=False):
            node = PathToString(file_path="")
            with pytest.raises(ValueError, match="cannot be empty"):
                await node.process(context)


class TestCreateDirectory:
    """Tests for CreateDirectory node."""

    @pytest.mark.asyncio
    async def test_create_directory(self, context: ProcessingContext, temp_dir):
        with patch("nodetool.config.environment.Environment.is_production", return_value=False):
            new_dir = os.path.join(temp_dir, "new_subdir")
            node = CreateDirectory(path=new_dir)
            await node.process(context)
            assert os.path.isdir(new_dir)

    @pytest.mark.asyncio
    async def test_exist_ok(self, context: ProcessingContext, temp_dir):
        with patch("nodetool.config.environment.Environment.is_production", return_value=False):
            # Create twice should not fail with exist_ok=True
            node = CreateDirectory(path=temp_dir, exist_ok=True)
            await node.process(context)  # Should not raise


class TestCopyFile:
    """Tests for CopyFile node."""

    @pytest.mark.asyncio
    async def test_copy_file(self, context: ProcessingContext, temp_file, temp_dir):
        with patch("nodetool.config.environment.Environment.is_production", return_value=False):
            dest = os.path.join(temp_dir, "copied.txt")
            node = CopyFile(source_path=temp_file, destination_path=dest)
            await node.process(context)
            assert os.path.exists(dest)
            with open(dest, "r") as f:
                assert f.read() == "test content here"


class TestMoveFile:
    """Tests for MoveFile node."""

    @pytest.mark.asyncio
    async def test_move_file(self, context: ProcessingContext, temp_file, temp_dir):
        with patch("nodetool.config.environment.Environment.is_production", return_value=False):
            dest = os.path.join(temp_dir, "moved.txt")
            node = MoveFile(source_path=temp_file, destination_path=dest)
            await node.process(context)
            assert os.path.exists(dest)
            assert not os.path.exists(temp_file)


class TestListFiles:
    """Tests for ListFiles node."""

    @pytest.mark.asyncio
    async def test_list_files(self, context: ProcessingContext, temp_dir):
        with patch("nodetool.config.environment.Environment.is_production", return_value=False):
            # Create some test files
            for name in ["a.txt", "b.txt", "c.pdf"]:
                with open(os.path.join(temp_dir, name), "w") as f:
                    f.write("test")

            node = ListFiles(folder=temp_dir, pattern="*.txt")
            results = []
            async for item in node.gen_process(context):
                results.append(item)

            assert len(results) == 2
            filenames = [os.path.basename(r["file"]) for r in results]
            assert "a.txt" in filenames
            assert "b.txt" in filenames
