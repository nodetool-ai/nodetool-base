import pytest
import os
import base64
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.nodes.nodetool.workspace import (
    _validate_workspace_path,
    GetWorkspaceDir,
    ListWorkspaceFiles,
    ReadTextFile,
    WriteTextFile,
    ReadBinaryFile,
    WriteBinaryFile,
    DeleteWorkspaceFile,
    CreateWorkspaceDirectory,
    WorkspaceFileExists,
    GetWorkspaceFileInfo,
    CopyWorkspaceFile,
    MoveWorkspaceFile,
    GetWorkspaceFileSize,
    IsWorkspaceFile,
    IsWorkspaceDirectory,
    JoinWorkspacePaths,
)


@pytest.fixture
def context(tmp_path):
    ctx = ProcessingContext(user_id="test", auth_token="test")
    ctx.workspace_dir = str(tmp_path)
    return ctx


class TestValidateWorkspacePath:
    """Tests for _validate_workspace_path function."""

    def test_valid_relative_path(self, tmp_path):
        result = _validate_workspace_path(str(tmp_path), "subdir/file.txt")
        assert result == os.path.abspath(os.path.join(str(tmp_path), "subdir/file.txt"))

    def test_empty_path_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Path cannot be empty"):
            _validate_workspace_path(str(tmp_path), "")

    def test_absolute_path_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Absolute paths are not allowed"):
            _validate_workspace_path(str(tmp_path), "/etc/passwd")

    def test_parent_traversal_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Parent directory traversal"):
            _validate_workspace_path(str(tmp_path), "../outside")

    def test_current_dir_allowed(self, tmp_path):
        result = _validate_workspace_path(str(tmp_path), ".")
        assert result == os.path.abspath(str(tmp_path))


class TestGetWorkspaceDir:
    """Tests for GetWorkspaceDir node."""

    @pytest.mark.asyncio
    async def test_returns_workspace_dir(self, context):
        node = GetWorkspaceDir()
        result = await node.process(context)
        assert result == context.workspace_dir


class TestListWorkspaceFiles:
    """Tests for ListWorkspaceFiles node."""

    def test_is_cacheable(self):
        assert ListWorkspaceFiles.is_cacheable() is False

    @pytest.mark.asyncio
    async def test_list_files_basic(self, context, tmp_path):
        # Create test files
        (tmp_path / "file1.txt").write_text("content1")
        (tmp_path / "file2.txt").write_text("content2")
        (tmp_path / "file.json").write_text("{}")

        node = ListWorkspaceFiles(path=".", pattern="*.txt")
        files = []
        async for item in node.gen_process(context):
            files.append(item["file"])

        assert len(files) == 2
        assert "file1.txt" in files
        assert "file2.txt" in files

    @pytest.mark.asyncio
    async def test_list_files_recursive(self, context, tmp_path):
        # Create subdirectory with files
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (tmp_path / "root.txt").write_text("root")
        (subdir / "nested.txt").write_text("nested")

        node = ListWorkspaceFiles(path=".", pattern="*.txt", recursive=True)
        files = []
        async for item in node.gen_process(context):
            files.append(item["file"])

        assert len(files) == 2


class TestReadTextFile:
    """Tests for ReadTextFile node."""

    @pytest.mark.asyncio
    async def test_read_text_file(self, context, tmp_path):
        test_content = "Hello, World!"
        (tmp_path / "test.txt").write_text(test_content)

        node = ReadTextFile(path="test.txt")
        result = await node.process(context)
        assert result == test_content

    @pytest.mark.asyncio
    async def test_read_text_file_not_found(self, context):
        node = ReadTextFile(path="nonexistent.txt")
        with pytest.raises(FileNotFoundError):
            await node.process(context)

    @pytest.mark.asyncio
    async def test_read_text_file_is_directory(self, context, tmp_path):
        (tmp_path / "mydir").mkdir()
        node = ReadTextFile(path="mydir")
        with pytest.raises(ValueError, match="not a file"):
            await node.process(context)


class TestWriteTextFile:
    """Tests for WriteTextFile node."""

    @pytest.mark.asyncio
    async def test_write_text_file(self, context, tmp_path):
        node = WriteTextFile(path="output.txt", content="Test content")
        result = await node.process(context)

        assert result == "output.txt"
        assert (tmp_path / "output.txt").read_text() == "Test content"

    @pytest.mark.asyncio
    async def test_write_text_file_append(self, context, tmp_path):
        (tmp_path / "append.txt").write_text("First line\n")

        node = WriteTextFile(path="append.txt", content="Second line", append=True)
        await node.process(context)

        content = (tmp_path / "append.txt").read_text()
        assert content == "First line\nSecond line"

    @pytest.mark.asyncio
    async def test_write_creates_directories(self, context, tmp_path):
        node = WriteTextFile(path="subdir/deep/file.txt", content="nested content")
        await node.process(context)

        assert (tmp_path / "subdir" / "deep" / "file.txt").exists()


class TestReadBinaryFile:
    """Tests for ReadBinaryFile node."""

    @pytest.mark.asyncio
    async def test_read_binary_file(self, context, tmp_path):
        binary_data = b"\x00\x01\x02\x03\x04"
        (tmp_path / "binary.bin").write_bytes(binary_data)

        node = ReadBinaryFile(path="binary.bin")
        result = await node.process(context)

        decoded = base64.b64decode(result)
        assert decoded == binary_data

    @pytest.mark.asyncio
    async def test_read_binary_file_not_found(self, context):
        node = ReadBinaryFile(path="nonexistent.bin")
        with pytest.raises(FileNotFoundError):
            await node.process(context)


class TestWriteBinaryFile:
    """Tests for WriteBinaryFile node."""

    @pytest.mark.asyncio
    async def test_write_binary_file(self, context, tmp_path):
        binary_data = b"\x00\x01\x02\x03\x04"
        encoded = base64.b64encode(binary_data).decode("ascii")

        node = WriteBinaryFile(path="output.bin", content=encoded)
        result = await node.process(context)

        assert result == "output.bin"
        assert (tmp_path / "output.bin").read_bytes() == binary_data


class TestDeleteWorkspaceFile:
    """Tests for DeleteWorkspaceFile node."""

    def test_is_cacheable(self):
        assert DeleteWorkspaceFile.is_cacheable() is False

    @pytest.mark.asyncio
    async def test_delete_file(self, context, tmp_path):
        (tmp_path / "to_delete.txt").write_text("delete me")

        node = DeleteWorkspaceFile(path="to_delete.txt")
        await node.process(context)

        assert not (tmp_path / "to_delete.txt").exists()

    @pytest.mark.asyncio
    async def test_delete_nonexistent_raises(self, context):
        node = DeleteWorkspaceFile(path="nonexistent.txt")
        with pytest.raises(FileNotFoundError):
            await node.process(context)

    @pytest.mark.asyncio
    async def test_delete_directory_without_recursive_raises(self, context, tmp_path):
        (tmp_path / "mydir").mkdir()

        node = DeleteWorkspaceFile(path="mydir", recursive=False)
        with pytest.raises(ValueError, match="Set recursive=True"):
            await node.process(context)

    @pytest.mark.asyncio
    async def test_delete_directory_recursive(self, context, tmp_path):
        subdir = tmp_path / "mydir"
        subdir.mkdir()
        (subdir / "file.txt").write_text("content")

        node = DeleteWorkspaceFile(path="mydir", recursive=True)
        await node.process(context)

        assert not subdir.exists()


class TestCreateWorkspaceDirectory:
    """Tests for CreateWorkspaceDirectory node."""

    @pytest.mark.asyncio
    async def test_create_directory(self, context, tmp_path):
        node = CreateWorkspaceDirectory(path="newdir")
        result = await node.process(context)

        assert result == "newdir"
        assert (tmp_path / "newdir").is_dir()

    @pytest.mark.asyncio
    async def test_create_nested_directory(self, context, tmp_path):
        node = CreateWorkspaceDirectory(path="a/b/c")
        await node.process(context)

        assert (tmp_path / "a" / "b" / "c").is_dir()


class TestWorkspaceFileExists:
    """Tests for WorkspaceFileExists node."""

    def test_is_cacheable(self):
        assert WorkspaceFileExists.is_cacheable() is False

    @pytest.mark.asyncio
    async def test_file_exists(self, context, tmp_path):
        (tmp_path / "exists.txt").write_text("content")

        node = WorkspaceFileExists(path="exists.txt")
        result = await node.process(context)
        assert result is True

    @pytest.mark.asyncio
    async def test_file_not_exists(self, context):
        node = WorkspaceFileExists(path="nonexistent.txt")
        result = await node.process(context)
        assert result is False


class TestGetWorkspaceFileInfo:
    """Tests for GetWorkspaceFileInfo node."""

    def test_is_cacheable(self):
        assert GetWorkspaceFileInfo.is_cacheable() is False

    @pytest.mark.asyncio
    async def test_get_file_info(self, context, tmp_path):
        test_file = tmp_path / "info.txt"
        test_file.write_text("content")

        node = GetWorkspaceFileInfo(path="info.txt")
        result = await node.process(context)

        assert result["path"] == "info.txt"
        assert result["name"] == "info.txt"
        assert result["size"] == 7  # length of "content"
        assert result["is_file"] is True
        assert result["is_directory"] is False
        assert "created" in result
        assert "modified" in result
        assert "accessed" in result

    @pytest.mark.asyncio
    async def test_get_directory_info(self, context, tmp_path):
        (tmp_path / "mydir").mkdir()

        node = GetWorkspaceFileInfo(path="mydir")
        result = await node.process(context)

        assert result["is_file"] is False
        assert result["is_directory"] is True

    @pytest.mark.asyncio
    async def test_get_info_not_found(self, context):
        node = GetWorkspaceFileInfo(path="nonexistent.txt")
        with pytest.raises(FileNotFoundError):
            await node.process(context)


class TestCopyWorkspaceFile:
    """Tests for CopyWorkspaceFile node."""

    @pytest.mark.asyncio
    async def test_copy_file(self, context, tmp_path):
        (tmp_path / "source.txt").write_text("copy me")

        node = CopyWorkspaceFile(source="source.txt", destination="copy.txt")
        result = await node.process(context)

        assert result == "copy.txt"
        assert (tmp_path / "source.txt").read_text() == "copy me"
        assert (tmp_path / "copy.txt").read_text() == "copy me"

    @pytest.mark.asyncio
    async def test_copy_directory(self, context, tmp_path):
        subdir = tmp_path / "source_dir"
        subdir.mkdir()
        (subdir / "file.txt").write_text("nested")

        node = CopyWorkspaceFile(source="source_dir", destination="dest_dir")
        await node.process(context)

        assert (tmp_path / "dest_dir" / "file.txt").read_text() == "nested"

    @pytest.mark.asyncio
    async def test_copy_nonexistent_raises(self, context):
        node = CopyWorkspaceFile(source="nonexistent.txt", destination="dest.txt")
        with pytest.raises(FileNotFoundError):
            await node.process(context)


class TestMoveWorkspaceFile:
    """Tests for MoveWorkspaceFile node."""

    @pytest.mark.asyncio
    async def test_move_file(self, context, tmp_path):
        (tmp_path / "source.txt").write_text("move me")

        node = MoveWorkspaceFile(source="source.txt", destination="moved.txt")
        result = await node.process(context)

        assert result == "moved.txt"
        assert not (tmp_path / "source.txt").exists()
        assert (tmp_path / "moved.txt").read_text() == "move me"

    @pytest.mark.asyncio
    async def test_move_nonexistent_raises(self, context):
        node = MoveWorkspaceFile(source="nonexistent.txt", destination="dest.txt")
        with pytest.raises(FileNotFoundError):
            await node.process(context)


class TestGetWorkspaceFileSize:
    """Tests for GetWorkspaceFileSize node."""

    def test_is_cacheable(self):
        assert GetWorkspaceFileSize.is_cacheable() is False

    @pytest.mark.asyncio
    async def test_get_file_size(self, context, tmp_path):
        content = "hello world"
        (tmp_path / "size.txt").write_text(content)

        node = GetWorkspaceFileSize(path="size.txt")
        result = await node.process(context)
        assert result == len(content)

    @pytest.mark.asyncio
    async def test_get_size_not_found(self, context):
        node = GetWorkspaceFileSize(path="nonexistent.txt")
        with pytest.raises(FileNotFoundError):
            await node.process(context)

    @pytest.mark.asyncio
    async def test_get_size_is_directory(self, context, tmp_path):
        (tmp_path / "mydir").mkdir()
        node = GetWorkspaceFileSize(path="mydir")
        with pytest.raises(ValueError, match="not a file"):
            await node.process(context)


class TestIsWorkspaceFile:
    """Tests for IsWorkspaceFile node."""

    def test_is_cacheable(self):
        assert IsWorkspaceFile.is_cacheable() is False

    @pytest.mark.asyncio
    async def test_is_file_true(self, context, tmp_path):
        (tmp_path / "file.txt").write_text("content")

        node = IsWorkspaceFile(path="file.txt")
        result = await node.process(context)
        assert result is True

    @pytest.mark.asyncio
    async def test_is_file_false_directory(self, context, tmp_path):
        (tmp_path / "mydir").mkdir()

        node = IsWorkspaceFile(path="mydir")
        result = await node.process(context)
        assert result is False


class TestIsWorkspaceDirectory:
    """Tests for IsWorkspaceDirectory node."""

    def test_is_cacheable(self):
        assert IsWorkspaceDirectory.is_cacheable() is False

    @pytest.mark.asyncio
    async def test_is_directory_true(self, context, tmp_path):
        (tmp_path / "mydir").mkdir()

        node = IsWorkspaceDirectory(path="mydir")
        result = await node.process(context)
        assert result is True

    @pytest.mark.asyncio
    async def test_is_directory_false_file(self, context, tmp_path):
        (tmp_path / "file.txt").write_text("content")

        node = IsWorkspaceDirectory(path="file.txt")
        result = await node.process(context)
        assert result is False


class TestJoinWorkspacePaths:
    """Tests for JoinWorkspacePaths node."""

    @pytest.mark.asyncio
    async def test_join_paths(self, context):
        node = JoinWorkspacePaths(paths=["dir1", "dir2", "file.txt"])
        result = await node.process(context)
        assert result == os.path.join("dir1", "dir2", "file.txt")

    @pytest.mark.asyncio
    async def test_join_empty_raises(self, context):
        node = JoinWorkspacePaths(paths=[])
        with pytest.raises(ValueError, match="paths cannot be empty"):
            await node.process(context)
