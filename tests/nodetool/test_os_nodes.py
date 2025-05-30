import os
import pytest
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.metadata.types import FilePath, FolderPath
from nodetool.nodes.nodetool.os import (
    SetEnvironmentVariable,
    GetEnvironmentVariable,
    FileExists,
    ListFiles,
    CreateDirectory,
    ZipFiles,
    UnzipFile,
    ListZipContents,
)


@pytest.fixture
def context(tmp_path):
    return ProcessingContext(user_id="test", auth_token="test")


@pytest.mark.asyncio
async def test_env_var_nodes(context: ProcessingContext):
    set_node = SetEnvironmentVariable(name="TEST_ENV_VAR", value="42")
    await set_node.process(context)
    get_node = GetEnvironmentVariable(name="TEST_ENV_VAR")
    result = await get_node.process(context)
    assert result == "42"


@pytest.mark.asyncio
async def test_file_operations(context: ProcessingContext, tmp_path):
    test_dir = tmp_path / "data"
    create_dir = CreateDirectory(path=FilePath(path=str(test_dir)))
    await create_dir.process(context)
    file_path = test_dir / "example.txt"
    file_path.write_text("hello")

    exists_node = FileExists(path=FilePath(path=str(file_path)))
    assert await exists_node.process(context) is True

    list_node = ListFiles(directory=FilePath(path=str(test_dir)), pattern="*.txt")
    files = await list_node.process(context)
    assert len(files) == 1
    assert files[0].path == str(file_path)


@pytest.mark.asyncio
async def test_zipfile_nodes(context: ProcessingContext, tmp_path):
    file1 = tmp_path / "a.txt"
    file1.write_text("one")
    file2 = tmp_path / "b.txt"
    file2.write_text("two")

    zip_path = tmp_path / "archive.zip"
    zip_node = ZipFiles(
        files=[FilePath(path=str(file1)), FilePath(path=str(file2))],
        zip_path=FilePath(path=str(zip_path)),
    )
    result_path = await zip_node.process(context)
    assert result_path.path == str(zip_path)
    assert zip_path.exists()

    list_node = ListZipContents(zip_path=FilePath(path=str(zip_path)))
    contents = await list_node.process(context)
    assert sorted(contents) == ["a.txt", "b.txt"]

    out_dir = tmp_path / "out"
    unzip_node = UnzipFile(
        zip_path=FilePath(path=str(zip_path)),
        output_folder=FolderPath(path=str(out_dir)),
    )
    extracted = await unzip_node.process(context)
    extracted_names = {os.path.basename(fp.path) for fp in extracted}
    assert extracted_names == {"a.txt", "b.txt"}
