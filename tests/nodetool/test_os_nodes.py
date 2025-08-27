import pytest
from nodetool.common.environment import Environment
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.metadata.types import FilePath
from nodetool.nodes.lib.os import (
    SetEnvironmentVariable,
    GetEnvironmentVariable,
    FileExists,
    ListFiles,
    CreateDirectory,
    CreateTarFile,
    ExtractTarFile,
    ListTarFile,
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
async def test_tarfile_nodes(context: ProcessingContext, tmp_path):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "a.txt").write_text("a")
    (src_dir / "b.txt").write_text("b")

    tar_path = tmp_path / "archive.tar"
    create_tar = CreateTarFile(
        source_folder=FilePath(path=str(src_dir)),
        tar_path=FilePath(path=str(tar_path)),
    )
    await create_tar.process(context)
    assert tar_path.exists()

    list_tar = ListTarFile(tar_path=FilePath(path=str(tar_path)))
    contents = await list_tar.process(context)
    assert f"{src_dir.name}/a.txt" in contents
    assert f"{src_dir.name}/b.txt" in contents

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    extract_tar = ExtractTarFile(
        tar_path=FilePath(path=str(tar_path)),
        output_folder=FilePath(path=str(out_dir)),
    )
    await extract_tar.process(context)
    assert (out_dir / src_dir.name / "a.txt").exists()
    assert (out_dir / src_dir.name / "b.txt").exists()
