import pytest
from nodetool.config.environment import Environment
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.nodes.lib.os import (
    SetEnvironmentVariable,
    GetEnvironmentVariable,
    FileExists,
    ListFiles,
    CreateDirectory,
)

from nodetool.nodes.lib.tar import (
    CreateTar,
    ExtractTar,
    ListTar,
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
    create_dir = CreateDirectory(path=str(test_dir))
    await create_dir.process(context)
    file_path = test_dir / "example.txt"
    file_path.write_text("hello")

    exists_node = FileExists(path=str(file_path))
    assert await exists_node.process(context) is True

    list_node = ListFiles(folder=str(test_dir), pattern="*.txt")
    files = []
    async for item in list_node.gen_process(context):
        files.append(item["file"])
    assert len(files) == 1
    assert files[0] == str(file_path)


@pytest.mark.asyncio
async def test_tarfile_nodes(context: ProcessingContext, tmp_path):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "a.txt").write_text("a")
    (src_dir / "b.txt").write_text("b")

    tar_path = tmp_path / "archive.tar"
    create_tar = CreateTar(
        source_folder=str(src_dir),
        tar_path=str(tar_path),
    )
    await create_tar.process(context)
    assert tar_path.exists()

    list_tar = ListTar(tar_path=str(tar_path))
    contents = await list_tar.process(context)
    assert f"{src_dir.name}/a.txt" in contents
    assert f"{src_dir.name}/b.txt" in contents

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    extract_tar = ExtractTar(
        tar_path=str(tar_path),
        output_folder=str(out_dir),
    )
    await extract_tar.process(context)
    assert (out_dir / src_dir.name / "a.txt").exists()
    assert (out_dir / src_dir.name / "b.txt").exists()
