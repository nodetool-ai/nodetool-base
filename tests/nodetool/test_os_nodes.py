import pytest
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.metadata.types import FilePath
from nodetool.nodes.nodetool.os import (
    SetEnvironmentVariable,
    GetEnvironmentVariable,
    FileExists,
    ListFiles,
    CreateDirectory,
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
