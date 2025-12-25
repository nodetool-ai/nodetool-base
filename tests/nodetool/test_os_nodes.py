import pytest
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.nodes.lib.os import (
    FileExists,
    ListFiles,
    CreateDirectory,
)


@pytest.fixture
def context(tmp_path):
    return ProcessingContext(user_id="test", auth_token="test")


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
