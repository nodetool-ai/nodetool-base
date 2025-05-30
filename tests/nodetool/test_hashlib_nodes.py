import hashlib
import pytest
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.metadata.types import FilePath
from nodetool.nodes.lib.hashlib import HashString, HashFile


@pytest.fixture
def context():
    return ProcessingContext(user_id="test", auth_token="test")


@pytest.mark.asyncio
async def test_hash_string_md5(context: ProcessingContext):
    node = HashString(text="hello", algorithm="md5")
    result = await node.process(context)
    assert result == hashlib.md5(b"hello").hexdigest()


@pytest.mark.asyncio
async def test_hash_file_sha256(context: ProcessingContext, tmp_path):
    file = tmp_path / "data.txt"
    file.write_text("world")
    node = HashFile(file=FilePath(path=str(file)), algorithm="sha256")
    result = await node.process(context)
    assert result == hashlib.sha256(b"world").hexdigest()
