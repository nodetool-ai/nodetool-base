import io
import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from nodetool.workflows.processing_context import ProcessingContext
from nodetool.nodes.lib.json import LoadJSONAssets
from nodetool.metadata.types import FolderRef


@pytest.mark.asyncio
async def test_load_json_assets_yields_pairs():
    ctx = MagicMock(spec=ProcessingContext)

    # Mock folder with an asset_id
    folder = FolderRef(asset_id="parent-1")

    # Mock assets returned by list_assets
    assets = [
        SimpleNamespace(id="a1", name="one.json"),
        SimpleNamespace(id="a2", name="two.json"),
    ]
    ctx.list_assets = AsyncMock(return_value=(assets, None))

    # Mock download_asset to return file-like objects
    downloads = {
        "a1": io.StringIO('{"x": 1}'),
        "a2": io.StringIO('{"y": 2}'),
    }

    async def download(asset_id):
        return downloads[asset_id]

    ctx.download_asset = AsyncMock(side_effect=download)

    node = LoadJSONAssets(folder=folder)

    seen = []
    async for item in node.gen_process(ctx):
        seen.append(item)

    assert seen == [
        {"name": "one.json", "json": {"x": 1}},
        {"name": "two.json", "json": {"y": 2}},
    ]
