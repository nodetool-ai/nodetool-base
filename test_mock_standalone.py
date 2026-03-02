import sys
import os

# mock dependencies
sys.modules['nodetool.config'] = __import__('unittest.mock').mock.Mock()
sys.modules['nodetool.config.logging_config'] = __import__('unittest.mock').mock.Mock()
sys.modules['nodetool.config.environment'] = __import__('unittest.mock').mock.Mock()
sys.modules['nodetool.workflows'] = __import__('unittest.mock').mock.Mock()
sys.modules['nodetool.workflows.base_node'] = __import__('unittest.mock').mock.Mock()
sys.modules['nodetool.workflows.processing_context'] = __import__('unittest.mock').mock.Mock()
sys.modules['nodetool.metadata'] = __import__('unittest.mock').mock.Mock()
sys.modules['nodetool.metadata.types'] = __import__('unittest.mock').mock.Mock()
sys.modules['nodetool.providers'] = __import__('unittest.mock').mock.Mock()
sys.modules['nodetool.providers.types'] = __import__('unittest.mock').mock.Mock()

# create mock BaseNode
from pydantic import BaseModel
class BaseNode(BaseModel):
    pass

sys.modules['nodetool.workflows.base_node'].BaseNode = BaseNode

class Environment:
    @staticmethod
    def is_production():
        return False
sys.modules['nodetool.config.environment'].Environment = Environment

sys.path.insert(0, os.path.abspath('src'))
from nodetool.nodes.nodetool.dictionary import SaveCSVFile, LoadCSVFile

import pytest
import asyncio
import tempfile

@pytest.mark.asyncio
async def test_csv_perf_optimized():
    data = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]

    save_node = SaveCSVFile(data=data, folder="", filename="test_%Y.csv")

    with tempfile.TemporaryDirectory() as d:
        save_node.folder = d
        await save_node.process(None)

        load_node = LoadCSVFile(path=os.path.join(d, "test_" + __import__('datetime').datetime.now().strftime("%Y") + ".csv"))
        res = await load_node.process(None)

        assert len(res) == 2
        assert res[0]["a"] == "1"
        assert res[0]["b"] == "2"
        assert res[1]["a"] == "3"
        assert res[1]["b"] == "4"

if __name__ == "__main__":
    asyncio.run(test_csv_perf_optimized())
    print("Test passed successfully.")
