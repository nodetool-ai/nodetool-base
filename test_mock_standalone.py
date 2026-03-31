import sys
from unittest.mock import MagicMock
from pydantic import BaseModel

# First mock the exact modules that could cause issues
sys.modules['nodetool.config'] = MagicMock()
sys.modules['nodetool.config.logging_config'] = MagicMock()
sys.modules['nodetool.metadata'] = MagicMock()
sys.modules['nodetool.metadata.types'] = MagicMock()
sys.modules['nodetool.workflows'] = MagicMock()
sys.modules['nodetool.workflows.types'] = MagicMock()
sys.modules['nodetool.workflows.processing_context'] = MagicMock()
sys.modules['nodetool.workflows.base_node'] = MagicMock()

class BaseNodeMock(BaseModel):
    pass
sys.modules['nodetool.workflows.base_node'].BaseNode = BaseNodeMock
sys.modules['nodetool.workflows.processing_context'].ProcessingContext = MagicMock

# Need to map the src directory structure to nodetool namespace
import src.nodetool.nodes.nodetool.list as list_module
sys.modules['nodetool.nodes.nodetool.list'] = list_module

import pytest
sys.exit(pytest.main(['-q', 'tests/nodetool/test_list.py']))
