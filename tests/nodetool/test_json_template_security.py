import json
import pytest
from typing import Any
from unittest.mock import MagicMock
import sys
import os
import types
from pydantic import BaseModel, ConfigDict, Field

# Ensure src is in sys.path
sys.path.insert(0, os.path.abspath("src"))

# Mock modules if they are not available
try:
    import nodetool.workflows.base_node
except ImportError:
    mock_workflows = types.ModuleType("nodetool.workflows")
    mock_base_node = types.ModuleType("nodetool.workflows.base_node")
    mock_processing_context = types.ModuleType("nodetool.workflows.processing_context")
    mock_metadata = types.ModuleType("nodetool.metadata")
    mock_metadata_types = types.ModuleType("nodetool.metadata.types")

    sys.modules["nodetool.workflows"] = mock_workflows
    sys.modules["nodetool.workflows.base_node"] = mock_base_node
    sys.modules["nodetool.workflows.processing_context"] = mock_processing_context
    sys.modules["nodetool.metadata"] = mock_metadata
    sys.modules["nodetool.metadata.types"] = mock_metadata_types

    class BaseNode(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)

    class ProcessingContext:
        def __init__(self, user_id="test", auth_token="test"):
            self.user_id = user_id
            self.auth_token = auth_token

    class FolderRef(BaseModel):
        asset_id: str = "test_id"
        def is_empty(self):
            return False

    mock_base_node.BaseNode = BaseNode
    mock_processing_context.ProcessingContext = ProcessingContext
    mock_metadata_types.FolderRef = FolderRef

from nodetool.nodes.lib.json import JSONTemplate
from nodetool.workflows.processing_context import ProcessingContext

@pytest.fixture
def context():
    return ProcessingContext()

@pytest.mark.asyncio
async def test_json_template_injection_vulnerability(context):
    """
    Test case demonstrating JSON injection vulnerability.
    """
    # Vulnerable input: injecting a new key-value pair
    injection_payload = 'John", "admin": true, "dummy": "'

    node = JSONTemplate(
        template='{"name": "$user"}',
        values={"user": injection_payload}
    )

    result = await node.process(context)

    # If the vulnerability exists, "admin" will be present in the result
    # We want to assert that this does NOT happen after the fix.
    assert "admin" not in result, "JSON Injection vulnerability detected!"
    # The value should be the full string with quotes escaped
    assert result["name"] == injection_payload

@pytest.mark.asyncio
async def test_json_template_special_characters(context):
    """
    Test case ensuring special characters like quotes are handled correctly.
    """
    # Input with quotes that would break JSON if not escaped properly
    input_value = 'John "The Rock" Doe'

    node = JSONTemplate(
        template='{"name": "$user"}',
        values={"user": input_value}
    )

    result = await node.process(context)
    assert result["name"] == input_value

@pytest.mark.asyncio
async def test_json_template_non_string_types(context):
    """
    Test case for non-string types (int, bool).
    """
    node = JSONTemplate(
        template='{"name": "$user", "active": $active, "count": $count}',
        values={
            "user": "Alice",
            "active": True,
            "count": 42
        }
    )

    result = await node.process(context)
    assert result["name"] == "Alice"
    assert result["active"] is True
    assert result["count"] == 42

@pytest.mark.asyncio
async def test_json_template_partial_interpolation(context):
    """
    Test case for partial string interpolation (e.g. "Hello $name").
    This ensures backward compatibility for users constructing strings.
    """
    node = JSONTemplate(
        template='{"greeting": "Hello $name", "farewell": "Bye $name!"}',
        values={"name": "Alice"}
    )

    result = await node.process(context)
    assert result["greeting"] == "Hello Alice"
    assert result["farewell"] == "Bye Alice!"

@pytest.mark.asyncio
async def test_json_template_partial_interpolation_with_quotes(context):
    """
    Test partial interpolation where the value itself contains quotes.
    """
    node = JSONTemplate(
        template='{"greeting": "Hello $name"}',
        values={"name": 'Alice "Wonderland"'}
    )

    result = await node.process(context)
    # The resulting JSON string should have escaped quotes: "Hello Alice \"Wonderland\""
    # When parsed back to dict, it should be: Hello Alice "Wonderland"
    assert result["greeting"] == 'Hello Alice "Wonderland"'
