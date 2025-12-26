import pytest
from unittest.mock import AsyncMock, patch
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.inbox import NodeInbox
from nodetool.workflows.io import NodeInputs, NodeOutputs
from nodetool.workflows.run_workflow import WorkflowRunner
from nodetool.nodes.anthropic.agents import ClaudeAgent
from claude_agent_sdk.types import AssistantMessage, TextBlock


@pytest.fixture
def context():
    return ProcessingContext(user_id="test", auth_token="test")


@pytest.mark.asyncio
async def test_claude_agent_requires_prompt(context: ProcessingContext):
    """Test that ClaudeAgent requires a prompt."""
    node = ClaudeAgent(prompt="")  # type: ignore[call-arg]

    inbox = NodeInbox()
    inputs = NodeInputs(inbox)
    runner = WorkflowRunner(job_id="test")
    outputs = NodeOutputs(runner=runner, node=node, context=context, capture_only=True)

    with pytest.raises(RuntimeError, match="Prompt is required"):
        await node.run(context, inputs, outputs)


@pytest.mark.asyncio
async def test_claude_agent_basic_response(context: ProcessingContext):
    """Test that ClaudeAgent can process a simple prompt and stream responses."""
    node = ClaudeAgent(
        prompt="Hello, Claude!",
        model=ClaudeAgent.Model.CLAUDE_3_5_HAIKU,
    )  # type: ignore[call-arg]

    # Mock the Claude SDK client
    mock_client = AsyncMock()

    # Create proper AssistantMessage with TextBlock
    text_block = TextBlock(text="Hello! How can I help you?")
    assistant_message = AssistantMessage(
        content=[text_block], model="claude-3-5-haiku-20241022"
    )

    async def mock_receive_response():
        yield assistant_message

    mock_client.receive_response = mock_receive_response
    mock_client.connect = AsyncMock()
    mock_client.query = AsyncMock()
    mock_client.disconnect = AsyncMock()

    with patch("nodetool.nodes.anthropic.agents.ClaudeSDKClient") as MockClient:
        MockClient.return_value = mock_client

        inbox = NodeInbox()
        inputs = NodeInputs(inbox)
        runner = WorkflowRunner(job_id="test")
        outputs = NodeOutputs(
            runner=runner, node=node, context=context, capture_only=True
        )

        await node.run(context, inputs, outputs)

        # Verify client was used correctly
        mock_client.connect.assert_awaited_once()
        mock_client.query.assert_awaited_once_with("Hello, Claude!")
        mock_client.disconnect.assert_awaited()

        # Verify outputs
        collected = outputs.collected()
        assert "text" in collected
        assert collected["text"] == "Hello! How can I help you?"
        assert "chunk" in collected


@pytest.mark.asyncio
async def test_claude_agent_with_system_prompt(context: ProcessingContext):
    """Test that ClaudeAgent accepts system prompt configuration."""
    node = ClaudeAgent(
        prompt="What is 2+2?",
        system_prompt="You are a helpful math tutor.",
        model=ClaudeAgent.Model.CLAUDE_3_5_HAIKU,
    )  # type: ignore[call-arg]

    # Mock the Claude SDK client
    mock_client = AsyncMock()

    # Create proper AssistantMessage with TextBlock
    text_block = TextBlock(text="2 + 2 equals 4")
    assistant_message = AssistantMessage(
        content=[text_block], model="claude-3-5-haiku-20241022"
    )

    async def mock_receive_response():
        yield assistant_message

    mock_client.receive_response = mock_receive_response
    mock_client.connect = AsyncMock()
    mock_client.query = AsyncMock()
    mock_client.disconnect = AsyncMock()

    with patch("nodetool.nodes.anthropic.agents.ClaudeSDKClient") as MockClient:
        MockClient.return_value = mock_client

        inbox = NodeInbox()
        inputs = NodeInputs(inbox)
        runner = WorkflowRunner(job_id="test")
        outputs = NodeOutputs(
            runner=runner, node=node, context=context, capture_only=True
        )

        await node.run(context, inputs, outputs)

        # Verify the system prompt was passed to the options
        call_args = MockClient.call_args
        options = call_args.kwargs["options"]
        assert options.system_prompt == "You are a helpful math tutor."

        # Verify outputs
        collected = outputs.collected()
        assert collected["text"] == "2 + 2 equals 4"


@pytest.mark.asyncio
async def test_claude_agent_sandbox_settings(context: ProcessingContext):
    """Test that ClaudeAgent properly configures sandbox settings."""
    node = ClaudeAgent(
        prompt="Test sandbox",
        enable_sandbox=True,
        auto_allow_bash=False,
    )  # type: ignore[call-arg]

    # Mock the Claude SDK client
    mock_client = AsyncMock()

    # Create proper AssistantMessage with TextBlock
    text_block = TextBlock(text="Sandbox test response")
    assistant_message = AssistantMessage(
        content=[text_block], model="claude-3-5-sonnet-20241022"
    )

    async def mock_receive_response():
        yield assistant_message

    mock_client.receive_response = mock_receive_response
    mock_client.connect = AsyncMock()
    mock_client.query = AsyncMock()
    mock_client.disconnect = AsyncMock()

    with patch("nodetool.nodes.anthropic.agents.ClaudeSDKClient") as MockClient:
        MockClient.return_value = mock_client

        inbox = NodeInbox()
        inputs = NodeInputs(inbox)
        runner = WorkflowRunner(job_id="test")
        outputs = NodeOutputs(
            runner=runner, node=node, context=context, capture_only=True
        )

        await node.run(context, inputs, outputs)

        # Verify sandbox settings were configured
        call_args = MockClient.call_args
        options = call_args.kwargs["options"]
        assert options.sandbox["enabled"] is True
        assert options.sandbox["autoAllowBashIfSandboxed"] is False
