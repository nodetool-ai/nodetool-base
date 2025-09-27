import pytest
from unittest.mock import MagicMock, patch
from typing import Any, AsyncGenerator

from nodetool.workflows.processing_context import ProcessingContext
from nodetool.nodes.nodetool.agents import Summarizer, Extractor, Classifier, Agent
from nodetool.metadata.types import (
    LanguageModel,
    MessageAudioContent,
    MessageContent,
    MessageImageContent,
    Provider,
    ImageRef,
    AudioRef,
    Message,
    MessageTextContent,
    ToolName,
    TypeMetadata,
)
from typing import ClassVar
from nodetool.chat.providers import (
    Chunk,
    FakeProvider,
    create_simple_fake_provider,
    create_streaming_fake_provider,
)
from nodetool.workflows.io import NodeInputs, NodeOutputs
from nodetool.workflows.workflow_runner import WorkflowRunner
from nodetool.workflows.types import ToolCallUpdate


class JsonResponseFakeProvider(FakeProvider):
    """Fake provider that returns raw string content in Message.content."""

    def __init__(self, response_text: str):
        super().__init__(text_response=response_text, should_stream=False)

    async def generate_message(
        self,
        messages,
        model,
        tools=(),
        max_tokens: int = 8192,
        context_window: int = 4096,
        response_format: dict | None = None,
        **kwargs,
    ) -> Message:
        self.call_count += 1
        self.last_messages = messages
        self.last_model = model
        self.last_tools = tools
        self.last_kwargs = kwargs

        response = self.get_response(messages, model)

        if isinstance(response, list):
            return Message(
                role="assistant",
                content=[],
                tool_calls=response,
            )

        if isinstance(response, dict):
            content: list[MessageContent] = []
            text = response.get("text") or self.text_response
            if text is not None:
                content.append(MessageTextContent(text=text))
            if isinstance(response.get("image"), ImageRef):
                content.append(MessageImageContent(image=response["image"]))
            if isinstance(response.get("audio"), AudioRef):
                content.append(MessageAudioContent(audio=response["audio"]))
            return Message(role="assistant", content=content)

        return Message(role="assistant", content=response)


@pytest.fixture
def context():
    return ProcessingContext(user_id="test", auth_token="test")


@pytest.fixture
def mock_model():
    return LanguageModel(provider=Provider.OpenAI, id="gpt-4")


class TestSummarizer:
    @pytest.mark.asyncio
    async def test_summarizer_basic(
        self, context: ProcessingContext, mock_model: LanguageModel
    ):
        """Test basic summarization functionality"""
        node = Summarizer(
            text="This is a very long text that needs to be summarized into something shorter and more concise.",
            max_tokens=100,
            model=mock_model,
        )

        # Mock the provider's generate_messages method
        from nodetool.chat.providers import FakeProvider

        fake_provider = FakeProvider(
            text_response="This is a summary", should_stream=False
        )

        with patch(
            "nodetool.nodes.nodetool.agents.get_provider", return_value=fake_provider
        ):
            result_text = None
            result_chunks = []

            async for item in node.gen_process(context):
                if item["text"] is not None:
                    result_text = item["text"]
                elif item["chunk"] is not None:
                    result_chunks.append(item["chunk"])

            assert result_text == "This is a summary"
            assert len(result_chunks) == 1
            assert result_chunks[0].content == "This is a summary"

    @pytest.mark.asyncio
    async def test_summarizer_no_model_error(self, context: ProcessingContext):
        """Test error when no model is provided"""
        node = Summarizer(
            text="Some text",
            model=LanguageModel(provider=Provider.Empty),
        )

        with pytest.raises(ValueError, match="Select a model"):
            async for _ in node.gen_process(context):
                pass

    @pytest.mark.asyncio
    async def test_summarizer_streaming(
        self, context: ProcessingContext, mock_model: LanguageModel
    ):
        """Test streaming chunks before final text"""
        node = Summarizer(
            text="Long text to summarize",
            model=mock_model,
        )

        # Mock streaming chunks
        from nodetool.chat.providers import FakeProvider

        fake_provider = FakeProvider(
            text_response="This is a summary", should_stream=True, chunk_size=4
        )

        with patch(
            "nodetool.nodes.nodetool.agents.get_provider", return_value=fake_provider
        ):
            chunks = []
            final_text = None

            async for item in node.gen_process(context):
                if item["chunk"] is not None:
                    if isinstance(item["chunk"], Chunk):
                        chunks.append(item["chunk"].content)
                elif item["text"] is not None:
                    final_text = item["text"]

            assert len(chunks) >= 1, "Should have multiple chunks"
            assert "".join(chunks) == "This is a summary", "Final text from last chunk"
            assert final_text == "This is a summary", "Final text from last chunk"


class TestExtractor:
    @pytest.mark.asyncio
    async def test_extractor_basic(
        self, context: ProcessingContext, mock_model: LanguageModel
    ):
        """Test basic data extraction"""
        node = Extractor(
            text="John Doe is 30 years old and lives in New York",
            model=mock_model,
        )

        fake_provider = JsonResponseFakeProvider(
            '{"name": "John Doe", "age": "30", "city": "New York"}'
        )

        # Add mock dynamic outputs
        node._dynamic_outputs = {
            "name": TypeMetadata(type="str"),
            "age": TypeMetadata(type="int"),
            "city": TypeMetadata(type="str"),
        }

        # Mock outputs_for_instance to return test outputs
        mock_slot = MagicMock()
        mock_slot.name = "name"
        mock_slot.type.get_json_schema.return_value = {"type": "string"}

        with (
            patch(
                "nodetool.nodes.nodetool.agents.get_provider",
                return_value=fake_provider,
            ),
            patch.object(Extractor, "outputs_for_instance", return_value=[mock_slot]),
        ):
            result = await node.process(context)

        assert result == {"name": "John Doe", "age": "30", "city": "New York"}
        assert fake_provider.call_count == 1

    @pytest.mark.asyncio
    async def test_extractor_no_model_error(self, context: ProcessingContext):
        """Test error when no model is provided"""
        node = Extractor(
            text="Some text",
            model=LanguageModel(provider=Provider.Empty),
        )

        with pytest.raises(ValueError, match="Select a model"):
            await node.process(context)

    @pytest.mark.asyncio
    async def test_extractor_invalid_json_response(
        self, context: ProcessingContext, mock_model: LanguageModel
    ):
        """Test handling of invalid JSON response"""
        node = Extractor(
            text="Some text",
            model=mock_model,
        )

        fake_provider = JsonResponseFakeProvider("not valid json")

        with (
            patch(
                "nodetool.nodes.nodetool.agents.get_provider",
                return_value=fake_provider,
            ),
            patch.object(Extractor, "outputs_for_instance", return_value=[]),
        ):
            with pytest.raises(Exception):  # JSON decode error
                await node.process(context)

    @pytest.mark.asyncio
    async def test_extractor_non_dict_response(
        self, context: ProcessingContext, mock_model: LanguageModel
    ):
        """Test handling when LLM returns non-dictionary"""
        node = Extractor(
            text="Some text",
            model=mock_model,
        )

        fake_provider = JsonResponseFakeProvider('"just a string"')

        with patch(
            "nodetool.nodes.nodetool.agents.get_provider",
            return_value=fake_provider,
        ):
            node._dynamic_outputs = {"field": TypeMetadata(type="str")}
            with pytest.raises(
                ValueError, match="Extractor did not return a dictionary"
            ):
                await node.process(context)


class TestClassifier:
    @pytest.mark.asyncio
    async def test_classifier_basic(
        self, context: ProcessingContext, mock_model: LanguageModel
    ):
        """Test basic text classification"""
        node = Classifier(
            text="I love this product! It's amazing!",
            categories=["positive", "negative", "neutral"],
            model=mock_model,
        )

        fake_provider = JsonResponseFakeProvider('{"category": "positive"}')

        with patch(
            "nodetool.nodes.nodetool.agents.get_provider", return_value=fake_provider
        ):
            result = await node.process(context)

        assert result == "positive"
        assert fake_provider.call_count == 1

    @pytest.mark.asyncio
    async def test_classifier_no_model_error(self, context: ProcessingContext):
        """Test error when no model is provided"""
        node = Classifier(
            text="Some text",
            categories=["cat1", "cat2"],
            model=LanguageModel(provider=Provider.Empty),
        )

        with pytest.raises(ValueError, match="Select a model"):
            await node.process(context)

    @pytest.mark.asyncio
    async def test_classifier_insufficient_categories(
        self, context: ProcessingContext, mock_model: LanguageModel
    ):
        """Test error when less than 2 categories provided"""
        node = Classifier(
            text="Some text",
            categories=["only_one"],
            model=mock_model,
        )

        with pytest.raises(ValueError, match="At least 2 categories are required"):
            await node.process(context)

    @pytest.mark.asyncio
    async def test_classifier_empty_categories(
        self, context: ProcessingContext, mock_model: LanguageModel
    ):
        """Test error when no categories provided"""
        node = Classifier(
            text="Some text",
            categories=[],
            model=mock_model,
        )

        with pytest.raises(ValueError, match="At least 2 categories are required"):
            await node.process(context)


class TestAgent:
    @pytest.mark.asyncio
    async def test_agent_basic_text_generation(
        self, context: ProcessingContext, mock_model: LanguageModel
    ):
        """Test basic agent text generation without tools"""
        node = Agent(
            prompt="Hello, how are you?",
            model=mock_model,
        )

        # Mock generate_messages
        from nodetool.chat.providers import FakeProvider

        fake_provider = FakeProvider(
            text_response="I'm doing well, thank you!", should_stream=False
        )

        with patch(
            "nodetool.nodes.nodetool.agents.get_provider", return_value=fake_provider
        ):
            runner = WorkflowRunner(job_id="test")
            runner.context = context
            outputs = NodeOutputs(runner, node, context, capture_only=True)

            async def mock_any():
                yield "prompt", "Hello, how are you?"

            inbox = MagicMock()
            inbox.iter_any = mock_any

            await node.run(context, NodeInputs(inbox), outputs)

            collected = outputs.collected()
            assert "text" in collected
            assert collected["text"] == "I'm doing well, thank you!"

    @pytest.mark.asyncio
    async def test_agent_with_image(
        self, context: ProcessingContext, mock_model: LanguageModel
    ):
        """Test agent with image input"""
        test_image = ImageRef(
            uri="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        )

        node = Agent(
            prompt="Describe this image",
            image=test_image,
            model=mock_model,
        )

        fake_provider = FakeProvider(
            text_response="This is a small test image", should_stream=False
        )

        with patch(
            "nodetool.nodes.nodetool.agents.get_provider", return_value=fake_provider
        ):
            runner = WorkflowRunner(job_id="test")
            runner.context = context
            outputs = NodeOutputs(runner, node, context, capture_only=True)

            async def mock_any():
                yield "image", test_image

            inbox = MagicMock()
            inbox.iter_any = mock_any

            await node.run(context, NodeInputs(inbox), outputs)

            collected = outputs.collected()
            assert "text" in collected
            assert collected["text"] == "This is a small test image"

    @pytest.mark.asyncio
    async def test_agent_with_audio(
        self, context: ProcessingContext, mock_model: LanguageModel
    ):
        """Test agent with audio input"""
        test_audio = AudioRef(
            uri="data:audio/wav;base64,UklGRnoAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoAAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+TyvmwhCUpAULnfr2QdBSJ+yLvd80EB"
        )

        node = Agent(
            prompt="Transcribe this audio",
            audio=test_audio,
            model=mock_model,
        )

        fake_provider = FakeProvider(
            text_response="This is transcribed text", should_stream=False
        )

        with patch(
            "nodetool.nodes.nodetool.agents.get_provider", return_value=fake_provider
        ):
            runner = WorkflowRunner(job_id="test")
            runner.context = context
            outputs = NodeOutputs(runner, node, context, capture_only=True)

            async def mock_any():
                yield "audio", test_audio

            inbox = MagicMock()
            inbox.iter_any = mock_any

            await node.run(context, NodeInputs(inbox), outputs)

            collected = outputs.collected()
            assert "text" in collected
            assert collected["text"] == "This is transcribed text"

    @pytest.mark.asyncio
    async def test_agent_no_model_error(self, context: ProcessingContext):
        """Test error when no model is provided"""
        node = Agent(
            prompt="Test prompt",
            model=LanguageModel(provider=Provider.Empty),
        )

        from nodetool.workflows.io import NodeInputs

        runner = WorkflowRunner(job_id="test")
        runner.context = context
        outputs = NodeOutputs(runner, node, context, capture_only=True)

        async def mock_any():
            yield "prompt", "Test prompt"

        inbox = MagicMock()
        inbox.iter_any = mock_any

        with pytest.raises(ValueError, match="Select a model"):
            await node.run(context, NodeInputs(inbox), outputs)

    @pytest.mark.asyncio
    async def test_agent_with_messages(
        self, context: ProcessingContext, mock_model: LanguageModel
    ):
        """Test agent with previous conversation messages"""
        messages = [
            Message(role="user", content=[MessageTextContent(text="What's 2+2?")]),
            Message(
                role="assistant", content=[MessageTextContent(text="2+2 equals 4.")]
            ),
        ]

        node = Agent(
            prompt="Now what's 3+3?",
            history=messages,
            model=mock_model,
        )

        fake_provider = FakeProvider(text_response="3+3 equals 6.", should_stream=False)

        with patch(
            "nodetool.nodes.nodetool.agents.get_provider", return_value=fake_provider
        ):
            runner = WorkflowRunner(job_id="test")
            runner.context = context
            outputs = NodeOutputs(runner, node, context, capture_only=True)

            async def mock_any():
                yield "prompt", "Now what's 3+3?"

            inbox = MagicMock()
            inbox.iter_any = mock_any

            await node.run(context, NodeInputs(inbox), outputs)

            collected = outputs.collected()
            assert "text" in collected
            assert collected["text"] == "3+3 equals 6."
            # Verify the messages were passed correctly - should have system + conversation + user
            assert fake_provider.call_count == 1

    @pytest.mark.asyncio
    async def test_agent_streaming_text(
        self, context: ProcessingContext, mock_model: LanguageModel
    ):
        """Test agent streaming text output"""
        node = Agent(
            prompt="Count to three",
            model=mock_model,
        )

        fake_provider = FakeProvider(
            text_response="One, two, three", should_stream=True, chunk_size=5
        )

        with patch(
            "nodetool.nodes.nodetool.agents.get_provider", return_value=fake_provider
        ):
            runner = WorkflowRunner(job_id="test")
            runner.context = context
            outputs = NodeOutputs(runner, node, context, capture_only=True)

            async def mock_any():
                yield "prompt", "Count to three"

            inbox = MagicMock()
            inbox.iter_any = mock_any

            await node.run(context, NodeInputs(inbox), outputs)

            collected = outputs.collected()
            assert "text" in collected
            # final_text contains the complete text at the end
            assert collected["text"] == "One, two, three"

    @pytest.mark.asyncio
    async def test_agent_audio_output(
        self, context: ProcessingContext, mock_model: LanguageModel
    ):
        """Test agent generating audio output"""
        node = Agent(
            prompt="Generate some speech",
            model=mock_model,
        )

        import base64

        test_audio_data = base64.b64encode(b"fake_audio_data").decode()

        from nodetool.chat.providers import FakeProvider
        from nodetool.workflows.types import Chunk
        from nodetool.workflows.io import NodeInputs

        # Create custom response function that returns audio chunk
        async def audio_response_fn(messages, model, **kwargs):
            yield Chunk(content=test_audio_data, done=True, content_type="audio")

        class AudioFakeProvider(FakeProvider):
            async def generate_messages(
                self,
                messages,
                model,
                tools=(),
                max_tokens: int = 8192,
                context_window: int = 4096,
                response_format=None,
                **kwargs,
            ):
                async for c in audio_response_fn(messages, model, **kwargs):
                    yield c

        fake_provider = AudioFakeProvider()

        with patch(
            "nodetool.nodes.nodetool.agents.get_provider", return_value=fake_provider
        ):
            runner = WorkflowRunner(job_id="test")
            runner.context = context
            outputs = NodeOutputs(runner, node, context, capture_only=True)

            async def mock_any():
                yield "prompt", "Generate some speech"

            inbox = MagicMock()
            inbox.iter_any = mock_any

            await node.run(context, NodeInputs(inbox), outputs)

            collected = outputs.collected()
            assert "audio" in collected
            assert isinstance(collected["audio"], AudioRef)
            assert collected["audio"].data == b"fake_audio_data"

    def test_serialize_tool_result_dict(self):
        """Test serialization of dictionary tool result"""
        from nodetool.nodes.nodetool.agents import serialize_tool_result

        result = {"key": "value", "nested": {"inner": "data"}}
        serialized = serialize_tool_result(result)

        assert serialized == {"key": "value", "nested": {"inner": "data"}}

    def test_serialize_tool_result_list(self):
        """Test serialization of list tool result"""
        from nodetool.nodes.nodetool.agents import serialize_tool_result

        result = ["item1", "item2", {"key": "value"}]
        serialized = serialize_tool_result(result)

        assert serialized == ["item1", "item2", {"key": "value"}]

    def test_serialize_tool_result_bytes(self):
        """Test serialization of bytes tool result"""
        from nodetool.nodes.nodetool.agents import serialize_tool_result

        result = b"binary data"
        serialized = serialize_tool_result(result)

        assert serialized["__type__"] == "bytes"
        assert "base64" in serialized

    def test_serialize_tool_result_fallback(self):
        """Test serialization fallback for complex objects"""
        from nodetool.nodes.nodetool.agents import serialize_tool_result

        class ComplexObject:
            def __str__(self):
                return "complex_object_string"

        result = ComplexObject()
        serialized = serialize_tool_result(result)

        # The function returns the object itself unless there's an exception
        assert serialized is result

    @pytest.mark.asyncio
    async def test_agent_with_custom_response_logic(
        self, context: ProcessingContext, mock_model: LanguageModel
    ):
        """Test agent with custom response logic based on input."""
        node = Agent(
            prompt="Tell me about cats",
            model=mock_model,
        )

        # Custom response function that varies based on the prompt
        def smart_response(messages, model):
            prompt_text = str(messages[-1].content) if messages else ""
            if "cats" in prompt_text.lower():
                return "Cats are wonderful pets!"
            else:
                return "I don't know about that."

        fake_provider = FakeProvider(
            custom_response_fn=smart_response, should_stream=False
        )

        with patch(
            "nodetool.nodes.nodetool.agents.get_provider", return_value=fake_provider
        ):
            runner = WorkflowRunner(job_id="test")
            runner.context = context
            outputs = NodeOutputs(runner, node, context, capture_only=True)

            async def mock_any():
                yield "prompt", "Tell me about cats"

            inbox = MagicMock()
            inbox.iter_any = mock_any

            await node.run(context, NodeInputs(inbox), outputs)

            collected = outputs.collected()
            assert "text" in collected
            assert collected["text"] == "Cats are wonderful pets!"
            assert fake_provider.call_count == 1

    @pytest.mark.asyncio
    async def test_multiple_calls_with_fake_provider(
        self, context: ProcessingContext, mock_model: LanguageModel
    ):
        """Test that fake provider can handle multiple calls."""
        fake_provider = create_simple_fake_provider("Response")

        with patch(
            "nodetool.nodes.nodetool.agents.get_provider", return_value=fake_provider
        ):
            # Create multiple agents
            agent1 = Agent(prompt="Question 1", model=mock_model)
            agent2 = Agent(prompt="Question 2", model=mock_model)

            # Both should work with the same fake provider
            for agent in [agent1, agent2]:
                runner = WorkflowRunner(job_id="test")
                runner.context = context
                outputs = NodeOutputs(runner, agent, context, capture_only=True)

                async def mock_any():
                    yield "prompt", agent.prompt

                inbox = MagicMock()
                inbox.iter_any = mock_any

                await agent.run(context, NodeInputs(inbox), outputs)

            assert fake_provider.call_count == 2
            assert fake_provider.last_model == mock_model.id


class TestAgentFields:
    """Test field properties and validation for all agent nodes"""

    def test_summarizer_return_types(self):
        """Test Summarizer return types"""
        assert Summarizer.return_type() == Summarizer.OutputType

    def test_agent_return_types(self):
        """Test Agent return types"""
        assert Agent.return_type() == Agent.OutputType
