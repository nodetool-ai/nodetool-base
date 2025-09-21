import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import AsyncGenerator

from nodetool.workflows.processing_context import ProcessingContext
from nodetool.nodes.nodetool.agents import Summarizer, Extractor, Classifier, Agent
from nodetool.metadata.types import (
    LanguageModel,
    Provider,
    ImageRef,
    AudioRef,
    Message,
    MessageTextContent,
    ToolName,
)
from typing import ClassVar
from nodetool.chat.providers import (
    Chunk,
    FakeProvider,
    create_simple_fake_provider,
    create_streaming_fake_provider,
)
from nodetool.workflows.types import ToolCallUpdate


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

            async for output_type, output_value in node.gen_process(context):
                if output_type == "text":
                    result_text = output_value
                elif output_type == "chunk":
                    result_chunks.append(output_value)

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

            async for output_type, output_value in node.gen_process(context):
                if output_type == "chunk":
                    if isinstance(output_value, Chunk):
                        chunks.append(output_value.content)
                elif output_type == "text":
                    final_text = output_value

            assert len(chunks) >= 1, "Should have multiple chunks"
            assert "".join(chunks) == "This is a summary", "Final text from last chunk"


class TestExtractor:
    @pytest.mark.asyncio
    async def test_extractor_basic(
        self, context: ProcessingContext, mock_model: LanguageModel
    ):
        """Test basic data extraction"""
        node = Extractor(
            text="John Doe is 30 years old and lives in New York",
            extraction_prompt="Extract name, age, and city",
            model=mock_model,
        )

        # Mock the provider and its generate_message method
        mock_provider = MagicMock()
        mock_message = MagicMock()
        mock_message.content = '{"name": "John Doe", "age": "30", "city": "New York"}'
        mock_provider.generate_message = AsyncMock(return_value=mock_message)

        # Add mock dynamic outputs
        node._dynamic_outputs = {
            "name": MagicMock(),
            "age": MagicMock(),
            "city": MagicMock(),
        }

        # Mock outputs_for_instance to return test outputs
        mock_slot = MagicMock()
        mock_slot.name = "name"
        mock_slot.type.get_json_schema.return_value = {"type": "string"}

        with patch(
            "nodetool.nodes.nodetool.agents.get_provider", return_value=mock_provider
        ), patch.object(Extractor, "outputs_for_instance", return_value=[mock_slot]):
            result = await node.process(context)

        assert result == {"name": "John Doe", "age": "30", "city": "New York"}
        mock_provider.generate_message.assert_called_once()

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

        # Mock invalid JSON response
        mock_provider = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "not valid json"
        mock_provider.generate_message = AsyncMock(return_value=mock_message)

        with patch(
            "nodetool.nodes.nodetool.agents.get_provider", return_value=mock_provider
        ), patch.object(Extractor, "outputs_for_instance", return_value=[]):
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

        mock_provider = MagicMock()
        mock_message = MagicMock()
        mock_message.content = '"just a string"'  # Valid JSON but not a dict
        mock_provider.generate_message = AsyncMock(return_value=mock_message)

        with patch(
            "nodetool.nodes.nodetool.agents.get_provider", return_value=mock_provider
        ), patch.object(Extractor, "outputs_for_instance", return_value=[]):
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

            # Mock the provider to return a JSON content response
            mock_provider = MagicMock()
            mock_message = MagicMock()
            mock_message.content = '{"category": "positive"}'
            mock_provider.generate_message = AsyncMock(return_value=mock_message)

            with patch(
                "nodetool.nodes.nodetool.agents.get_provider", return_value=mock_provider
            ):
                result = await node.process(context)

        assert result == "positive"

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
        from nodetool.workflows.io import NodeInputs, NodeOutputs
        from unittest.mock import MagicMock

        fake_provider = FakeProvider(
            text_response="I'm doing well, thank you!", should_stream=False
        )

        with patch(
            "nodetool.nodes.nodetool.agents.get_provider", return_value=fake_provider
        ):
            # Mock inputs and outputs
            inbox = MagicMock()
            inputs = NodeInputs(inbox)
            
            # Make inputs.any() yield a prompt input to trigger execution
            async def mock_any():
                yield "prompt", "Hello, how are you?"
            inbox.iter_any = mock_any
            
            runner = MagicMock()
            outputs = NodeOutputs(runner, node, context, capture_only=True)
            
            await node.run(context, inputs, outputs)
            
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

        from nodetool.chat.providers import FakeProvider
        from nodetool.workflows.io import NodeInputs, NodeOutputs
        from unittest.mock import MagicMock

        fake_provider = FakeProvider(
            text_response="This is a small test image", should_stream=False
        )

        with patch(
            "nodetool.nodes.nodetool.agents.get_provider", return_value=fake_provider
        ):
            inbox = MagicMock()
            inputs = NodeInputs(inbox)
            
            async def mock_any():
                yield "image", test_image
            inbox.iter_any = mock_any

            runner = MagicMock()
            outputs = NodeOutputs(runner, node, context, capture_only=True)

            await node.run(context, inputs, outputs)

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

        from nodetool.chat.providers import FakeProvider
        from nodetool.workflows.io import NodeInputs, NodeOutputs
        from unittest.mock import MagicMock

        fake_provider = FakeProvider(
            text_response="This is transcribed text", should_stream=False
        )

        with patch(
            "nodetool.nodes.nodetool.agents.get_provider", return_value=fake_provider
        ):
            inbox = MagicMock()
            inputs = NodeInputs(inbox)
            
            async def mock_any():
                yield "audio", test_audio
            inbox.iter_any = mock_any

            runner = MagicMock()
            outputs = NodeOutputs(runner, node, context, capture_only=True)

            await node.run(context, inputs, outputs)

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

        from nodetool.workflows.io import NodeInputs, NodeOutputs
        from unittest.mock import MagicMock

        inbox = MagicMock()
        inputs = NodeInputs(inbox)
        
        async def mock_any():
            yield "prompt", "Test prompt"
        inbox.iter_any = mock_any
        
        runner = MagicMock()
        outputs = NodeOutputs(runner, node, context, capture_only=True)

        with pytest.raises(ValueError, match="Select a model"):
            await node.run(context, inputs, outputs)

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

        from nodetool.chat.providers import FakeProvider
        from nodetool.workflows.io import NodeInputs, NodeOutputs
        from unittest.mock import MagicMock

        fake_provider = FakeProvider(text_response="3+3 equals 6.", should_stream=False)

        with patch(
            "nodetool.nodes.nodetool.agents.get_provider", return_value=fake_provider
        ):
            inbox = MagicMock()
            inputs = NodeInputs(inbox)
            
            async def mock_any():
                yield "prompt", "Now what's 3+3?"
            inbox.iter_any = mock_any

            runner = MagicMock()
            outputs = NodeOutputs(runner, node, context, capture_only=True)

            await node.run(context, inputs, outputs)

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

        from nodetool.chat.providers import FakeProvider
        from nodetool.workflows.io import NodeInputs, NodeOutputs
        from unittest.mock import MagicMock

        fake_provider = FakeProvider(
            text_response="One, two, three", should_stream=True, chunk_size=5
        )

        with patch(
            "nodetool.nodes.nodetool.agents.get_provider", return_value=fake_provider
        ):
            inbox = MagicMock()
            inputs = NodeInputs(inbox)
            
            async def mock_any():
                yield "prompt", "Count to three"
            inbox.iter_any = mock_any

            runner = MagicMock()
            outputs = NodeOutputs(runner, node, context, capture_only=True)

            await node.run(context, inputs, outputs)

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
        from nodetool.workflows.io import NodeInputs, NodeOutputs
        from unittest.mock import MagicMock

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
            inbox = MagicMock()
            inputs = NodeInputs(inbox)
            
            async def mock_any():
                yield "prompt", "Generate some speech"
            inbox.iter_any = mock_any

            runner = MagicMock()
            outputs = NodeOutputs(runner, node, context, capture_only=True)

            await node.run(context, inputs, outputs)

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

        from nodetool.workflows.io import NodeInputs, NodeOutputs
        from unittest.mock import MagicMock

        fake_provider = FakeProvider(
            custom_response_fn=smart_response, should_stream=False
        )

        with patch(
            "nodetool.nodes.nodetool.agents.get_provider", return_value=fake_provider
        ):
            inbox = MagicMock()
            inputs = NodeInputs(inbox)
            
            async def mock_any():
                yield "prompt", "Tell me about cats"
            inbox.iter_any = mock_any

            runner = MagicMock()
            outputs = NodeOutputs(runner, node, context, capture_only=True)

            await node.run(context, inputs, outputs)

            collected = outputs.collected()
            assert "text" in collected
            assert collected["text"] == "Cats are wonderful pets!"
            assert fake_provider.call_count == 1

    @pytest.mark.asyncio
    async def test_multiple_calls_with_fake_provider(
        self, context: ProcessingContext, mock_model: LanguageModel
    ):
        """Test that fake provider can handle multiple calls."""
        from nodetool.workflows.io import NodeInputs, NodeOutputs
        from unittest.mock import MagicMock

        fake_provider = create_simple_fake_provider("Response")

        with patch(
            "nodetool.nodes.nodetool.agents.get_provider", return_value=fake_provider
        ):
            # Create multiple agents
            agent1 = Agent(prompt="Question 1", model=mock_model)
            agent2 = Agent(prompt="Question 2", model=mock_model)

            # Both should work with the same fake provider
            for agent in [agent1, agent2]:
                inbox = MagicMock()
                inputs = NodeInputs(inbox)
                
                async def mock_any():
                    yield "prompt", agent.prompt
                inbox.iter_any = mock_any
                
                runner = MagicMock()
                outputs = NodeOutputs(runner, agent, context, capture_only=True)
                
                await agent.run(context, inputs, outputs)

            assert fake_provider.call_count == 2
            assert fake_provider.last_model == mock_model.id


class TestAgentFields:
    """Test field properties and validation for all agent nodes"""

    def test_summarizer_basic_fields(self):
        """Test Summarizer basic fields"""
        fields = Summarizer.get_basic_fields()
        expected = ["text", "max_tokens", "model"]
        assert fields == expected

    def test_extractor_basic_fields(self):
        """Test Extractor basic fields"""
        fields = Extractor.get_basic_fields()
        expected = ["text", "extraction_prompt", "model"]
        assert fields == expected

    def test_classifier_basic_fields(self):
        """Test Classifier basic fields"""
        fields = Classifier.get_basic_fields()
        expected = ["text", "categories", "model"]
        assert fields == expected

    def test_agent_basic_fields(self):
        """Test Agent basic fields"""
        fields = Agent.get_basic_fields()
        expected = ["prompt", "model", "image"]
        assert fields == expected

    def test_summarizer_not_cacheable(self):
        """Test that Summarizer is not cacheable"""
        assert not Summarizer.is_cacheable()

    def test_agent_not_cacheable(self):
        """Test that Agent is not cacheable"""
        assert not Agent.is_cacheable()

    def test_extractor_supports_dynamic_outputs: ClassVar[bool] (self):
        """Test that Extractor supports dynamic outputs"""
        assert Extractor._supports_dynamic_outputs: ClassVar[bool] 

    def test_agent_supports_dynamic_outputs: ClassVar[bool] (self):
        """Test that Agent supports dynamic outputs"""
        assert Agent._supports_dynamic_outputs: ClassVar[bool] 

    def test_summarizer_return_types(self):
        """Test Summarizer return types"""
        return_types = Summarizer.return_type()
        expected_keys = {"text", "chunk"}
        assert set(return_types.keys()) == expected_keys

    def test_agent_return_types(self):
        """Test Agent return types"""
        return_types = Agent.return_type()
        expected_keys = {"text", "chunk", "audio"}
        assert set(return_types.keys()) == expected_keys

    def test_extractor_return_types(self):
        """Test Extractor return types (empty dict)"""
        return_types = Extractor.return_type()
        assert return_types == {}
