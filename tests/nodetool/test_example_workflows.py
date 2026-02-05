"""
Example workflow tests using FakeProvider.

These tests verify that various workflow patterns work end-to-end
by running them with the FakeProvider, which supports all capabilities
(text generation, image generation, audio/TTS, ASR, video, embeddings)
without requiring external API calls.

The streaming flow is particularly tricky and is tested here via
gen_process() on streaming nodes (Summarizer, DataGenerator, ListGenerator).
"""

import pytest
from unittest.mock import patch, AsyncMock

from nodetool.metadata.types import (
    LanguageModel,
    ImageModel,
    VideoModel,
    ASRModel,
    TTSModel,
    Provider,
    RecordType,
    ColumnDef,
    AudioRef,
    ImageRef,
)
from nodetool.providers import FakeProvider
from nodetool.workflows.processing_context import ProcessingContext


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FAKE_LANGUAGE_MODEL = LanguageModel(
    id="fake-model-v1",
    name="Fake Model v1",
    provider=Provider.Fake,
)

FAKE_IMAGE_MODEL = ImageModel(
    id="fake-image-model",
    name="Fake Image Model",
    provider=Provider.Fake,
    supported_tasks=["text_to_image", "image_to_image"],
)

FAKE_VIDEO_MODEL = VideoModel(
    id="fake-video-model",
    name="Fake Video Model",
    provider=Provider.Fake,
    supported_tasks=["text_to_video", "image_to_video"],
)

FAKE_ASR_MODEL = ASRModel(
    id="fake-asr",
    name="Fake ASR Model",
    provider=Provider.Fake,
)

FAKE_TTS_MODEL = TTSModel(
    id="fake-tts",
    name="Fake TTS Model",
    provider=Provider.Fake,
    voices=["default", "female", "male"],
    selected_voice="default",
)


@pytest.fixture
def context():
    return ProcessingContext(user_id="test", auth_token="test")


@pytest.fixture
def fake_provider():
    """Create a basic FakeProvider for text generation."""
    return FakeProvider(text_response="Fake response text.", should_stream=False)


@pytest.fixture
def streaming_fake_provider():
    """Create a FakeProvider that streams responses."""
    return FakeProvider(
        text_response="This is a streamed fake response.",
        should_stream=True,
        chunk_size=10,
    )


# ---------------------------------------------------------------------------
# Test: Summarizer workflow (non-streaming)
# ---------------------------------------------------------------------------


class TestSummarizerWorkflow:
    @pytest.mark.asyncio
    async def test_summarizer_non_streaming(self, context: ProcessingContext):
        """Summarizer returns full text in one shot with FakeProvider."""
        from nodetool.nodes.nodetool.agents import Summarizer
        from nodetool.providers import Chunk

        fake = FakeProvider(
            text_response="A concise summary of the input.",
            should_stream=False,
        )

        node = Summarizer(
            text="A very long document that should be summarized concisely.",
            model=FAKE_LANGUAGE_MODEL,
        )

        with patch.object(
            context, "get_provider", new_callable=AsyncMock, return_value=fake
        ):
            result_text = None
            chunks = []
            async for item in node.gen_process(context):
                if item["text"] is not None:
                    result_text = item["text"]
                elif item["chunk"] is not None:
                    chunks.append(item["chunk"])

            assert result_text == "A concise summary of the input."
            assert len(chunks) == 1
            assert chunks[0].content == "A concise summary of the input."

    @pytest.mark.asyncio
    async def test_summarizer_streaming(self, context: ProcessingContext):
        """Summarizer streams chunks then yields final text with FakeProvider."""
        from nodetool.nodes.nodetool.agents import Summarizer
        from nodetool.providers import Chunk

        fake = FakeProvider(
            text_response="Summary streamed in chunks.",
            should_stream=True,
            chunk_size=7,
        )

        node = Summarizer(
            text="Another long text.",
            model=FAKE_LANGUAGE_MODEL,
        )

        with patch.object(
            context, "get_provider", new_callable=AsyncMock, return_value=fake
        ):
            result_text = None
            chunk_contents = []
            async for item in node.gen_process(context):
                if item["chunk"] is not None:
                    if isinstance(item["chunk"], Chunk):
                        chunk_contents.append(item["chunk"].content)
                elif item["text"] is not None:
                    result_text = item["text"]

            # Streaming should produce multiple chunks
            assert len(chunk_contents) >= 2, "Expected multiple streaming chunks"
            # Concatenated chunks == full text
            assert "".join(chunk_contents) == "Summary streamed in chunks."
            # Final text matches
            assert result_text == "Summary streamed in chunks."


# ---------------------------------------------------------------------------
# Test: Classifier workflow
# ---------------------------------------------------------------------------


class TestClassifierWorkflow:
    @pytest.mark.asyncio
    async def test_classifier_returns_category(self, context: ProcessingContext):
        """Classifier returns one of the given categories with FakeProvider."""
        from nodetool.nodes.nodetool.agents import Classifier

        # Return JSON so the Classifier can parse the category field
        fake = FakeProvider(
            text_response='{"category": "positive"}',
            should_stream=False,
        )

        node = Classifier(
            text="I absolutely love this product! Best purchase ever.",
            model=FAKE_LANGUAGE_MODEL,
            categories=["positive", "negative", "neutral"],
        )

        with patch.object(
            context, "get_provider", new_callable=AsyncMock, return_value=fake
        ):
            result = await node.process(context)
            assert result == "positive"


# ---------------------------------------------------------------------------
# Test: DataGenerator workflow (streaming structured data)
# ---------------------------------------------------------------------------


class TestDataGeneratorWorkflow:
    @pytest.mark.asyncio
    async def test_data_generator_streaming(self, context: ProcessingContext):
        """DataGenerator streams records then yields final dataframe."""
        from nodetool.nodes.nodetool.generators import DataGenerator

        markdown_table = """| name | age |
|------|-----|
| Alice | 30 |
| Bob | 25 |
| Charlie | 35 |"""

        fake = FakeProvider(
            text_response=markdown_table,
            should_stream=True,
            chunk_size=15,
        )

        columns = RecordType(
            columns=[
                ColumnDef(name="name", data_type="string"),
                ColumnDef(name="age", data_type="int"),
            ]
        )

        node = DataGenerator(
            model=FAKE_LANGUAGE_MODEL,
            prompt="Generate people data",
            input_text="",
            columns=columns,
            max_tokens=256,
        )

        with patch(
            "nodetool.workflows.processing_context.ProcessingContext.get_provider",
            return_value=fake,
        ):
            records = []
            result_df = None

            async for output in node.gen_process(context):
                if output["record"] is not None:
                    records.append(output["record"])
                if output["dataframe"] is not None:
                    result_df = output["dataframe"]

            assert len(records) == 3
            assert records[0] == {"name": "Alice", "age": 30}
            assert records[1] == {"name": "Bob", "age": 25}
            assert records[2] == {"name": "Charlie", "age": 35}
            assert result_df is not None
            assert result_df.data == [
                ["Alice", 30],
                ["Bob", 25],
                ["Charlie", 35],
            ]


# ---------------------------------------------------------------------------
# Test: ListGenerator workflow (streaming items)
# ---------------------------------------------------------------------------


class TestListGeneratorWorkflow:
    @pytest.mark.asyncio
    async def test_list_generator_streaming(self, context: ProcessingContext):
        """ListGenerator streams individual items with index."""
        from nodetool.nodes.nodetool.generators import ListGenerator

        text = (
            "<LIST_ITEM>First item</LIST_ITEM>\n"
            "<LIST_ITEM>Second item</LIST_ITEM>\n"
            "<LIST_ITEM>Third item</LIST_ITEM>"
        )

        fake = FakeProvider(
            text_response=text,
            should_stream=True,
            chunk_size=10,
        )

        node = ListGenerator(
            model=FAKE_LANGUAGE_MODEL,
            prompt="Generate a list of items",
            input_text="",
            max_tokens=128,
        )

        with patch(
            "nodetool.workflows.processing_context.ProcessingContext.get_provider",
            return_value=fake,
        ):
            items = []
            indices = []
            async for output in node.gen_process(context):
                if "item" in output and output["item"] is not None:
                    items.append(output["item"])
                if "index" in output and output["index"] is not None:
                    indices.append(output["index"])

            assert items == ["First item", "Second item", "Third item"]
            assert indices == [0, 1, 2]


# ---------------------------------------------------------------------------
# Test: Graph-level workflows using DSL + run_graph_async
# ---------------------------------------------------------------------------


class TestDSLGraphWorkflows:
    @pytest.mark.asyncio
    async def test_text_format_workflow(self, context: ProcessingContext):
        """Simple text formatting workflow using DSL graph nodes."""
        from nodetool.dsl.graph import create_graph, run_graph_async
        from nodetool.dsl.nodetool.constant import String
        from nodetool.dsl.nodetool.text import ToUpperCase
        from nodetool.dsl.nodetool.output import Output

        input_text = String(value="hello world")
        upper = ToUpperCase(text=input_text.output)
        output = Output(name="result", value=upper.output)

        g = create_graph(output)
        result = await run_graph_async(g)
        assert result["result"] == "HELLO WORLD"

    @pytest.mark.asyncio
    async def test_chained_text_operations(self, context: ProcessingContext):
        """Chained text operations: trim -> uppercase -> join."""
        from nodetool.dsl.graph import create_graph, run_graph_async
        from nodetool.dsl.nodetool.constant import String
        from nodetool.dsl.nodetool.text import Trim, ToUpperCase
        from nodetool.dsl.nodetool.output import Output

        input_text = String(value="  hello  ")
        trimmed = Trim(text=input_text.output)
        upper = ToUpperCase(text=trimmed.output)
        output = Output(name="result", value=upper.output)

        g = create_graph(output)
        result = await run_graph_async(g)
        assert result["result"] == "HELLO"

    @pytest.mark.asyncio
    async def test_format_text_template(self, context: ProcessingContext):
        """FormatText with Jinja template substitution."""
        from nodetool.dsl.graph import create_graph, run_graph_async
        from nodetool.dsl.nodetool.constant import String, Integer
        from nodetool.dsl.nodetool.text import FormatText
        from nodetool.dsl.nodetool.output import Output

        name = String(value="Alice")
        age = Integer(value=30)

        formatted = FormatText(
            template="Hello, {{ name }}! You are {{ age }} years old.",
            name=name.output,
            age=age.output,
        )

        output = Output(name="result", value=formatted.output)
        g = create_graph(output)
        result = await run_graph_async(g)
        assert result["result"] == "Hello, Alice! You are 30 years old."

    @pytest.mark.asyncio
    async def test_conditional_logic_workflow(self, context: ProcessingContext):
        """Conditional branching with Compare and ConditionalSwitch."""
        from nodetool.dsl.graph import create_graph, run_graph_async
        from nodetool.dsl.nodetool.constant import Integer
        from nodetool.dsl.nodetool.boolean import Compare, ConditionalSwitch
        from nodetool.dsl.nodetool.output import Output

        value = Integer(value=75)
        compare = Compare(
            a=value.output,
            b=50,
            comparison=Compare.Comparison.GREATER_THAN,
        )
        switch = ConditionalSwitch(
            condition=compare.output,
            if_true="HIGH",
            if_false="LOW",
        )
        output = Output(name="result", value=switch.output)

        g = create_graph(output)
        result = await run_graph_async(g)
        assert result["result"] == "HIGH"

    @pytest.mark.asyncio
    async def test_conditional_logic_false_branch(self, context: ProcessingContext):
        """Conditional branching takes the false path."""
        from nodetool.dsl.graph import create_graph, run_graph_async
        from nodetool.dsl.nodetool.constant import Integer
        from nodetool.dsl.nodetool.boolean import Compare, ConditionalSwitch
        from nodetool.dsl.nodetool.output import Output

        value = Integer(value=25)
        compare = Compare(
            a=value.output,
            b=50,
            comparison=Compare.Comparison.GREATER_THAN,
        )
        switch = ConditionalSwitch(
            condition=compare.output,
            if_true="HIGH",
            if_false="LOW",
        )
        output = Output(name="result", value=switch.output)

        g = create_graph(output)
        result = await run_graph_async(g)
        assert result["result"] == "LOW"


# ---------------------------------------------------------------------------
# Test: Graph workflows with FakeProvider (AI-powered nodes)
# ---------------------------------------------------------------------------


class TestDSLGraphWithFakeProvider:
    @pytest.mark.asyncio
    async def test_summarizer_graph_workflow(self):
        """Full graph workflow: constant -> Summarizer -> Output with FakeProvider."""
        from nodetool.dsl.graph import create_graph, run_graph_async
        from nodetool.dsl.nodetool.constant import String
        from nodetool.dsl.nodetool.agents import Summarizer
        from nodetool.dsl.nodetool.output import Output

        fake = FakeProvider(
            text_response="A concise summary.",
            should_stream=False,
        )

        input_text = String(value="Very long document text goes here...")
        summarizer = Summarizer(
            text=input_text.output,
            model=FAKE_LANGUAGE_MODEL,
        )
        output = Output(name="summary", value=summarizer.out.text)

        g = create_graph(output)

        with patch(
            "nodetool.workflows.processing_context.ProcessingContext.get_provider",
            new_callable=AsyncMock,
            return_value=fake,
        ):
            result = await run_graph_async(g)
            assert result["summary"] == "A concise summary."

    @pytest.mark.asyncio
    async def test_summarizer_graph_streaming(self):
        """Full graph workflow with streaming Summarizer and FakeProvider."""
        from nodetool.dsl.graph import create_graph, run_graph_async
        from nodetool.dsl.nodetool.constant import String
        from nodetool.dsl.nodetool.agents import Summarizer
        from nodetool.dsl.nodetool.output import Output

        fake = FakeProvider(
            text_response="Streamed summary output.",
            should_stream=True,
            chunk_size=8,
        )

        input_text = String(value="Long text to be summarized...")
        summarizer = Summarizer(
            text=input_text.output,
            model=FAKE_LANGUAGE_MODEL,
        )
        output = Output(name="summary", value=summarizer.out.text)

        g = create_graph(output)

        with patch(
            "nodetool.workflows.processing_context.ProcessingContext.get_provider",
            new_callable=AsyncMock,
            return_value=fake,
        ):
            result = await run_graph_async(g)
            assert result["summary"] == "Streamed summary output."

    @pytest.mark.asyncio
    async def test_classifier_graph_workflow(self):
        """Full graph workflow: text -> Classifier -> Output with FakeProvider."""
        from nodetool.dsl.graph import create_graph, run_graph_async
        from nodetool.dsl.nodetool.constant import String
        from nodetool.dsl.nodetool.agents import Classifier
        from nodetool.dsl.nodetool.output import Output

        fake = FakeProvider(
            text_response='{"category": "negative"}',
            should_stream=False,
        )

        input_text = String(value="This product is terrible and broke on day one.")
        classifier = Classifier(
            text=input_text.output,
            model=FAKE_LANGUAGE_MODEL,
            categories=["positive", "negative", "neutral"],
        )
        output = Output(name="category", value=classifier.output)

        g = create_graph(output)

        with patch(
            "nodetool.workflows.processing_context.ProcessingContext.get_provider",
            new_callable=AsyncMock,
            return_value=fake,
        ):
            result = await run_graph_async(g)
            assert result["category"] == "negative"

    @pytest.mark.asyncio
    async def test_text_to_image_graph_workflow(self):
        """TextToImage workflow produces an ImageRef using FakeProvider."""
        from nodetool.dsl.graph import create_graph, run_graph_async
        from nodetool.dsl.nodetool.constant import String
        from nodetool.dsl.nodetool.image import TextToImage
        from nodetool.dsl.nodetool.output import Output

        fake = FakeProvider(should_stream=False)

        text_prompt = String(value="A beautiful sunset over the ocean")
        image_gen = TextToImage(
            model=FAKE_IMAGE_MODEL,
            prompt=text_prompt.output,
            width=256,
            height=256,
        )
        output = Output(name="image", value=image_gen.output)

        g = create_graph(output)

        with patch(
            "nodetool.workflows.processing_context.ProcessingContext.get_provider",
            new_callable=AsyncMock,
            return_value=fake,
        ):
            result = await run_graph_async(g)
            assert "image" in result
            # TextToImage should produce an ImageRef
            image_result = result["image"]
            assert isinstance(image_result, (ImageRef, dict))

    @pytest.mark.asyncio
    async def test_multi_node_pipeline(self):
        """Pipeline: FormatText -> Summarizer -> ToUpperCase -> Output."""
        from nodetool.dsl.graph import create_graph, run_graph_async
        from nodetool.dsl.nodetool.constant import String
        from nodetool.dsl.nodetool.text import FormatText, ToUpperCase
        from nodetool.dsl.nodetool.agents import Summarizer
        from nodetool.dsl.nodetool.output import Output

        fake = FakeProvider(
            text_response="brief summary",
            should_stream=False,
        )

        name = String(value="Quantum Computing")
        formatted = FormatText(
            template="Write a detailed report about {{ topic }}.",
            topic=name.output,
        )
        summarizer = Summarizer(
            text=formatted.output,
            model=FAKE_LANGUAGE_MODEL,
        )
        upper = ToUpperCase(text=summarizer.out.text)
        output = Output(name="result", value=upper.output)

        g = create_graph(output)

        with patch(
            "nodetool.workflows.processing_context.ProcessingContext.get_provider",
            new_callable=AsyncMock,
            return_value=fake,
        ):
            result = await run_graph_async(g)
            assert result["result"] == "BRIEF SUMMARY"

    @pytest.mark.asyncio
    async def test_multiple_outputs_workflow(self):
        """Workflow with multiple outputs from different branches."""
        from nodetool.dsl.graph import create_graph, run_graph_async
        from nodetool.dsl.nodetool.constant import String
        from nodetool.dsl.nodetool.text import ToUpperCase, ToLowerCase
        from nodetool.dsl.nodetool.output import Output

        text = String(value="Hello World")
        upper = ToUpperCase(text=text.output)
        lower = ToLowerCase(text=text.output)

        output_upper = Output(name="upper_result", value=upper.output)
        output_lower = Output(name="lower_result", value=lower.output)

        g = create_graph(output_upper, output_lower)

        result = await run_graph_async(g)
        assert result["upper_result"] == "HELLO WORLD"
        assert result["lower_result"] == "hello world"


# ---------------------------------------------------------------------------
# Test: FakeProvider capabilities directly
# ---------------------------------------------------------------------------


class TestFakeProviderCapabilities:
    @pytest.mark.asyncio
    async def test_fake_provider_language_models(self):
        """FakeProvider exposes language models for discovery."""
        provider = FakeProvider()
        models = await provider.get_available_language_models()
        assert len(models) == 3
        assert all(m.provider == Provider.Fake for m in models)
        model_ids = [m.id for m in models]
        assert "fake-model-v1" in model_ids
        assert "fake-model-v2" in model_ids
        assert "fake-fast-model" in model_ids

    @pytest.mark.asyncio
    async def test_fake_provider_image_models(self):
        """FakeProvider exposes image models."""
        provider = FakeProvider()
        models = await provider.get_available_image_models()
        assert len(models) == 2
        assert all(m.provider == Provider.Fake for m in models)

    @pytest.mark.asyncio
    async def test_fake_provider_tts_models(self):
        """FakeProvider exposes TTS models."""
        provider = FakeProvider()
        models = await provider.get_available_tts_models()
        assert len(models) == 2
        assert models[0].voices == ["default", "female", "male"]

    @pytest.mark.asyncio
    async def test_fake_provider_asr_models(self):
        """FakeProvider exposes ASR models."""
        provider = FakeProvider()
        models = await provider.get_available_asr_models()
        assert len(models) == 2
        assert "fake-asr" in [m.id for m in models]

    @pytest.mark.asyncio
    async def test_fake_provider_video_models(self):
        """FakeProvider exposes video models."""
        provider = FakeProvider()
        models = await provider.get_available_video_models()
        assert len(models) == 2
        assert all("video" in t for m in models for t in m.supported_tasks)

    @pytest.mark.asyncio
    async def test_fake_provider_embedding_models(self):
        """FakeProvider exposes embedding models."""
        provider = FakeProvider()
        models = await provider.get_available_embedding_models()
        assert len(models) == 2
        assert models[0].dimensions == 1536

    @pytest.mark.asyncio
    async def test_fake_provider_generate_message(self):
        """FakeProvider returns configured text response."""
        from nodetool.metadata.types import Message, MessageTextContent

        provider = FakeProvider(
            text_response="Hello from fake!",
            should_stream=False,
        )

        msg = await provider.generate_message(
            messages=[Message(role="user", content="test")],
            model="fake-model-v1",
        )

        assert msg.role == "assistant"
        assert len(msg.content) == 1
        assert isinstance(msg.content[0], MessageTextContent)
        assert msg.content[0].text == "Hello from fake!"
        assert provider.call_count == 1

    @pytest.mark.asyncio
    async def test_fake_provider_streaming_chunks(self):
        """FakeProvider streams text in chunks of configured size."""
        from nodetool.metadata.types import Message
        from nodetool.workflows.types import Chunk

        provider = FakeProvider(
            text_response="ABCDEFGHIJ",
            should_stream=True,
            chunk_size=3,
        )

        chunks = []
        async for chunk in provider.generate_messages(
            messages=[Message(role="user", content="test")],
            model="fake-model-v1",
        ):
            assert isinstance(chunk, Chunk)
            chunks.append(chunk)

        # 10 chars / 3 per chunk = 4 chunks (3+3+3+1)
        assert len(chunks) == 4
        assert "".join(c.content for c in chunks) == "ABCDEFGHIJ"
        assert chunks[-1].done is True

    @pytest.mark.asyncio
    async def test_fake_provider_tool_calls(self):
        """FakeProvider returns tool calls when configured."""
        from nodetool.metadata.types import Message, ToolCall
        from nodetool.providers import create_fake_tool_call

        tool_calls = [create_fake_tool_call("search", {"query": "test"})]
        provider = FakeProvider(tool_calls=tool_calls)

        msg = await provider.generate_message(
            messages=[Message(role="user", content="search for test")],
            model="fake-model-v1",
        )

        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].name == "search"
        assert msg.tool_calls[0].args == {"query": "test"}

    @pytest.mark.asyncio
    async def test_fake_provider_custom_response_fn(self):
        """FakeProvider uses custom response function."""
        from nodetool.metadata.types import Message, MessageTextContent

        def smart_response(messages, model):
            if "math" in str(messages):
                return "42"
            return "I don't know"

        provider = FakeProvider(custom_response_fn=smart_response)

        msg = await provider.generate_message(
            messages=[Message(role="user", content="What is math?")],
            model="fake-model-v1",
        )
        assert msg.content[0].text == "42"

        msg2 = await provider.generate_message(
            messages=[Message(role="user", content="What is life?")],
            model="fake-model-v1",
        )
        assert msg2.content[0].text == "I don't know"

    @pytest.mark.asyncio
    async def test_fake_provider_asr(self):
        """FakeProvider ASR returns configured transcription."""
        provider = FakeProvider(asr_response="Hello, world!")
        result = await provider.automatic_speech_recognition(
            audio=b"fake-audio-bytes",
            model="fake-asr",
        )
        assert result == "Hello, world!"
        assert provider.asr_count == 1

    @pytest.mark.asyncio
    async def test_fake_provider_embeddings(self):
        """FakeProvider generates deterministic embeddings."""
        provider = FakeProvider(embedding_dimensions=128)
        embeddings = await provider.generate_embedding(
            text=["hello", "world"],
            model="fake-embedding",
        )
        assert len(embeddings) == 2
        assert len(embeddings[0]) == 128
        assert len(embeddings[1]) == 128
        # Same text should produce same embedding
        embeddings2 = await provider.generate_embedding(
            text="hello",
            model="fake-embedding",
        )
        assert embeddings2[0] == embeddings[0]

    @pytest.mark.asyncio
    async def test_fake_provider_text_to_image(self):
        """FakeProvider generates valid PNG image bytes."""
        from nodetool.providers.types import TextToImageParams

        provider = FakeProvider(image_color=(255, 0, 0))
        params = TextToImageParams(
            model=FAKE_IMAGE_MODEL,
            prompt="A red square",
            width=64,
            height=64,
        )
        image_bytes = await provider.text_to_image(params)
        assert isinstance(image_bytes, bytes)
        assert len(image_bytes) > 0
        # PNG magic bytes
        assert image_bytes[:4] == b"\x89PNG"
        assert provider.image_generation_count == 1

    @pytest.mark.asyncio
    async def test_fake_provider_tts_streaming(self):
        """FakeProvider TTS streams audio chunks as numpy arrays."""
        import numpy as np

        provider = FakeProvider(audio_duration_ms=500)
        chunks = []
        async for chunk in provider.text_to_speech(
            text="Hello, world!",
            model="fake-tts",
        ):
            assert isinstance(chunk, np.ndarray)
            assert chunk.dtype == np.int16
            chunks.append(chunk)

        assert len(chunks) >= 1
        assert provider.audio_generation_count == 1

    @pytest.mark.asyncio
    async def test_fake_provider_call_tracking(self):
        """FakeProvider tracks calls and stores last invocation details."""
        from nodetool.metadata.types import Message

        provider = FakeProvider(text_response="ok")
        msg = Message(role="user", content="test prompt")

        await provider.generate_message(
            messages=[msg],
            model="fake-model-v1",
        )

        assert provider.call_count == 1
        assert provider.last_model == "fake-model-v1"
        assert provider.last_messages is not None
        assert len(provider.last_messages) == 1

        provider.reset_all_counts()
        assert provider.call_count == 0
        assert provider.image_generation_count == 0
