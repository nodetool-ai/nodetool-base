import types
import PIL.Image
from nodetool.metadata.types import AssetRef, ImageRef
import pytest
from unittest.mock import patch

from nodetool.workflows.processing_context import ProcessingContext
from nodetool.nodes.huggingface_hub import (
    TextClassification,
    Summarization,
    Translation,
    ImageClassification,
    ImageSegmentation,
)


@pytest.fixture
def context():
    return ProcessingContext(user_id="test", auth_token="test")


class DummyHFClient:
    def __init__(self, *args, **kwargs):
        pass

    async def text_classification(self, text, model, function_to_apply=None, top_k=1):
        return [types.SimpleNamespace(label="POS", score=0.9)]

    async def summarization(
        self,
        text,
        model,
        clean_up_tokenization_spaces=True,
        truncation=None,
        generate_parameters=None,
    ):
        return types.SimpleNamespace(summary_text="short summary")

    async def chat_completion(self, messages, model, max_tokens, temperature, top_p):
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(message=types.SimpleNamespace(content="hello"))
            ]
        )

    async def translation(
        self,
        text,
        model,
        src_lang=None,
        tgt_lang=None,
        clean_up_tokenization_spaces=True,
        truncation=None,
        generate_parameters=None,
    ):
        return types.SimpleNamespace(translation_text="hola")

    async def image_classification(
        self, image_bytes, model, function_to_apply=None, top_k=1
    ):
        return [
            types.SimpleNamespace(label="cat", score=0.8),
            types.SimpleNamespace(label="dog", score=0.2),
        ]

    async def image_segmentation(
        self,
        image_bytes,
        model,
        mask_threshold,
        overlap_mask_area_threshold,
        subtask,
        threshold,
    ):
        return [types.SimpleNamespace(label="person", mask="ZmFrZS1iYXNlNjQ=")]

    async def text_to_image(
        self,
        prompt,
        model,
        guidance_scale,
        num_inference_steps,
        width,
        height,
        negative_prompt=None,
        seed=None,
        scheduler=None,
    ):
        return object()

    async def image_to_image(
        self,
        image_bytes,
        model,
        prompt,
        guidance_scale,
        negative_prompt,
        num_inference_steps,
        target_size,
    ):
        return object()

    async def text_to_speech(
        self,
        text,
        model,
        do_sample=None,
        temperature=None,
        top_k=None,
        top_p=None,
        max_new_tokens=None,
        num_beams=None,
        use_cache=None,
    ):
        return b"\x00\x01\x02"


@pytest.mark.asyncio
async def test_hf_text_classification(context: ProcessingContext):
    with patch(
        "nodetool.nodes.huggingface_hub.Environment.get_environment",
        return_value={"HF_TOKEN": "x"},
    ):
        with patch(
            "nodetool.nodes.huggingface_hub.AsyncInferenceClient", DummyHFClient
        ):
            node = TextClassification(text="good")
            out = await node.process(context)
            assert out == {"POS": 0.9}


@pytest.mark.asyncio
async def test_hf_summarization_translation_chat(context: ProcessingContext):
    with patch(
        "nodetool.nodes.huggingface_hub.Environment.get_environment",
        return_value={"HF_TOKEN": "x"},
    ):
        with patch(
            "nodetool.nodes.huggingface_hub.AsyncInferenceClient", DummyHFClient
        ):
            s = await Summarization(text="long text").process(context)
            assert s == "short summary"

            t = await Translation(text="hello", src_lang="en", tgt_lang="es").process(
                context
            )
            assert t == "hola"


@pytest.mark.asyncio
async def test_hf_vision_and_audio_nodes(context: ProcessingContext):
    with patch(
        "nodetool.nodes.huggingface_hub.Environment.get_environment",
        return_value={"HF_TOKEN": "x"},
    ):
        with patch(
            "nodetool.nodes.huggingface_hub.AsyncInferenceClient", DummyHFClient
        ):
            # Mock context conversions
            from nodetool.metadata.types import AudioRef, ImageRef

            async def _asset_to_bytes(asset_ref: AssetRef) -> bytes:
                return b"img"


            context.asset_to_bytes = _asset_to_bytes
            ic = await ImageClassification().process(context)
            assert ic["cat"] == 0.8

            seg = await ImageSegmentation().process(context)
            assert seg and seg[0].label == "person"