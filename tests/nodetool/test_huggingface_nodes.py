import types
import PIL.Image
from nodetool.metadata.types import AssetRef, ImageRef
import pytest
from unittest.mock import patch

from nodetool.workflows.processing_context import ProcessingContext
from nodetool.nodes.huggingface_hub import (
    TextClassification,
    Summarization,
    ChatCompletion,
    Translation,
    ImageClassification,
    ImageSegmentation,
    TextToImage,
    ImageToImage,
    TextToSpeech,
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

            c = await ChatCompletion(prompt="Say hi").process(context)
            assert c == "hello"


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

            async def _image_from_base64(b64: str) -> ImageRef:
                return ImageRef(uri="mask://img")

            async def _image_from_pil(image: PIL.Image.Image) -> ImageRef:
                return ImageRef()

            async def _audio_from_bytes(
                b: bytes,
            ) -> AudioRef:
                return AudioRef()

            context.asset_to_bytes = _asset_to_bytes
            context.image_from_base64 = _image_from_base64
            context.image_from_pil = _image_from_pil
            context.audio_from_bytes = _audio_from_bytes

            ic = await ImageClassification().process(context)
            assert ic["cat"] == 0.8

            seg = await ImageSegmentation().process(context)
            assert seg and seg[0].label == "person"

            from nodetool.metadata.types import (
                InferenceProvider,
                InferenceProviderTextToImageModel,
            )

            tti = await TextToImage(
                prompt="a cat",
                model=InferenceProviderTextToImageModel(
                    provider=InferenceProvider.hf_inference,
                    model_id="black-forest-labs/FLUX.1-dev",
                ),
            ).process(context)
            assert tti is not None

            from nodetool.metadata.types import InferenceProviderImageToImageModel

            iti = await ImageToImage(
                prompt="enhance",
                model=InferenceProviderImageToImageModel(
                    provider=InferenceProvider.hf_inference,
                    model_id="black-forest-labs/FLUX.1-dev",
                ),
            ).process(context)
            assert iti is not None

            tts_result = await TextToSpeech(text="hello").process(context)
            # Returns AudioRef
            from nodetool.metadata.types import AudioRef

            assert isinstance(tts_result["output"], AudioRef)
