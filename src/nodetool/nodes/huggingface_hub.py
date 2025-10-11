from enum import Enum
from typing import ClassVar, TypedDict
from nodetool.config.environment import Environment
from nodetool.metadata.types import (
    AudioRef,
    ImageRef,
    ImageSegmentationResult,
    InferenceProvider,
    InferenceProviderAudioClassificationModel,
    InferenceProviderImageClassificationModel,
    InferenceProviderTextClassificationModel,
    InferenceProviderSummarizationModel,
    InferenceProviderTextToImageModel,
    InferenceProviderTranslationModel,
    InferenceProviderTextToSpeechModel,
    InferenceProviderTextGenerationModel,
    InferenceProviderImageToImageModel,
    InferenceProviderImageSegmentationModel,
)
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.base_node import BaseNode
from huggingface_hub import AsyncInferenceClient
from pydantic import Field


class HuggingFaceInferenceNode(BaseNode):
    """
    HuggingFace inference node.
    """

    @classmethod
    def is_visible(cls):
        return cls != HuggingFaceInferenceNode

    def get_client(self, provider: InferenceProvider) -> AsyncInferenceClient:
        """
        Get the HuggingFace inference client.
        """
        if provider is None:
            raise ValueError("Please select a provider")
        env = Environment.get_environment()
        api_key = env.get("HF_TOKEN")
        if not api_key:
            raise ValueError("HF_TOKEN is not set")
        return AsyncInferenceClient(api_key=api_key, provider=provider.value)



class AudioClassification(HuggingFaceInferenceNode):
    """
    Audio classification node using HuggingFace Inference API. Assigns a label or class to audio data.
    audio, classification, huggingface, inference
    """

    class OutputTransform(Enum):
        sigmoid = "sigmoid"
        softmax = "softmax"
        none = "none"

    model: InferenceProviderAudioClassificationModel = Field(
        default=InferenceProviderAudioClassificationModel(
            provider=InferenceProvider.fal_ai, model_id="superb/hubert-base-superb-er"
        ),
        description="The model to use for audio classification",
    )
    audio: AudioRef = Field(default=AudioRef(), description="The audio to classify")
    function_to_apply: OutputTransform = Field(
        default=OutputTransform.none,
        description="The function to apply to the model outputs",
    )
    top_k: int = Field(default=1, description="The number of top predictions to return")

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> dict[str, float]:
        client = self.get_client(self.model.provider)
        audio_bytes = await context.asset_to_bytes(self.audio)

        output = await client.audio_classification(
            audio_bytes,
            model=self.model.model_id,
            function_to_apply=(
                self.function_to_apply.value
                if self.function_to_apply != self.OutputTransform.none
                else None
            ),
            top_k=self.top_k,
        )
        result = {}
        for prediction in output:
            result[prediction.label] = prediction.score

        return result


class ImageClassification(HuggingFaceInferenceNode):
    """
    Image classification node using HuggingFace Inference API. Assigns a label or class to image data.
    image, classification, huggingface, inference
    """

    class OutputTransform(Enum):
        sigmoid = "sigmoid"
        softmax = "softmax"
        none = "none"

    model: InferenceProviderImageClassificationModel = Field(
        default=InferenceProviderImageClassificationModel(
            provider=InferenceProvider.hf_inference,
            model_id="google/vit-base-patch16-224",
        ),
        description="The model to use for image classification",
    )
    image: ImageRef = Field(default=ImageRef(), description="The image to classify")
    function_to_apply: OutputTransform = Field(
        default=OutputTransform.none,
        description="The function to apply to the model outputs",
    )
    top_k: int = Field(default=1, description="The number of top predictions to return")

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> dict[str, float]:
        client = self.get_client(self.model.provider)
        image_bytes = await context.asset_to_bytes(self.image)

        output = await client.image_classification(
            image_bytes,
            model=self.model.model_id,
            function_to_apply=(
                self.function_to_apply.value
                if self.function_to_apply != self.OutputTransform.none
                else None
            ),
            top_k=self.top_k,
        )

        result = {}
        for prediction in output:
            result[prediction.label] = prediction.score

        return result


class ImageSegmentation(HuggingFaceInferenceNode):
    """
    Image segmentation node using HuggingFace Inference API. Divides an image into segments where each pixel is mapped to an object.
    image, segmentation, huggingface, inference
    """

    class Subtask(str, Enum):
        instance = "instance"
        panoptic = "panoptic"
        semantic = "semantic"

    model: InferenceProviderImageSegmentationModel = Field(
        default=InferenceProviderImageSegmentationModel(
            provider=InferenceProvider.hf_inference,
            model_id="openmmlab/upernet-convnext-small",
        ),
        description="The model to use for image segmentation",
    )
    image: ImageRef = Field(default=ImageRef(), description="The image to segment")
    mask_threshold: float = Field(
        default=0.5,
        description="Threshold to use when turning the predicted masks into binary values",
    )
    overlap_mask_area_threshold: float = Field(
        default=0.5,
        description="Mask overlap threshold to eliminate small, disconnected segments",
    )
    subtask: Subtask = Field(
        default=Subtask.semantic, description="The segmentation subtask to perform"
    )
    threshold: float = Field(
        default=0.5, description="Probability threshold to filter out predicted masks"
    )

    _expose_as_tool: ClassVar[bool] = True

    async def process(
        self, context: ProcessingContext
    ) -> list[ImageSegmentationResult]:
        client = self.get_client(self.model.provider)
        image_bytes = await context.asset_to_bytes(self.image)

        output = await client.image_segmentation(
            image_bytes,
            model=self.model.model_id,
            mask_threshold=self.mask_threshold,
            overlap_mask_area_threshold=self.overlap_mask_area_threshold,
            subtask=self.subtask.value,
            threshold=self.threshold,
        )

        result = []
        for segment in output:
            image = await context.image_from_base64(segment.mask)
            result.append(
                ImageSegmentationResult(
                    label=segment.label,
                    mask=image,
                )
            )

        return result


class Translation(HuggingFaceInferenceNode):
    """
    Translation node using HuggingFace Inference API. Converts text from one language to another.
    translation, huggingface, inference
    """

    class Truncation(str, Enum):
        do_not_truncate = "do_not_truncate"
        longest_first = "longest_first"
        only_first = "only_first"
        only_second = "only_second"

    model: InferenceProviderTranslationModel = Field(
        default=InferenceProviderTranslationModel(
            provider=InferenceProvider.hf_inference, model_id="google-t5/t5-base"
        ),
        description="The model to use for translation",
    )
    text: str = Field(default="", description="The text to translate")
    src_lang: str = Field(
        default="",
        description="The source language of the text. Required for models that can translate from multiple languages",
    )
    tgt_lang: str = Field(
        default="",
        description="Target language to translate to. Required for models that can translate to multiple languages",
    )
    clean_up_tokenization_spaces: bool = Field(
        default=True,
        description="Whether to clean up the potential extra spaces in the text output",
    )
    truncation: Truncation = Field(
        default=Truncation.do_not_truncate,
        description="Truncation strategy for the input text",
    )
    max_length: int = Field(
        default=512,
        description="Maximum length of the generated translation",
        ge=1,
    )
    num_beams: int = Field(
        default=1,
        description="Number of beams for beam search. 1 means no beam search",
        ge=1,
    )
    temperature: float = Field(
        default=1.0,
        description="The value used to modulate the logits distribution",
        ge=0.1,
        le=2.0,
    )

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> str:
        client = self.get_client(self.model.provider)

        # Prepare generate_parameters
        generate_parameters = {}
        if self.max_length != 512:
            generate_parameters["max_length"] = self.max_length
        if self.num_beams != 1:
            generate_parameters["num_beams"] = self.num_beams
        if self.temperature != 1.0:
            generate_parameters["temperature"] = self.temperature

        output = await client.translation(
            self.text,
            model=self.model.model_id,
            src_lang=self.src_lang if self.src_lang else None,
            tgt_lang=self.tgt_lang if self.tgt_lang else None,
            clean_up_tokenization_spaces=self.clean_up_tokenization_spaces,
            truncation=(
                self.truncation.value
                if self.truncation != self.Truncation.do_not_truncate
                else None
            ),
            generate_parameters=generate_parameters if generate_parameters else None,
        )

        return output.translation_text


class TextClassification(HuggingFaceInferenceNode):
    """
    Text classification node using HuggingFace Inference API. Assigns a label or class to given text. Use cases include sentiment analysis, natural language inference, and assessing grammatical correctness.
    text, sentiment, classification, natural language, inference, grammatical correctness, huggingface, inference
    """

    class OutputTransform(Enum):
        sigmoid = "sigmoid"
        softmax = "softmax"
        none = "none"

    model: InferenceProviderTextClassificationModel = Field(
        default=InferenceProviderTextClassificationModel(
            provider=InferenceProvider.hf_inference,
            model_id="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        ),
        description="The model to use for text classification",
    )
    text: str = Field(default="", description="The text to classify")
    function_to_apply: OutputTransform = Field(
        default=OutputTransform.none,
        description="The function to apply to the model outputs",
    )
    top_k: int = Field(
        default=1,
        description="When specified, limits the output to the top K most probable classes",
    )

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> dict[str, float]:
        client = self.get_client(self.model.provider)

        output = await client.text_classification(
            self.text,
            model=self.model.model_id,
            function_to_apply=(
                self.function_to_apply.value
                if self.function_to_apply != self.OutputTransform.none
                else None
            ),
            top_k=self.top_k,
        )

        result = {}
        for prediction in output:
            result[prediction.label] = prediction.score

        return result


class Summarization(HuggingFaceInferenceNode):
    """
    Summarization node using HuggingFace Inference API. Produces a shorter version of a document while preserving its important information. Some models can extract text from the original input, while others can generate entirely new text.
    summarization, huggingface, inference
    """

    class Truncation(str, Enum):
        do_not_truncate = "do_not_truncate"
        longest_first = "longest_first"
        only_first = "only_first"
        only_second = "only_second"

    model: InferenceProviderSummarizationModel = Field(
        default=InferenceProviderSummarizationModel(
            provider=InferenceProvider.hf_inference, model_id="facebook/bart-large-cnn"
        ),
        description="The model to use for summarization",
    )
    text: str = Field(default="", description="The input text to summarize")
    clean_up_tokenization_spaces: bool = Field(
        default=True,
        description="Whether to clean up the potential extra spaces in the text output",
    )
    truncation: Truncation = Field(
        default=Truncation.do_not_truncate,
        description="Truncation strategy for the input text",
    )
    max_length: int = Field(
        default=150,
        description="Maximum length of the generated summary",
        ge=1,
    )
    min_length: int = Field(
        default=30,
        description="Minimum length of the generated summary",
        ge=1,
    )
    num_beams: int = Field(
        default=1,
        description="Number of beams for beam search. 1 means no beam search",
        ge=1,
    )
    temperature: float = Field(
        default=1.0,
        description="The value used to modulate the logits distribution",
        ge=0.1,
        le=2.0,
    )

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> str:
        client = self.get_client(self.model.provider)

        # Prepare generate_parameters
        generate_parameters = {}
        if self.max_length != 150:
            generate_parameters["max_length"] = self.max_length
        if self.min_length != 30:
            generate_parameters["min_length"] = self.min_length
        if self.num_beams != 1:
            generate_parameters["num_beams"] = self.num_beams
        if self.temperature != 1.0:
            generate_parameters["temperature"] = self.temperature

        output = await client.summarization(
            self.text,
            model=self.model.model_id,
            clean_up_tokenization_spaces=self.clean_up_tokenization_spaces,
            truncation=(
                self.truncation.value
                if self.truncation != self.Truncation.do_not_truncate
                else None
            ),
            generate_parameters=generate_parameters if generate_parameters else None,
        )

        return output.summary_text
