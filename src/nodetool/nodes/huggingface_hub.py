from enum import Enum
from nodetool.common.environment import Environment
from nodetool.metadata.types import (
    AudioChunk,
    AudioRef,
    ImageRef,
    ImageSegmentationResult,
    InferenceProvider,
    InferenceProviderAutomaticSpeechRecognitionModel,
    InferenceProviderAudioClassificationModel,
    InferenceProviderImageClassificationModel,
    InferenceProviderTextClassificationModel,
    InferenceProviderSummarizationModel,
    InferenceProviderTextToImageModel,
    InferenceProviderTranslationModel,
    InferenceProviderTextToTextModel,
    InferenceProviderTextToSpeechModel,
    InferenceProviderTextToAudioModel,
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

    def get_client(self, provider: InferenceProvider) -> AsyncInferenceClient:
        """
        Get the HuggingFace inference client.
        """
        if provider == InferenceProvider.none:
            raise ValueError("Please select a provider")
        env = Environment.get_environment()
        api_key = env.get("HF_TOKEN")
        if not api_key:
            raise ValueError("HF_TOKEN is not set")
        return AsyncInferenceClient(api_key=api_key, provider=provider.value)


class AutomaticSpeechRecognition(HuggingFaceInferenceNode):
    """
    Automatic speech recognition node.
    """

    model: InferenceProviderAutomaticSpeechRecognitionModel = Field(
        default=InferenceProviderAutomaticSpeechRecognitionModel(
            provider=InferenceProvider.fal_ai, model_id="openai/whisper-large-v3"
        )
    )
    audio: AudioRef = Field(default=AudioRef(), description="The audio to transcribe")

    @classmethod
    def return_type(cls):
        return {
            "text": str,
            "chunks": list[AudioChunk],
        }

    async def process(self, context: ProcessingContext):
        client = self.get_client(self.model.provider)
        audio_bytes = await context.asset_to_bytes(self.audio)
        output = await client.automatic_speech_recognition(
            audio_bytes, model=self.model.model_id
        )
        return {
            "text": output.text,
            "chunks": (
                [
                    AudioChunk(
                        text=chunk.text,
                        timestamp=(chunk.timestamp[0], chunk.timestamp[1]),
                    )
                    for chunk in output.chunks
                ]
                if output.chunks
                else []
            ),
        }


class AudioClassification(HuggingFaceInferenceNode):
    """
    Audio classification node using HuggingFace Inference API.
    Assigns a label or class to audio data.
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
    Image classification node using HuggingFace Inference API.
    Assigns a label or class to image data.
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
    Image segmentation node using HuggingFace Inference API.
    Divides an image into segments where each pixel is mapped to an object.
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


class ImageToImage(HuggingFaceInferenceNode):
    """
    Image-to-image node using HuggingFace Inference API.
    Transforms a source image to match the characteristics of a target image or domain.
    """

    model: InferenceProviderImageToImageModel = Field(
        default=InferenceProviderImageToImageModel(
            provider=InferenceProvider.black_forest_labs,
            model_id="black-forest-labs/FLUX.1-Kontext-dev",
        ),
        description="The model to use for image-to-image transformation",
    )
    image: ImageRef = Field(
        default=ImageRef(), description="The input image to transform"
    )
    prompt: str = Field(
        default="", description="The text prompt to guide the image transformation"
    )
    guidance_scale: float = Field(
        default=7.5,
        description="A higher guidance scale value encourages the model to generate images closely linked to the text prompt at the expense of lower image quality",
    )
    negative_prompt: str = Field(
        default="",
        description="One prompt to guide what NOT to include in image generation",
    )
    num_inference_steps: int = Field(
        default=30,
        description="The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference",
    )
    target_width: int = Field(
        default=512, description="The target width in pixels of the output image"
    )
    target_height: int = Field(
        default=512, description="The target height in pixels of the output image"
    )

    @classmethod
    def return_type(cls):
        return ImageRef

    async def process(self, context: ProcessingContext) -> ImageRef:
        client = self.get_client(self.model.provider)
        image_bytes = await context.asset_to_bytes(self.image)

        output = await client.image_to_image(
            image_bytes,
            model=self.model.model_id,
            prompt=self.prompt,
            guidance_scale=self.guidance_scale,
            negative_prompt=self.negative_prompt,
            num_inference_steps=self.num_inference_steps,
            target_size={
                "width": self.target_width,
                "height": self.target_height,
            },  # type: ignore
        )

        return await context.image_from_pil(output)


class TextToImage(HuggingFaceInferenceNode):
    """
    Text-to-image node using HuggingFace Inference API.
    Generates an image based on a given text prompt.
    """

    model: InferenceProviderTextToImageModel = Field(
        default=InferenceProviderTextToImageModel(
            provider=InferenceProvider.none, model_id="black-forest-labs/FLUX.1-dev"
        ),
        description="The model to use for text-to-image generation",
    )
    prompt: str = Field(
        default="", description="The input text prompt to generate an image from"
    )
    guidance_scale: float = Field(
        default=7.5,
        description="A higher guidance scale value encourages the model to generate images closely linked to the text prompt, but values too high may cause saturation and other artifacts",
    )
    negative_prompt: str = Field(
        default="",
        description="One prompt to guide what NOT to include in image generation",
    )
    num_inference_steps: int = Field(
        default=30,
        description="The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference",
    )
    width: int = Field(
        default=512, description="The width in pixels of the output image"
    )
    height: int = Field(
        default=512, description="The height in pixels of the output image"
    )
    seed: int = Field(
        default=-1,
        description="Seed for the random number generator. Use -1 for random seed",
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        client = self.get_client(self.model.provider)

        output = await client.text_to_image(
            self.prompt,
            model=self.model.model_id,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_inference_steps,
            width=self.width,
            height=self.height,
            negative_prompt=self.negative_prompt,
            seed=self.seed,
        )

        return await context.image_from_pil(output)


class Translation(HuggingFaceInferenceNode):
    """
    Translation node using HuggingFace Inference API.
    Converts text from one language to another.
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

    @classmethod
    def return_type(cls):
        return str

    async def process(self, context: ProcessingContext) -> str:
        client = self.get_client(self.model.provider)

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
        )

        return output.translation_text


class TextClassification(HuggingFaceInferenceNode):
    """
    Text classification node using HuggingFace Inference API.
    Assigns a label or class to given text. Use cases include sentiment analysis,
    natural language inference, and assessing grammatical correctness.
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
    Summarization node using HuggingFace Inference API.
    Produces a shorter version of a document while preserving its important information.
    Some models can extract text from the original input, while others can generate entirely new text.
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

    @classmethod
    def return_type(cls):
        return str

    async def process(self, context: ProcessingContext) -> str:
        client = self.get_client(self.model.provider)

        output = await client.summarization(
            self.text,
            model=self.model.model_id,
            clean_up_tokenization_spaces=self.clean_up_tokenization_spaces,
            truncation=self.truncation.value,
        )

        return output.summary_text
