from enum import Enum
from typing import ClassVar, TypedDict
from nodetool.config.environment import Environment
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

    @classmethod
    def is_visible(cls):
        return cls != HuggingFaceInferenceNode

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
    audio, speech, recognition, huggingface, inference
    """

    class OutputType(TypedDict):
        text: str
        chunks: list[AudioChunk]

    _expose_as_tool: ClassVar[bool] = True

    model: InferenceProviderAutomaticSpeechRecognitionModel = Field(
        default=InferenceProviderAutomaticSpeechRecognitionModel(
            provider=InferenceProvider.fal_ai, model_id="openai/whisper-large-v3"
        )
    )
    audio: AudioRef = Field(default=AudioRef(), description="The audio to transcribe")

    async def process(self, context: ProcessingContext) -> OutputType:
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


class ImageToImage(HuggingFaceInferenceNode):
    """
    Image-to-image node using HuggingFace Inference API. Transforms a source image to match the characteristics of a target image or domain.
    img2img, img-to-img, huggingface, inference
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

    _expose_as_tool: ClassVar[bool] = True

    class OutputType(TypedDict):
        output: ImageRef

    async def process(self, context: ProcessingContext) -> OutputType:
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

        return {"output": await context.image_from_pil(output)}


class TextToImage(HuggingFaceInferenceNode):
    """
    Text-to-image node using HuggingFace Inference API. Generates an image based on a given text prompt.
    text2img, text-to-img, huggingface, inference
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
    scheduler: str = Field(
        default="",
        description="Override the scheduler with a compatible one",
    )

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> ImageRef:
        client = self.get_client(self.model.provider)

        output = await client.text_to_image(
            self.prompt,
            model=self.model.model_id,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_inference_steps,
            width=self.width,
            height=self.height,
            negative_prompt=self.negative_prompt if self.negative_prompt else None,
            seed=self.seed if self.seed != -1 else None,
            scheduler=self.scheduler if self.scheduler else None,
        )

        return await context.image_from_pil(output)


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


class ChatCompletion(HuggingFaceInferenceNode):
    """
    Chat completion node using HuggingFace Inference API. Generates text based on a given prompt.
    chat, completion, huggingface, inference
    """

    model: InferenceProviderTextGenerationModel = Field(
        default=InferenceProviderTextGenerationModel(
            provider=InferenceProvider.cerebras,
            model_id="Qwen/Qwen3-235B-A22B-Thinking-2507",
        ),
        description="The model to use for text generation",
    )
    prompt: str = Field(
        default="", description="The input text prompt to generate from"
    )
    max_tokens: int = Field(
        default=4096,
        description="Maximum number of tokens to generate",
        ge=1,
        le=16384,
    )
    temperature: float = Field(
        default=1.0,
        description="The value used to module the logits distribution",
        ge=0.0,
        le=2.0,
    )
    top_p: float = Field(
        default=1.0,
        description="Top-p value for nucleus sampling",
        ge=0.0,
        le=1.0,
    )
    top_k: int = Field(
        default=50,
        description="The number of highest probability vocabulary tokens to keep for top-k-filtering",
    )

    async def process(self, context: ProcessingContext) -> str:
        client = self.get_client(self.model.provider)

        output = await client.chat_completion(
            messages=[{"role": "user", "content": self.prompt}],
            model=self.model.model_id,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        assert output.choices[0].message.content is not None

        return output.choices[0].message.content


class TextToSpeech(HuggingFaceInferenceNode):
    """
    Text-to-speech node using HuggingFace Inference API. Generates speech audio from input text.
    text-to-speech, text-to-audio, speech, huggingface, inference
    """

    model: InferenceProviderTextToSpeechModel = Field(
        default=InferenceProviderTextToSpeechModel(
            provider=InferenceProvider.hf_inference, model_id="microsoft/speecht5_tts"
        ),
        description="The model to use for text-to-speech synthesis",
    )
    text: str = Field(default="", description="The input text to convert to speech")
    do_sample: bool = Field(
        default=False,
        description="Whether to use sampling instead of greedy decoding when generating new tokens",
    )
    temperature: float = Field(
        default=1.0,
        description="The value used to modulate the next token probabilities",
        ge=0.1,
        le=2.0,
    )
    top_k: int = Field(
        default=50,
        description="The number of highest probability vocabulary tokens to keep for top-k-filtering",
        ge=1,
    )
    top_p: float = Field(
        default=1.0,
        description="If set to float < 1, only the smallest set of most probable tokens are kept for generation",
        ge=0.0,
        le=1.0,
    )
    max_new_tokens: int = Field(
        default=512,
        description="The maximum number of tokens to generate",
        ge=1,
    )
    num_beams: int = Field(
        default=1,
        description="Number of beams to use for beam search",
        ge=1,
    )
    use_cache: bool = Field(
        default=True,
        description="Whether the model should use the past last key/values attentions to speed up decoding",
    )

    class OutputType(TypedDict):
        output: AudioRef

    async def process(self, context: ProcessingContext) -> OutputType:
        client = self.get_client(self.model.provider)

        output = await client.text_to_speech(
            self.text,
            model=self.model.model_id,
            do_sample=self.do_sample if self.do_sample else None,
            temperature=self.temperature if self.temperature != 1.0 else None,
            top_k=self.top_k if self.top_k != 50 else None,
            top_p=self.top_p if self.top_p != 1.0 else None,
            max_new_tokens=self.max_new_tokens if self.max_new_tokens != 512 else None,
            num_beams=self.num_beams if self.num_beams != 1 else None,
            use_cache=self.use_cache if not self.use_cache else None,
        )

        return {"output": await context.audio_from_bytes(output)}
