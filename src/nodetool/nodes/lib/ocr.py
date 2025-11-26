from enum import Enum
from typing import TYPE_CHECKING, Any, Optional, TypedDict

from pydantic import Field

from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.types import NodeUpdate
from nodetool.metadata.types import ImageRef, OCRResult

if TYPE_CHECKING:
    from paddleocr import PaddleOCR
else:
    PaddleOCR = Any


class OCRLanguage(str, Enum):
    # Latin script languages
    ENGLISH = "en"
    FRENCH = "fr"
    GERMAN = "de"
    SPANISH = "es"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    DUTCH = "nl"
    POLISH = "pl"
    ROMANIAN = "ro"
    CROATIAN = "hr"
    CZECH = "cs"
    HUNGARIAN = "hu"
    SLOVAK = "sk"
    SLOVENIAN = "sl"
    TURKISH = "tr"
    VIETNAMESE = "vi"
    INDONESIAN = "id"
    MALAY = "ms"
    LATIN = "la"

    # Cyrillic script languages
    RUSSIAN = "ru"
    BULGARIAN = "bg"
    UKRAINIAN = "uk"
    BELARUSIAN = "be"
    MONGOLIAN = "mn"

    # CJK languages
    CHINESE = "ch"
    JAPANESE = "ja"
    KOREAN = "ko"

    # Arabic script languages
    ARABIC = "ar"
    PERSIAN = "fa"
    URDU = "ur"

    # Indic scripts
    HINDI = "hi"
    MARATHI = "mr"
    NEPALI = "ne"
    SANSKRIT = "sa"


class PaddleOCRNode(BaseNode):
    """
    Performs Optical Character Recognition (OCR) on images using PaddleOCR.
    image, text, ocr, document

    Use cases:
    - Text extraction from images
    - Document digitization
    - Receipt/invoice processing
    - Handwriting recognition
    """

    image: ImageRef = Field(
        default=ImageRef(),
        title="Input Image",
        description="The image to perform OCR on",
    )
    language: OCRLanguage = Field(
        default=OCRLanguage.ENGLISH, description="Language code for OCR"
    )

    _ocr: Optional["PaddleOCR"] = None

    def required_inputs(self):
        return ["image"]

    async def initialize(self, context: ProcessingContext):
        from paddleocr import PaddleOCR

        context.post_message(
            NodeUpdate(
                node_id=self.id,
                node_name="PaddleOCR",
                node_type=self.get_node_type(),
                status="downloading model",
            )
        )
        self._ocr = PaddleOCR(lang=self.language)

    class OutputType(TypedDict):
        boxes: list[OCRResult]
        text: str

    async def process(self, context: ProcessingContext) -> OutputType:
        assert self._ocr is not None
        image = await context.image_to_numpy(self.image)

        result = self._ocr.ocr(image)

        processed_results = []
        for res in result or []:
            if not res:
                continue
            for line in res:
                if not line:
                    continue
                box_data = line[0] if len(line) > 0 else []
                text_info = line[1] if len(line) > 1 else ""
                extra_score = line[2] if len(line) > 2 else None

                points = list(box_data) if isinstance(box_data, (list, tuple)) else []
                # Pad or trim to exactly four points
                while len(points) < 4:
                    points.append((0.0, 0.0))
                top_left, top_right, bottom_right, bottom_left = points[:4]

                text = ""
                score = 0.0
                if isinstance(text_info, (list, tuple)):
                    if text_info:
                        text = text_info[0]
                    if len(text_info) > 1:
                        try:
                            score = float(text_info[1])
                        except (TypeError, ValueError):
                            score = 0.0
                elif isinstance(text_info, str):
                    text = text_info

                if not text and extra_score and isinstance(extra_score, str):
                    text = extra_score
                    extra_score = None

                if score == 0.0 and isinstance(extra_score, (int, float)):
                    score = float(extra_score)

                processed_results.append(
                    OCRResult(
                        text=text,
                        score=score,
                        top_left=top_left,
                        top_right=top_right,
                        bottom_right=bottom_right,
                        bottom_left=bottom_left,
                    )
                )

        return {
            "boxes": processed_results,
            "text": "\n".join([result.text for result in processed_results]),
        }
