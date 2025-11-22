from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict


class Provider(str, Enum):
    OpenAI = "openai"
    Gemini = "gemini"
    FalAI = "falai"
    HuggingFace = "huggingface"
    HuggingFaceFalAI = "huggingface_falai"


class BaseRef(BaseModel):
    type: str = ""
    uri: str = ""
    name: str = ""
    asset_id: Optional[str] = None
    data: Any = None

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    def is_empty(self) -> bool:
        return not self.uri and self.data is None and self.asset_id is None

    def is_set(self) -> bool:
        return not self.is_empty()


class DocumentRef(BaseRef):
    type: str = "document"


class AudioRef(BaseRef):
    type: str = "audio"


class VideoRef(BaseRef):
    type: str = "video"
    duration: float | None = None
    format: str | None = None


class ImageRef(BaseRef):
    type: str = "image"


class FolderRef(BaseRef):
    type: str = "folder"


class FilePath(BaseRef):
    path: str = ""


class FolderPath(BaseRef):
    path: str = ""


class TextRef(BaseRef):
    type: str = "text"


class ModelRef(BaseRef):
    type: str = "model"


class DataframeRef(BaseRef):
    type: str = "dataframe"
    columns: Optional[Sequence[str]] = None


class NPArray(BaseRef):
    type: str = "np_array"
    value: Any = None
    dtype: str | None = None
    shape: tuple[int, ...] | None = None


class AudioChunk(BaseRef):
    start: float = 0.0
    end: float = 0.0
    text: str = ""


class LanguageModel(BaseRef):
    type: str = "language_model"
    provider: Provider = Provider.OpenAI
    id: str = "gpt-4o-mini"


class HFTextGeneration(BaseRef):
    type: str = "hf_text_generation"
    model: str = ""


class ImageModel(BaseRef):
    type: str = "image_model"
    provider: Provider = Provider.OpenAI
    id: str = ""


class VideoModel(BaseRef):
    type: str = "video_model"
    provider: Provider = Provider.OpenAI
    id: str = ""


class TTSModel(BaseRef):
    type: str = "tts_model"
    provider: Provider = Provider.OpenAI
    id: str = ""


class ASRModel(BaseRef):
    type: str = "asr_model"
    provider: Provider = Provider.OpenAI
    id: str = ""


class LlamaModel(BaseRef):
    type: str = "llama_model"
    id: str = ""


class Collection(BaseRef):
    type: str = "collection"
    name: str = ""


class ColumnDef(BaseRef):
    name: str = ""
    dtype: str = "str"


class RecordType(BaseRef):
    columns: list[ColumnDef] = []


class ChartConfig(BaseRef):
    spec: dict | None = None


class FaissIndex(BaseRef):
    dim: int = 0


class ExcelRef(BaseRef):
    type: str = "excel"


class JSONRef(BaseRef):
    type: str = "json"


class SVGRef(BaseRef):
    type: str = "svg"


class SVGElement(BaseRef):
    tag: str = ""
    attributes: dict[str, str] = {}


class OCRResult(BaseRef):
    text: str = ""
    bbox: Tuple[int, int, int, int] | None = None


class Source(BaseRef):
    uri: str = ""
    text: str = ""


class ColorRef(BaseRef):
    hex: str = "#000000"


class FontRef(BaseRef):
    family: str = "Arial"
    size: int = 12


class TextChunk(BaseRef):
    text: str = ""


class ToolName(BaseRef):
    name: str = ""


Date = date
Datetime = datetime


def to_numpy(values: Any) -> Any:
    return values
