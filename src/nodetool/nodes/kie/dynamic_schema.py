"""
Dynamic Kie.ai node that creates inputs/outputs from pasted API documentation.

Users paste the kie.ai markdown API docs for any model, and this module:
1. Parses the markdown to extract model ID, input parameters, and output type
2. Populates dynamic properties/inputs/outputs on the node
3. Executes via KieBaseNode's createTask/recordInfo lifecycle
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, ClassVar

from pydantic import Field

from nodetool.metadata.type_metadata import TypeMetadata
from nodetool.metadata.types import ImageRef
from nodetool.nodes.kie.image import KieBaseNode
from nodetool.workflows.processing_context import ProcessingContext

HIDDEN_PARAMS = frozenset({"upload_method", "callBackUrl", "callback_url"})


@dataclass(frozen=True)
class KieParamInfo:
    name: str
    type: str
    required: bool
    description: str
    default: Any = None
    options: list[str] | None = None
    min_val: float | None = None
    max_val: float | None = None
    is_file_url: bool = False
    is_file_url_array: bool = False
    accepted_file_types: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class KieSchemaBundle:
    model_id: str
    params: list[KieParamInfo]
    output_type: str  # "image", "video", "audio", "dict"


def _extract_model_id(text: str) -> str | None:
    """Extract the kie.ai model ID from the documentation markdown."""
    m = re.search(r"\*\*Format\*\*\s*\|\s*`([^`]+)`", text)
    if m:
        return m.group(1).strip()

    m = re.search(
        r"[Mm]odel\s+name,?\s*format:\s*`([^`]+)`", text
    )
    if m:
        return m.group(1).strip()

    m = re.search(r'"model"\s*:\s*"([^"]+)"', text)
    if m:
        return m.group(1).strip()

    return None


def _infer_output_type(model_id: str, text: str) -> str:
    """Infer whether the model produces images, videos, or audio."""
    lower = model_id.lower()

    if any(
        kw in lower
        for kw in (
            "video",
            "storyboard",
            "avatar",
            "seedance",
            "kling",
            "hailuo",
            "sora",
            "wan",
            "infinitalk",
        )
    ):
        return "video"

    if any(kw in lower for kw in ("audio", "music", "suno", "speech", "tts")):
        return "audio"

    result_url_match = re.search(r"resultUrls.*?\[.*?\"([^\"]+)\"", text)
    if result_url_match:
        url = result_url_match.group(1).lower()
        if any(url.endswith(ext) for ext in (".mp4", ".webm", ".mov")):
            return "video"
        if any(url.endswith(ext) for ext in (".mp3", ".wav", ".ogg", ".flac")):
            return "audio"

    return "image"


def _parse_input_params(text: str) -> list[KieParamInfo]:
    """Parse the `### input Object Parameters` section of kie.ai docs."""
    section_match = re.search(
        r"###\s*input\s+Object\s+Parameters(.*?)(?=\n##\s|\n---|\Z)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if not section_match:
        return []

    section = section_match.group(1)

    param_blocks = re.split(r"\n####\s+", section)
    params: list[KieParamInfo] = []

    for block in param_blocks:
        block = block.strip()
        if not block:
            continue

        name_match = re.match(r"^(\w+)", block)
        if not name_match:
            continue
        name = name_match.group(1)

        if name in HIDDEN_PARAMS:
            continue

        type_match = re.search(
            r"\*\*Type\*\*:\s*`([^`]+)`", block
        )
        param_type = type_match.group(1).strip() if type_match else "string"

        req_match = re.search(
            r"\*\*Required\*\*:\s*(Yes|No)", block, re.IGNORECASE
        )
        required = req_match.group(1).lower() == "yes" if req_match else False

        desc_match = re.search(
            r"\*\*Description\*\*:\s*(.+?)(?=\n\s*-\s*\*\*|\Z)",
            block,
            re.DOTALL,
        )
        description = desc_match.group(1).strip() if desc_match else ""

        default_val: Any = None
        default_match = re.search(
            r"\*\*Default Value\*\*:\s*`([^`]*)`", block
        )
        if default_match:
            raw_default = default_match.group(1)
            default_val = _coerce_default(raw_default, param_type)
        elif not default_match:
            default_match_json = re.search(
                r"\*\*Default Value\*\*:\s*(`[^`]*`|\[.*?\]|\"[^\"]*\")",
                block,
            )
            if default_match_json:
                try:
                    default_val = json.loads(default_match_json.group(1))
                except (json.JSONDecodeError, ValueError):
                    pass

        options: list[str] | None = None
        options_match = re.search(
            r"\*\*Options\*\*:\s*\n((?:\s*-\s*`[^`]+`.*\n?)+)", block
        )
        if options_match:
            options = re.findall(r"`([^`]+)`", options_match.group(1))

        max_match = re.search(r"\*\*Range\*\*:\s*`(\d+)`\s*to\s*`(\d+)`", block)
        min_val: float | None = None
        max_val: float | None = None
        if max_match:
            min_val = float(max_match.group(1))
            max_val = float(max_match.group(2))
        else:
            range_parts = re.findall(
                r"-\s*(?:Range|Min|Max|Minimum|Maximum):\s*`?(\d+(?:\.\d+)?)`?",
                block,
                re.IGNORECASE,
            )
            if len(range_parts) >= 2:
                min_val = float(range_parts[0])
                max_val = float(range_parts[1])

        is_file_url = False
        is_file_url_array = False
        accepted_file_types: list[str] = []

        file_types_match = re.search(
            r"\*\*Accepted File Types\*\*:\s*(.+)", block
        )
        if file_types_match:
            accepted_file_types = [
                t.strip()
                for t in file_types_match.group(1).split(",")
            ]

        if accepted_file_types or "Upload" in description or "URL" in description:
            if param_type == "array":
                is_file_url_array = True
            else:
                is_file_url = True

        if name.endswith("_url") and param_type == "string":
            is_file_url = True
        if name.endswith("_urls") and param_type == "array":
            is_file_url_array = True

        params.append(
            KieParamInfo(
                name=name,
                type=param_type,
                required=required,
                description=description,
                default=default_val,
                options=options,
                min_val=min_val,
                max_val=max_val,
                is_file_url=is_file_url,
                is_file_url_array=is_file_url_array,
                accepted_file_types=accepted_file_types,
            )
        )

    return params


def _coerce_default(raw: str, param_type: str) -> Any:
    if param_type == "integer":
        try:
            return int(raw)
        except ValueError:
            return raw
    if param_type == "number":
        try:
            return float(raw)
        except ValueError:
            return raw
    if param_type == "boolean":
        return raw.lower() in ("true", "1", "yes")
    if raw.startswith("[") or raw.startswith("{"):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            pass
    if raw.startswith('"') and raw.endswith('"'):
        return raw[1:-1]
    return raw


def parse_kie_docs(text: str) -> KieSchemaBundle:
    """Parse kie.ai API documentation markdown into a schema bundle."""
    model_id = _extract_model_id(text)
    if not model_id:
        raise ValueError(
            "Could not find model ID in the documentation. "
            "Look for a line like: Model name, format: `model-id`"
        )

    params = _parse_input_params(text)
    output_type = _infer_output_type(model_id, text)

    return KieSchemaBundle(
        model_id=model_id,
        params=params,
        output_type=output_type,
    )


def _param_to_type_metadata(p: KieParamInfo) -> TypeMetadata:
    if p.is_file_url_array:
        file_types = [ft.lower() for ft in p.accepted_file_types]
        if any("video" in ft for ft in file_types):
            return TypeMetadata(
                type="list", type_args=[TypeMetadata(type="video")]
            )
        if any("audio" in ft for ft in file_types):
            return TypeMetadata(
                type="list", type_args=[TypeMetadata(type="audio")]
            )
        return TypeMetadata(
            type="list", type_args=[TypeMetadata(type="image")]
        )

    if p.is_file_url:
        file_types = [ft.lower() for ft in p.accepted_file_types]
        if any("video" in ft for ft in file_types):
            return TypeMetadata(type="video")
        if any("audio" in ft for ft in file_types):
            return TypeMetadata(type="audio")
        return TypeMetadata(type="image")

    if p.options:
        return TypeMetadata(type="enum", values=p.options)

    type_map = {
        "string": "str",
        "integer": "int",
        "number": "float",
        "boolean": "bool",
        "array": "list",
    }
    return TypeMetadata(type=type_map.get(p.type, "str"))


def _param_default_value(p: KieParamInfo) -> Any:
    if p.is_file_url_array:
        return []
    if p.is_file_url:
        return ImageRef()
    if p.default is not None:
        return p.default
    if p.options:
        return p.options[0]
    defaults = {
        "string": "",
        "integer": 0,
        "number": 0.0,
        "boolean": False,
        "array": [],
    }
    return defaults.get(p.type, "")


def _output_type_to_metadata(output_type: str) -> dict[str, TypeMetadata]:
    if output_type == "video":
        return {"video": TypeMetadata(type="video")}
    if output_type == "audio":
        return {"audio": TypeMetadata(type="audio")}
    return {"image": TypeMetadata(type="image")}


def _type_metadata_to_dict(meta: TypeMetadata) -> dict[str, Any]:
    out: dict[str, Any] = {
        "type": meta.type,
        "type_args": [],
        "optional": getattr(meta, "optional", False),
    }
    if getattr(meta, "values", None):
        out["values"] = meta.values
    if getattr(meta, "type_args", None):
        out["type_args"] = [
            _type_metadata_to_dict(a) if isinstance(a, TypeMetadata) else a
            for a in meta.type_args
        ]
    return out


def _bundle_to_resolve_result(bundle: KieSchemaBundle) -> dict[str, Any]:
    """Convert a parsed schema bundle into the API response format."""
    dynamic_properties: dict[str, Any] = {}
    dynamic_inputs: dict[str, Any] = {}

    for p in bundle.params:
        default = _param_default_value(p)
        if isinstance(default, ImageRef):
            dynamic_properties[p.name] = default.model_dump()
        elif isinstance(default, list) and any(
            isinstance(item, ImageRef) for item in default
        ):
            dynamic_properties[p.name] = [
                item.model_dump() if isinstance(item, ImageRef) else item
                for item in default
            ]
        else:
            dynamic_properties[p.name] = default

        meta = _param_to_type_metadata(p)
        entry = _type_metadata_to_dict(meta)
        entry["description"] = p.description
        entry["default"] = dynamic_properties[p.name]
        if p.min_val is not None:
            entry["min"] = p.min_val
        if p.max_val is not None:
            entry["max"] = p.max_val
        if p.options:
            entry["values"] = p.options
        dynamic_inputs[p.name] = entry

    output_types = _output_type_to_metadata(bundle.output_type)
    dynamic_outputs = {
        name: _type_metadata_to_dict(meta)
        for name, meta in output_types.items()
    }

    return {
        "model_id": bundle.model_id,
        "dynamic_properties": dynamic_properties,
        "dynamic_inputs": dynamic_inputs,
        "dynamic_outputs": dynamic_outputs,
    }


async def resolve_dynamic_schema(model_info: str) -> dict[str, Any]:
    """
    Parse pasted kie.ai API documentation and return the schema for the UI.

    Returns dict with:
      - model_id: str
      - dynamic_properties: dict[str, Any]
      - dynamic_inputs: dict[str, dict]
      - dynamic_outputs: dict[str, dict]
    """
    raw = (model_info or "").strip()
    if not raw:
        raise ValueError(
            "model_info is required: paste kie.ai API documentation"
        )

    bundle = parse_kie_docs(raw)
    return _bundle_to_resolve_result(bundle)


class DynamicKie(KieBaseNode):
    """
    Dynamic Kie.ai node for running any kie.ai model.
    kie, dynamic, schema, api, inference, runtime, model

    Use cases:
    - Call any kie.ai model without a dedicated Python node
    - Prototype workflows with new models as they appear
    - Run models by pasting their API documentation
    - Access the full kie.ai catalog dynamically
    """

    _is_dynamic = True
    _supports_dynamic_outputs = True
    _dynamic_input_types: dict[str, TypeMetadata] = {}

    _auto_save_asset: ClassVar[bool] = True
    _poll_interval: float = 2.0
    _max_poll_attempts: int = 300

    model_info: str = Field(
        default="",
        description="Paste the full API documentation from the kie.ai model page.",
    )

    _parsed_bundle: KieSchemaBundle | None = None

    def __init__(self, **data: Any):
        super().__init__(**data)
        self._dynamic_input_types = {}
        self._parsed_bundle = None
        self._prime_from_docs()

    @classmethod
    def get_node_type(cls) -> str:
        return "kie.DynamicKie"

    @classmethod
    def get_namespace(cls) -> str:
        return "kie"

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["model_info"]

    def _prime_from_docs(self) -> None:
        """Parse docs at init time to populate dynamic slots from saved state."""
        if not self.model_info.strip():
            return
        try:
            bundle = parse_kie_docs(self.model_info)
            self._parsed_bundle = bundle
            self._apply_bundle(bundle)
        except ValueError:
            pass

    def _apply_bundle(self, bundle: KieSchemaBundle) -> None:
        self._dynamic_input_types = {}
        for p in bundle.params:
            self._dynamic_input_types[p.name] = _param_to_type_metadata(p)
            if p.name not in self._dynamic_properties:
                self._dynamic_properties[p.name] = _param_default_value(p)

        output_types = _output_type_to_metadata(bundle.output_type)
        self._dynamic_outputs = output_types

    def _get_model(self) -> str:
        if self._parsed_bundle:
            return self._parsed_bundle.model_id
        bundle = parse_kie_docs(self.model_info)
        return bundle.model_id

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        bundle = self._parsed_bundle or parse_kie_docs(self.model_info)
        param_lookup = {p.name: p for p in bundle.params}
        arguments: dict[str, Any] = {}

        for name, value in self._dynamic_properties.items():
            if name not in param_lookup:
                continue
            p = param_lookup[name]

            if value is None:
                if p.required:
                    raise ValueError(f"Missing required input: {name}")
                continue

            if p.is_file_url_array:
                urls = await self._resolve_file_urls(value, p, context)
                if urls:
                    arguments[name] = urls
                elif p.required:
                    raise ValueError(f"Missing required input: {name}")
                continue

            if p.is_file_url:
                url = await self._resolve_single_file_url(value, p, context)
                if url:
                    arguments[name] = url
                elif p.required:
                    raise ValueError(f"Missing required input: {name}")
                continue

            if isinstance(value, str) and not value and not p.required:
                continue
            if isinstance(value, list) and not value and not p.required:
                continue

            arguments[name] = value

        return arguments

    async def _resolve_file_urls(
        self,
        value: Any,
        param: KieParamInfo,
        context: ProcessingContext | None,
    ) -> list[str]:
        if not isinstance(value, list):
            value = [value]
        urls: list[str] = []
        for item in value:
            url = await self._resolve_single_file_url(item, param, context)
            if url:
                urls.append(url)
        return urls

    async def _resolve_single_file_url(
        self,
        value: Any,
        param: KieParamInfo,
        context: ProcessingContext | None,
    ) -> str | None:
        if context is None:
            return None

        if isinstance(value, dict) and "type" in value:
            asset_type = value.get("type", "")
            if asset_type == "image" or "image" in asset_type:
                ref = ImageRef(**{k: v for k, v in value.items() if k != "type"})
                if ref.is_set():
                    return await self._upload_image(context, ref)
            elif asset_type == "video" or "video" in asset_type:
                from nodetool.metadata.types import VideoRef as VR
                ref = VR(**{k: v for k, v in value.items() if k != "type"})
                if ref.is_set():
                    return await self._upload_video(context, ref)
            elif asset_type == "audio" or "audio" in asset_type:
                from nodetool.metadata.types import AudioRef as AR
                ref = AR(**{k: v for k, v in value.items() if k != "type"})
                if ref.is_set():
                    return await self._upload_audio(context, ref)

        if isinstance(value, ImageRef) and value.is_set():
            return await self._upload_image(context, value)

        if isinstance(value, str) and value.startswith(("http://", "https://")):
            return value

        return None

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        if not self.model_info.strip():
            raise ValueError(
                "model_info is empty. Paste kie.ai API documentation to configure this node."
            )

        bundle = parse_kie_docs(self.model_info)
        self._parsed_bundle = bundle
        self._apply_bundle(bundle)

        result_bytes, task_id = await self._execute_task(context)

        output_type = bundle.output_type
        if output_type == "video":
            video = await context.video_from_bytes(
                result_bytes, metadata={"task_id": task_id}
            )
            return {"video": video}
        elif output_type == "audio":
            audio = await context.audio_from_bytes(result_bytes)
            return {"audio": audio}
        else:
            image = await context.image_from_bytes(
                result_bytes, metadata={"task_id": task_id}
            )
            return {"image": image}

    def find_property(self, name: str):
        from nodetool.workflows.property import Property

        if name in self._dynamic_input_types:
            return Property(name=name, type=self._dynamic_input_types[name])
        return super().find_property(name)

    def find_output_instance(self, name: str):
        slot = super().find_output_instance(name)
        if slot is not None:
            return slot
        if self._supports_dynamic_outputs:
            from nodetool.metadata.types import OutputSlot

            return OutputSlot(type=TypeMetadata(type="any"), name=name)
        return None
