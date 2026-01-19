"""Kie.ai audio/music generation nodes.

This module provides nodes for generating audio using Kie.ai's APIs:
- Suno Music API (AI music generation with vocals and instrumentals)
- ElevenLabs Text-to-Speech API (AI voice generation)
"""

import asyncio
from enum import Enum
from typing import Any, ClassVar

import aiohttp
from pydantic import Field

from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import AudioRef, ImageRef, TextRef, VideoRef
from nodetool.workflows.processing_context import ProcessingContext

from .image import KIE_API_BASE_URL, KieBaseNode

log = get_logger(__name__)


class VocalGender(str, Enum):
    UNSPECIFIED = ""
    MALE = "m"
    FEMALE = "f"


DUMMY_CALLBACK_URL = "https://example.com/callback"


class GenerateMusic(KieBaseNode):
    """Generate music using Suno AI via Kie.ai.

    kie, suno, music, audio, ai, generation, vocals, instrumental

    Creates full tracks with vocals and instrumentals using Suno models.
    Supports custom mode for strict lyric control and non-custom mode for easy prompts.

    Use cases:
    - Generate background music for projects
    - Create AI-composed songs with vocals
    - Produce instrumentals for content
    - Generate music in various genres and styles
    """

    _expose_as_tool: ClassVar[bool] = True
    _poll_interval: float = 4.0
    _max_poll_attempts: int = 120

    custom_mode: bool = Field(
        default=False,
        description="Enable custom mode for detailed control over style and title.",
    )

    prompt: str = Field(
        default="",
        description=(
            "Music description or lyrics. In custom mode, this is used as lyrics "
            "when instrumental is false. In non-custom mode, this is the core idea."
        ),
    )

    style: str = Field(
        default="",
        description="Music style specification (required in custom mode).",
    )

    title: str = Field(
        default="",
        description="Track title (required in custom mode, max 80 characters).",
    )

    instrumental: bool = Field(
        default=False,
        description="Generate instrumental-only (no vocals).",
    )

    class Model(str, Enum):
        V4 = "V4"
        V4_5 = "V4_5"
        V4_5PLUS = "V4_5PLUS"
        V4_5ALL = "V4_5ALL"
        V5 = "V5"

    model: Model = Field(
        default=Model.V4_5PLUS,
        description="Suno model version to use.",
    )

    negative_tags: str = Field(
        default="",
        description="Music styles or traits to exclude from the generated audio.",
    )

    vocal_gender: VocalGender = Field(
        default=VocalGender.UNSPECIFIED,
        description="Vocal gender preference (custom mode only).",
    )

    style_weight: float = Field(
        default=0.0,
        description="Strength of adherence to style (0-1).",
    )

    weirdness_constraint: float = Field(
        default=0.0,
        description="Creative deviation control (0-1).",
    )

    audio_weight: float = Field(
        default=0.0,
        description="Balance weight for audio features (0-1).",
    )

    persona_id: str = Field(
        default="",
        description="Persona ID to apply (custom mode only).",
    )

    @classmethod
    def get_title(cls) -> str:
        return "Generate Music"

    def _get_model(self) -> str:
        return "suno"

    def _get_prompt_limit(self) -> int:
        if self.custom_mode:
            if self.model == self.Model.V4:
                return 3000
            return 5000
        return 500

    def _get_style_limit(self) -> int:
        if self.model == self.Model.V4:
            return 200
        return 1000

    def _validate_length(self, value: str, limit: int, field_name: str) -> None:
        if value and len(value) > limit:
            raise ValueError(f"{field_name} exceeds {limit} characters")

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        prompt_limit = self._get_prompt_limit()
        self._validate_length(self.prompt, prompt_limit, "prompt")

        payload: dict[str, Any] = {
            "customMode": self.custom_mode,
            "instrumental": self.instrumental,
            "callBackUrl": DUMMY_CALLBACK_URL,
            "model": self.model.value,
            "prompt": self.prompt or "",
        }

        if self.custom_mode:
            if not self.style:
                raise ValueError("style is required when custom_mode is true")
            if not self.title:
                raise ValueError("title is required when custom_mode is true")
            if not self.instrumental and not self.prompt:
                raise ValueError(
                    "prompt is required when custom_mode is true and instrumental is false"
                )

            self._validate_length(self.style, self._get_style_limit(), "style")
            self._validate_length(self.title, 80, "title")

            payload["style"] = self.style
            payload["title"] = self.title

            if self.negative_tags:
                payload["negativeTags"] = self.negative_tags
            if self.vocal_gender != VocalGender.UNSPECIFIED:
                payload["vocalGender"] = self.vocal_gender.value
            if self.style_weight != 0.0:
                payload["styleWeight"] = self.style_weight
            if self.weirdness_constraint != 0.0:
                payload["weirdnessConstraint"] = self.weirdness_constraint
            if self.audio_weight != 0.0:
                payload["audioWeight"] = self.audio_weight
            if self.persona_id:
                payload["personaId"] = self.persona_id
        else:
            if not self.prompt:
                raise ValueError("prompt is required when custom_mode is false")

        return payload

    async def _submit_task(
        self,
        session: aiohttp.ClientSession,
        api_key: str,
        context: ProcessingContext | None = None,
    ) -> dict[str, Any]:
        url = f"{KIE_API_BASE_URL}/api/v1/generate"
        payload = await self._get_input_params(context)
        headers = self._get_headers(api_key)
        log.info(f"Submitting Suno task to {url} with payload: {payload}")
        async with session.post(url, json=payload, headers=headers) as response:
            response_data = await response.json()
            if "code" in response_data:
                self._check_response_status(response_data)

            if response.status != 200:
                raise ValueError(
                    f"Failed to submit task: {response.status} - {response_data}"
                )
            return response_data

    async def _poll_status(
        self, session: aiohttp.ClientSession, api_key: str, task_id: str
    ) -> dict[str, Any]:
        url = f"{KIE_API_BASE_URL}/api/v1/generate/record-info?taskId={task_id}"
        headers = self._get_headers(api_key)

        failed_statuses = {
            "CREATE_TASK_FAILED",
            "GENERATE_AUDIO_FAILED",
            "CALLBACK_EXCEPTION",
            "SENSITIVE_WORD_ERROR",
        }

        for attempt in range(self._max_poll_attempts):
            log.debug(
                f"Polling Suno task status (attempt {attempt + 1}/{self._max_poll_attempts})"
            )
            async with session.get(url, headers=headers) as response:
                status_data = await response.json()
                log.info("Suno status response: %s", status_data)
                if "code" in status_data:
                    self._check_response_status(status_data)

                status = status_data.get("data", {}).get("status")
                if status == "SUCCESS":
                    return status_data
                if status in failed_statuses:
                    error_message = status_data.get("data", {}).get("errorMessage")
                    error_code = status_data.get("data", {}).get("errorCode")
                    raise ValueError(
                        f"Suno task failed: {status} ({error_code}) {error_message}"
                    )

            await asyncio.sleep(self._poll_interval)

        raise TimeoutError(
            f"Task did not complete within {self._max_poll_attempts * self._poll_interval} seconds"
        )

    async def _download_result(
        self, session: aiohttp.ClientSession, api_key: str, task_id: str
    ) -> bytes:
        url = f"{KIE_API_BASE_URL}/api/v1/generate/record-info?taskId={task_id}"
        headers = self._get_headers(api_key)

        async with session.get(url, headers=headers) as response:
            if response.status != 200:
                response_text = await response.text()
                raise ValueError(
                    f"Failed to get result: {response.status} - {response_text}"
                )

            status_data = await response.json()
            log.info("Suno result response: %s", status_data)
            if "code" in status_data:
                self._check_response_status(status_data)

            response_payload = status_data.get("data", {}).get("response", {})
            if isinstance(response_payload, str):
                try:
                    import json

                    response_payload = json.loads(response_payload)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid response payload JSON: {response_payload}"
                    ) from exc
            suno_data = response_payload.get("sunoData", [])
            if not suno_data:
                raise ValueError("No sunoData found in response")

            first_track = suno_data[0]
            audio_url = first_track.get("audioUrl") or first_track.get("audio_url")
            if not audio_url:
                raise ValueError("No audioUrl found in sunoData")

            log.debug(f"Downloading audio from {audio_url}")
            async with session.get(audio_url, headers=headers) as audio_response:
                if audio_response.status != 200:
                    raise ValueError(
                        f"Failed to download audio: {audio_response.status}"
                    )
                audio_bytes = await audio_response.read()
                if not audio_bytes:
                    raise ValueError("Downloaded audio was empty")
                return audio_bytes

    async def process(self, context: ProcessingContext) -> AudioRef:
        audio_bytes, _ = await self._execute_task(context)
        return await context.audio_from_bytes(audio_bytes)


class ExtendMusic(KieBaseNode):
    """Extend music using Suno AI via Kie.ai.

    kie, suno, music, audio, ai, extension, continuation, remix

    Extends an existing track by continuing from a specified time point.
    Can reuse original parameters or override them with custom settings.
    """

    _expose_as_tool: ClassVar[bool] = True
    _poll_interval: float = 4.0
    _max_poll_attempts: int = 120

    default_param_flag: bool = Field(
        default=False,
        description=(
            "If true, use custom parameters (prompt/style/title/continue_at). "
            "If false, inherit parameters from the source audio."
        ),
    )

    audio_id: str = Field(
        default="",
        description="Audio ID to extend.",
    )

    prompt: str = Field(
        default="",
        description="Description of the desired extension content.",
    )

    style: str = Field(
        default="",
        description="Music style for the extension (required for custom params).",
    )

    title: str = Field(
        default="",
        description="Title for the extended track (required for custom params).",
    )

    continue_at: float = Field(
        default=0.0,
        description="Time in seconds to start extending from (required for custom params).",
        ge=0.0,
    )

    class Model(str, Enum):
        V4 = "V4"
        V4_5 = "V4_5"
        V4_5PLUS = "V4_5PLUS"
        V4_5ALL = "V4_5ALL"
        V5 = "V5"

    model: Model = Field(
        default=Model.V4_5PLUS,
        description="Suno model version to use (must match source audio).",
    )

    negative_tags: str = Field(
        default="",
        description="Music styles or traits to exclude from the extension.",
    )

    vocal_gender: VocalGender = Field(
        default=VocalGender.UNSPECIFIED,
        description="Vocal gender preference.",
    )

    style_weight: float = Field(
        default=0.0,
        description="Strength of adherence to style (0-1).",
    )

    weirdness_constraint: float = Field(
        default=0.0,
        description="Creative deviation control (0-1).",
    )

    audio_weight: float = Field(
        default=0.0,
        description="Balance weight for audio features (0-1).",
    )

    persona_id: str = Field(
        default="",
        description="Persona ID to apply (custom params only).",
    )

    @classmethod
    def get_title(cls) -> str:
        return "Extend Music"

    def _get_model(self) -> str:
        return "suno"

    def _get_prompt_limit(self) -> int:
        if self.model == self.Model.V4:
            return 3000
        return 5000

    def _get_style_limit(self) -> int:
        if self.model == self.Model.V4:
            return 200
        return 1000

    def _get_title_limit(self) -> int:
        if self.model in {self.Model.V4_5, self.Model.V4_5PLUS, self.Model.V5}:
            return 100
        return 80

    def _validate_length(self, value: str, limit: int, field_name: str) -> None:
        if value and len(value) > limit:
            raise ValueError(f"{field_name} exceeds {limit} characters")

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if not self.audio_id:
            raise ValueError("audio_id is required")

        payload: dict[str, Any] = {
            "defaultParamFlag": self.default_param_flag,
            "audioId": self.audio_id,
            "callBackUrl": DUMMY_CALLBACK_URL,
            "model": self.model.value,
            "prompt": self.prompt or "",
        }

        if self.default_param_flag:
            if not self.prompt:
                raise ValueError("prompt is required when default_param_flag is true")
            if not self.style:
                raise ValueError("style is required when default_param_flag is true")
            if not self.title:
                raise ValueError("title is required when default_param_flag is true")
            if self.continue_at <= 0:
                raise ValueError(
                    "continue_at must be greater than 0 when default_param_flag is true"
                )

            self._validate_length(self.prompt, self._get_prompt_limit(), "prompt")
            self._validate_length(self.style, self._get_style_limit(), "style")
            self._validate_length(self.title, self._get_title_limit(), "title")

            payload["style"] = self.style
            payload["title"] = self.title
            payload["continueAt"] = self.continue_at

            if self.negative_tags:
                payload["negativeTags"] = self.negative_tags
            if self.vocal_gender != VocalGender.UNSPECIFIED:
                payload["vocalGender"] = self.vocal_gender.value
            if self.style_weight != 0.0:
                payload["styleWeight"] = self.style_weight
            if self.weirdness_constraint != 0.0:
                payload["weirdnessConstraint"] = self.weirdness_constraint
            if self.audio_weight != 0.0:
                payload["audioWeight"] = self.audio_weight
            if self.persona_id:
                payload["personaId"] = self.persona_id
        else:
            if self.prompt:
                self._validate_length(self.prompt, self._get_prompt_limit(), "prompt")

        return payload

    async def _submit_task(
        self,
        session: aiohttp.ClientSession,
        api_key: str,
        context: ProcessingContext | None = None,
    ) -> dict[str, Any]:
        url = f"{KIE_API_BASE_URL}/api/v1/generate/extend"
        payload = await self._get_input_params(context)
        headers = self._get_headers(api_key)
        log.info(f"Submitting Suno extend task to {url} with payload: {payload}")
        async with session.post(url, json=payload, headers=headers) as response:
            response_data = await response.json()
            if "code" in response_data:
                self._check_response_status(response_data)

            if response.status != 200:
                raise ValueError(
                    f"Failed to submit task: {response.status} - {response_data}"
                )
            return response_data

    async def _poll_status(
        self, session: aiohttp.ClientSession, api_key: str, task_id: str
    ) -> dict[str, Any]:
        url = f"{KIE_API_BASE_URL}/api/v1/generate/record-info?taskId={task_id}"
        headers = self._get_headers(api_key)

        failed_statuses = {
            "CREATE_TASK_FAILED",
            "GENERATE_AUDIO_FAILED",
            "CALLBACK_EXCEPTION",
            "SENSITIVE_WORD_ERROR",
        }

        for attempt in range(self._max_poll_attempts):
            log.debug(
                f"Polling Suno task status (attempt {attempt + 1}/{self._max_poll_attempts})"
            )
            async with session.get(url, headers=headers) as response:
                status_data = await response.json()
                log.info("Suno extend status response: %s", status_data)
                if "code" in status_data:
                    self._check_response_status(status_data)

                status = status_data.get("data", {}).get("status")
                if status == "SUCCESS":
                    return status_data
                if status in failed_statuses:
                    error_message = status_data.get("data", {}).get("errorMessage")
                    error_code = status_data.get("data", {}).get("errorCode")
                    raise ValueError(
                        f"Suno task failed: {status} ({error_code}) {error_message}"
                    )

            await asyncio.sleep(self._poll_interval)

        raise TimeoutError(
            f"Task did not complete within {self._max_poll_attempts * self._poll_interval} seconds"
        )

    async def _download_result(
        self, session: aiohttp.ClientSession, api_key: str, task_id: str
    ) -> bytes:
        url = f"{KIE_API_BASE_URL}/api/v1/generate/record-info?taskId={task_id}"
        headers = self._get_headers(api_key)

        async with session.get(url, headers=headers) as response:
            if response.status != 200:
                response_text = await response.text()
                raise ValueError(
                    f"Failed to get result: {response.status} - {response_text}"
                )

            status_data = await response.json()
            log.info("Suno extend result response: %s", status_data)
            if "code" in status_data:
                self._check_response_status(status_data)

            response_payload = status_data.get("data", {}).get("response", {})
            if isinstance(response_payload, str):
                try:
                    import json

                    response_payload = json.loads(response_payload)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid response payload JSON: {response_payload}"
                    ) from exc

            suno_data = response_payload.get("sunoData", [])
            if not suno_data:
                raise ValueError("No sunoData found in response")

            first_track = suno_data[0]
            audio_url = first_track.get("audioUrl") or first_track.get("audio_url")
            if not audio_url:
                raise ValueError("No audioUrl found in sunoData")

            log.debug(f"Downloading audio from {audio_url}")
            async with session.get(audio_url, headers=headers) as audio_response:
                if audio_response.status != 200:
                    raise ValueError(
                        f"Failed to download audio: {audio_response.status}"
                    )
                audio_bytes = await audio_response.read()
                if not audio_bytes:
                    raise ValueError("Downloaded audio was empty")
                return audio_bytes

    async def process(self, context: ProcessingContext) -> AudioRef:
        audio_bytes, _ = await self._execute_task(context)
        return await context.audio_from_bytes(audio_bytes)


class CoverAudio(KieBaseNode):
    """Cover an uploaded audio track using Suno AI via Kie.ai.

    kie, suno, music, audio, ai, cover, upload, style transfer

    Uploads a source track and generates a covered version in a new style while
    retaining the original melody.
    """

    _expose_as_tool: ClassVar[bool] = True
    _poll_interval: float = 4.0
    _max_poll_attempts: int = 120

    custom_mode: bool = Field(
        default=False,
        description="Enable custom mode for detailed control over style and title.",
    )

    audio: AudioRef = Field(
        default=AudioRef(),
        description="Source audio to upload for covering.",
    )

    prompt: str = Field(
        default="",
        description=(
            "Music description or lyrics. In custom mode, this is used as lyrics "
            "when instrumental is false. In non-custom mode, this is the core idea."
        ),
    )

    style: str = Field(
        default="",
        description="Music style specification (required in custom mode).",
    )

    title: str = Field(
        default="",
        description="Track title (required in custom mode).",
    )

    instrumental: bool = Field(
        default=False,
        description="Generate instrumental-only (no vocals).",
    )

    class Model(str, Enum):
        V4 = "V4"
        V4_5 = "V4_5"
        V4_5PLUS = "V4_5PLUS"
        V4_5ALL = "V4_5ALL"
        V5 = "V5"

    model: Model = Field(
        default=Model.V4_5PLUS,
        description="Suno model version to use.",
    )

    negative_tags: str = Field(
        default="",
        description="Music styles or traits to exclude from the generated audio.",
    )

    vocal_gender: VocalGender = Field(
        default=VocalGender.UNSPECIFIED,
        description="Vocal gender preference (custom mode only).",
    )

    style_weight: float = Field(
        default=0.0,
        description="Strength of adherence to style (0-1).",
    )

    weirdness_constraint: float = Field(
        default=0.0,
        description="Creative deviation control (0-1).",
    )

    audio_weight: float = Field(
        default=0.0,
        description="Balance weight for audio features (0-1).",
    )

    persona_id: str = Field(
        default="",
        description="Persona ID to apply (custom mode only).",
    )

    @classmethod
    def get_title(cls) -> str:
        return "Cover Audio"

    def _get_model(self) -> str:
        return "suno"

    def _get_prompt_limit(self) -> int:
        if self.custom_mode:
            if self.model == self.Model.V4:
                return 3000
            return 5000
        return 500

    def _get_style_limit(self) -> int:
        if self.model == self.Model.V4:
            return 200
        return 1000

    def _get_title_limit(self) -> int:
        if self.model in {self.Model.V4_5, self.Model.V4_5PLUS, self.Model.V5}:
            return 100
        return 80

    def _validate_length(self, value: str, limit: int, field_name: str) -> None:
        if value and len(value) > limit:
            raise ValueError(f"{field_name} exceeds {limit} characters")

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if context is None:
            raise ValueError("Processing context is required for audio upload")

        upload_url = await self._upload_audio(context, self.audio)

        prompt_limit = self._get_prompt_limit()
        self._validate_length(self.prompt, prompt_limit, "prompt")

        payload: dict[str, Any] = {
            "uploadUrl": upload_url,
            "customMode": self.custom_mode,
            "instrumental": self.instrumental,
            "callBackUrl": DUMMY_CALLBACK_URL,
            "model": self.model.value,
            "prompt": self.prompt or "",
        }

        if self.custom_mode:
            if not self.style:
                raise ValueError("style is required when custom_mode is true")
            if not self.title:
                raise ValueError("title is required when custom_mode is true")
            if not self.instrumental and not self.prompt:
                raise ValueError(
                    "prompt is required when custom_mode is true and instrumental is false"
                )

            self._validate_length(self.style, self._get_style_limit(), "style")
            self._validate_length(self.title, self._get_title_limit(), "title")

            payload["style"] = self.style
            payload["title"] = self.title

            if self.negative_tags:
                payload["negativeTags"] = self.negative_tags
            if self.vocal_gender != VocalGender.UNSPECIFIED:
                payload["vocalGender"] = self.vocal_gender.value
            if self.style_weight != 0.0:
                payload["styleWeight"] = self.style_weight
            if self.weirdness_constraint != 0.0:
                payload["weirdnessConstraint"] = self.weirdness_constraint
            if self.audio_weight != 0.0:
                payload["audioWeight"] = self.audio_weight
            if self.persona_id:
                payload["personaId"] = self.persona_id
        else:
            if not self.prompt:
                raise ValueError("prompt is required when custom_mode is false")

        return payload

    async def _submit_task(
        self,
        session: aiohttp.ClientSession,
        api_key: str,
        context: ProcessingContext | None = None,
    ) -> dict[str, Any]:
        url = f"{KIE_API_BASE_URL}/api/v1/generate/upload-cover"
        payload = await self._get_input_params(context)
        headers = self._get_headers(api_key)
        log.info(f"Submitting Suno cover task to {url} with payload: {payload}")
        async with session.post(url, json=payload, headers=headers) as response:
            response_data = await response.json()
            if "code" in response_data:
                self._check_response_status(response_data)

            if response.status != 200:
                raise ValueError(
                    f"Failed to submit task: {response.status} - {response_data}"
                )
            return response_data

    async def _poll_status(
        self, session: aiohttp.ClientSession, api_key: str, task_id: str
    ) -> dict[str, Any]:
        url = f"{KIE_API_BASE_URL}/api/v1/generate/record-info?taskId={task_id}"
        headers = self._get_headers(api_key)

        failed_statuses = {
            "CREATE_TASK_FAILED",
            "GENERATE_AUDIO_FAILED",
            "CALLBACK_EXCEPTION",
            "SENSITIVE_WORD_ERROR",
        }

        for attempt in range(self._max_poll_attempts):
            log.debug(
                f"Polling Suno task status (attempt {attempt + 1}/{self._max_poll_attempts})"
            )
            async with session.get(url, headers=headers) as response:
                status_data = await response.json()
                log.info("Suno cover status response: %s", status_data)
                if "code" in status_data:
                    self._check_response_status(status_data)

                status = status_data.get("data", {}).get("status")
                if status == "SUCCESS":
                    return status_data
                if status in failed_statuses:
                    error_message = status_data.get("data", {}).get("errorMessage")
                    error_code = status_data.get("data", {}).get("errorCode")
                    raise ValueError(
                        f"Suno task failed: {status} ({error_code}) {error_message}"
                    )

            await asyncio.sleep(self._poll_interval)

        raise TimeoutError(
            f"Task did not complete within {self._max_poll_attempts * self._poll_interval} seconds"
        )

    async def _download_result(
        self, session: aiohttp.ClientSession, api_key: str, task_id: str
    ) -> bytes:
        url = f"{KIE_API_BASE_URL}/api/v1/generate/record-info?taskId={task_id}"
        headers = self._get_headers(api_key)

        async with session.get(url, headers=headers) as response:
            if response.status != 200:
                response_text = await response.text()
                raise ValueError(
                    f"Failed to get result: {response.status} - {response_text}"
                )

            status_data = await response.json()
            log.info("Suno cover result response: %s", status_data)
            if "code" in status_data:
                self._check_response_status(status_data)

            response_payload = status_data.get("data", {}).get("response", {})
            if isinstance(response_payload, str):
                try:
                    import json

                    response_payload = json.loads(response_payload)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid response payload JSON: {response_payload}"
                    ) from exc

            suno_data = response_payload.get("sunoData", [])
            if not suno_data:
                raise ValueError("No sunoData found in response")

            first_track = suno_data[0]
            audio_url = first_track.get("audioUrl") or first_track.get("audio_url")
            if not audio_url:
                raise ValueError("No audioUrl found in sunoData")

            log.debug(f"Downloading audio from {audio_url}")
            async with session.get(audio_url, headers=headers) as audio_response:
                if audio_response.status != 200:
                    raise ValueError(
                        f"Failed to download audio: {audio_response.status}"
                    )
                audio_bytes = await audio_response.read()
                if not audio_bytes:
                    raise ValueError("Downloaded audio was empty")
                return audio_bytes

    async def process(self, context: ProcessingContext) -> AudioRef:
        audio_bytes, _ = await self._execute_task(context)
        return await context.audio_from_bytes(audio_bytes)


class AddInstrumental(KieBaseNode):
    """Add instrumental accompaniment to uploaded audio via Suno AI.

    kie, suno, music, audio, ai, instrumental, accompaniment, upload

    Uploads a source track (e.g., vocals/stems) and generates a backing track.
    """

    _expose_as_tool: ClassVar[bool] = True
    _poll_interval: float = 4.0
    _max_poll_attempts: int = 120

    audio: AudioRef = Field(
        default=AudioRef(),
        description="Source audio to upload for instrumental generation.",
    )

    title: str = Field(
        default="",
        description="Title of the generated music.",
    )

    tags: str = Field(
        default="",
        description="Music styles or tags to include in the generated music.",
    )

    negative_tags: str = Field(
        default="",
        description="Music styles or characteristics to exclude.",
    )

    class Model(str, Enum):
        V4_5PLUS = "V4_5PLUS"
        V5 = "V5"

    model: Model = Field(
        default=Model.V4_5PLUS,
        description="Suno model version to use.",
    )

    vocal_gender: VocalGender = Field(
        default=VocalGender.UNSPECIFIED,
        description="Vocal gender preference.",
    )

    style_weight: float = Field(
        default=0.0,
        description="Strength of adherence to style (0-1).",
    )

    weirdness_constraint: float = Field(
        default=0.0,
        description="Creative deviation control (0-1).",
    )

    audio_weight: float = Field(
        default=0.0,
        description="Balance weight for audio features (0-1).",
    )

    @classmethod
    def get_title(cls) -> str:
        return "Add Instrumental"

    def _get_model(self) -> str:
        return "suno"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if not self.title:
            raise ValueError("title is required")
        if not self.tags:
            raise ValueError("tags is required")
        if not self.negative_tags:
            raise ValueError("negative_tags is required")

        if context is None:
            raise ValueError("Processing context is required for audio upload")

        upload_url = await self._upload_audio(context, self.audio)

        payload: dict[str, Any] = {
            "uploadUrl": upload_url,
            "title": self.title,
            "tags": self.tags,
            "negativeTags": self.negative_tags,
            "callBackUrl": DUMMY_CALLBACK_URL,
            "model": self.model.value,
        }

        if self.vocal_gender != VocalGender.UNSPECIFIED:
            payload["vocalGender"] = self.vocal_gender.value
        if self.style_weight != 0.0:
            payload["styleWeight"] = self.style_weight
        if self.weirdness_constraint != 0.0:
            payload["weirdnessConstraint"] = self.weirdness_constraint
        if self.audio_weight != 0.0:
            payload["audioWeight"] = self.audio_weight

        return payload

    async def _submit_task(
        self,
        session: aiohttp.ClientSession,
        api_key: str,
        context: ProcessingContext | None = None,
    ) -> dict[str, Any]:
        url = f"{KIE_API_BASE_URL}/api/v1/generate/add-instrumental"
        payload = await self._get_input_params(context)
        headers = self._get_headers(api_key)
        log.info(f"Submitting Suno add-instrumental task to {url} with payload: {payload}")
        async with session.post(url, json=payload, headers=headers) as response:
            response_data = await response.json()
            if "code" in response_data:
                self._check_response_status(response_data)

            if response.status != 200:
                raise ValueError(
                    f"Failed to submit task: {response.status} - {response_data}"
                )
            return response_data

    async def _poll_status(
        self, session: aiohttp.ClientSession, api_key: str, task_id: str
    ) -> dict[str, Any]:
        url = f"{KIE_API_BASE_URL}/api/v1/generate/record-info?taskId={task_id}"
        headers = self._get_headers(api_key)

        failed_statuses = {
            "CREATE_TASK_FAILED",
            "GENERATE_AUDIO_FAILED",
            "CALLBACK_EXCEPTION",
            "SENSITIVE_WORD_ERROR",
        }

        for attempt in range(self._max_poll_attempts):
            log.debug(
                f"Polling Suno task status (attempt {attempt + 1}/{self._max_poll_attempts})"
            )
            async with session.get(url, headers=headers) as response:
                status_data = await response.json()
                log.info("Suno add-instrumental status response: %s", status_data)
                if "code" in status_data:
                    self._check_response_status(status_data)

                status = status_data.get("data", {}).get("status")
                if status == "SUCCESS":
                    return status_data
                if status in failed_statuses:
                    error_message = status_data.get("data", {}).get("errorMessage")
                    error_code = status_data.get("data", {}).get("errorCode")
                    raise ValueError(
                        f"Suno task failed: {status} ({error_code}) {error_message}"
                    )

            await asyncio.sleep(self._poll_interval)

        raise TimeoutError(
            f"Task did not complete within {self._max_poll_attempts * self._poll_interval} seconds"
        )

    async def _download_result(
        self, session: aiohttp.ClientSession, api_key: str, task_id: str
    ) -> bytes:
        url = f"{KIE_API_BASE_URL}/api/v1/generate/record-info?taskId={task_id}"
        headers = self._get_headers(api_key)

        async with session.get(url, headers=headers) as response:
            if response.status != 200:
                response_text = await response.text()
                raise ValueError(
                    f"Failed to get result: {response.status} - {response_text}"
                )

            status_data = await response.json()
            log.info("Suno add-instrumental result response: %s", status_data)
            if "code" in status_data:
                self._check_response_status(status_data)

            response_payload = status_data.get("data", {}).get("response", {})
            if isinstance(response_payload, str):
                try:
                    import json

                    response_payload = json.loads(response_payload)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid response payload JSON: {response_payload}"
                    ) from exc

            suno_data = response_payload.get("sunoData", [])
            if not suno_data:
                raise ValueError("No sunoData found in response")

            first_track = suno_data[0]
            audio_url = first_track.get("audioUrl") or first_track.get("audio_url")
            if not audio_url:
                raise ValueError("No audioUrl found in sunoData")

            log.debug(f"Downloading audio from {audio_url}")
            async with session.get(audio_url, headers=headers) as audio_response:
                if audio_response.status != 200:
                    raise ValueError(
                        f"Failed to download audio: {audio_response.status}"
                    )
                audio_bytes = await audio_response.read()
                if not audio_bytes:
                    raise ValueError("Downloaded audio was empty")
                return audio_bytes

    async def process(self, context: ProcessingContext) -> AudioRef:
        audio_bytes, _ = await self._execute_task(context)
        return await context.audio_from_bytes(audio_bytes)


class AddVocals(KieBaseNode):
    """Add AI vocals to uploaded audio via Suno AI.

    kie, suno, music, audio, ai, vocals, singing, upload

    Uploads an instrumental track and generates vocal layers on top.
    """

    _expose_as_tool: ClassVar[bool] = True
    _poll_interval: float = 4.0
    _max_poll_attempts: int = 120

    audio: AudioRef = Field(
        default=AudioRef(),
        description="Source audio to upload for vocal generation.",
    )

    prompt: str = Field(
        default="",
        description="Prompt describing lyric content and singing style.",
    )

    title: str = Field(
        default="",
        description="Title of the generated music.",
    )

    style: str = Field(
        default="",
        description="Music style for vocal generation.",
    )

    tags: str = Field(
        default="",
        description="Optional music tags to include in the generation.",
    )

    negative_tags: str = Field(
        default="",
        description="Excluded music styles or elements.",
    )

    class Model(str, Enum):
        V4_5PLUS = "V4_5PLUS"
        V5 = "V5"

    model: Model = Field(
        default=Model.V4_5PLUS,
        description="Suno model version to use.",
    )

    vocal_gender: VocalGender = Field(
        default=VocalGender.UNSPECIFIED,
        description="Vocal gender preference.",
    )

    style_weight: float = Field(
        default=0.0,
        description="Strength of adherence to style (0-1).",
    )

    weirdness_constraint: float = Field(
        default=0.0,
        description="Creative deviation control (0-1).",
    )

    audio_weight: float = Field(
        default=0.0,
        description="Balance weight for audio features (0-1).",
    )

    @classmethod
    def get_title(cls) -> str:
        return "Add Vocals"

    def _get_model(self) -> str:
        return "suno"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if not self.prompt:
            raise ValueError("prompt is required")
        if not self.title:
            raise ValueError("title is required")
        if not self.style:
            raise ValueError("style is required")
        if not self.negative_tags:
            raise ValueError("negative_tags is required")

        if context is None:
            raise ValueError("Processing context is required for audio upload")

        upload_url = await self._upload_audio(context, self.audio)

        payload: dict[str, Any] = {
            "uploadUrl": upload_url,
            "prompt": self.prompt,
            "title": self.title,
            "style": self.style,
            "negativeTags": self.negative_tags,
            "callBackUrl": DUMMY_CALLBACK_URL,
            "model": self.model.value,
        }

        if self.tags:
            payload["tags"] = self.tags
        if self.vocal_gender != VocalGender.UNSPECIFIED:
            payload["vocalGender"] = self.vocal_gender.value
        if self.style_weight != 0.0:
            payload["styleWeight"] = self.style_weight
        if self.weirdness_constraint != 0.0:
            payload["weirdnessConstraint"] = self.weirdness_constraint
        if self.audio_weight != 0.0:
            payload["audioWeight"] = self.audio_weight

        return payload

    async def _submit_task(
        self,
        session: aiohttp.ClientSession,
        api_key: str,
        context: ProcessingContext | None = None,
    ) -> dict[str, Any]:
        url = f"{KIE_API_BASE_URL}/api/v1/generate/add-vocals"
        payload = await self._get_input_params(context)
        headers = self._get_headers(api_key)
        log.info(f"Submitting Suno add-vocals task to {url} with payload: {payload}")
        async with session.post(url, json=payload, headers=headers) as response:
            response_data = await response.json()
            if "code" in response_data:
                self._check_response_status(response_data)

            if response.status != 200:
                raise ValueError(
                    f"Failed to submit task: {response.status} - {response_data}"
                )
            return response_data

    async def _poll_status(
        self, session: aiohttp.ClientSession, api_key: str, task_id: str
    ) -> dict[str, Any]:
        url = f"{KIE_API_BASE_URL}/api/v1/generate/record-info?taskId={task_id}"
        headers = self._get_headers(api_key)

        failed_statuses = {
            "CREATE_TASK_FAILED",
            "GENERATE_AUDIO_FAILED",
            "CALLBACK_EXCEPTION",
            "SENSITIVE_WORD_ERROR",
        }

        for attempt in range(self._max_poll_attempts):
            log.debug(
                f"Polling Suno task status (attempt {attempt + 1}/{self._max_poll_attempts})"
            )
            async with session.get(url, headers=headers) as response:
                status_data = await response.json()
                log.info("Suno add-vocals status response: %s", status_data)
                if "code" in status_data:
                    self._check_response_status(status_data)

                status = status_data.get("data", {}).get("status")
                if status == "SUCCESS":
                    return status_data
                if status in failed_statuses:
                    error_message = status_data.get("data", {}).get("errorMessage")
                    error_code = status_data.get("data", {}).get("errorCode")
                    raise ValueError(
                        f"Suno task failed: {status} ({error_code}) {error_message}"
                    )

            await asyncio.sleep(self._poll_interval)

        raise TimeoutError(
            f"Task did not complete within {self._max_poll_attempts * self._poll_interval} seconds"
        )

    async def _download_result(
        self, session: aiohttp.ClientSession, api_key: str, task_id: str
    ) -> bytes:
        url = f"{KIE_API_BASE_URL}/api/v1/generate/record-info?taskId={task_id}"
        headers = self._get_headers(api_key)

        async with session.get(url, headers=headers) as response:
            if response.status != 200:
                response_text = await response.text()
                raise ValueError(
                    f"Failed to get result: {response.status} - {response_text}"
                )

            status_data = await response.json()
            log.info("Suno add-vocals result response: %s", status_data)
            if "code" in status_data:
                self._check_response_status(status_data)

            response_payload = status_data.get("data", {}).get("response", {})
            if isinstance(response_payload, str):
                try:
                    import json

                    response_payload = json.loads(response_payload)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid response payload JSON: {response_payload}"
                    ) from exc

            suno_data = response_payload.get("sunoData", [])
            if not suno_data:
                raise ValueError("No sunoData found in response")

            first_track = suno_data[0]
            audio_url = first_track.get("audioUrl") or first_track.get("audio_url")
            if not audio_url:
                raise ValueError("No audioUrl found in sunoData")

            log.debug(f"Downloading audio from {audio_url}")
            async with session.get(audio_url, headers=headers) as audio_response:
                if audio_response.status != 200:
                    raise ValueError(
                        f"Failed to download audio: {audio_response.status}"
                    )
                audio_bytes = await audio_response.read()
                if not audio_bytes:
                    raise ValueError("Downloaded audio was empty")
                return audio_bytes

    async def process(self, context: ProcessingContext) -> AudioRef:
        audio_bytes, _ = await self._execute_task(context)
        return await context.audio_from_bytes(audio_bytes)


class GetTimestampedLyrics(KieBaseNode):
    """Retrieve timestamped lyrics for a generated Suno track.

    kie, suno, music, audio, lyrics, timestamps, karaoke

    Fetches word-level alignment and waveform data for a specific task/audio pair.
    """

    _expose_as_tool: ClassVar[bool] = True

    task_id: str = Field(
        default="",
        description="Task ID from Generate Music or Extend Music.",
    )

    audio_id: str = Field(
        default="",
        description="Audio ID for the specific track.",
    )

    @classmethod
    def get_title(cls) -> str:
        return "Get Timestamped Lyrics"

    def _get_model(self) -> str:
        return "suno"

    async def process(self, context: ProcessingContext) -> TextRef:
        if not self.task_id:
            raise ValueError("task_id is required")
        if not self.audio_id:
            raise ValueError("audio_id is required")

        api_key = await self._get_api_key(context)
        headers = self._get_headers(api_key)
        payload = {"taskId": self.task_id, "audioId": self.audio_id}
        url = f"{KIE_API_BASE_URL}/api/v1/generate/get-timestamped-lyrics"
        log.info(f"Requesting timestamped lyrics from {url} with payload: {payload}")

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as response:
                response_data = await response.json()
                if "code" in response_data:
                    self._check_response_status(response_data)

                if response.status != 200:
                    raise ValueError(
                        f"Failed to fetch timestamped lyrics: {response.status} - {response_data}"
                    )

                data = response_data.get("data", {}) or {}
                import json

                return await context.text_from_str(
                    json.dumps(data, ensure_ascii=False)
                )


class BoostMusicStyle(KieBaseNode):
    """Boost music style text using Suno V4_5 style generation.

    kie, suno, music, style, prompt, enhancement
    """

    _expose_as_tool: ClassVar[bool] = True

    content: str = Field(
        default="",
        description="Style description to enhance.",
    )

    @classmethod
    def get_title(cls) -> str:
        return "Boost Music Style"

    def _get_model(self) -> str:
        return "suno"

    async def process(self, context: ProcessingContext) -> TextRef:
        if not self.content:
            raise ValueError("content is required")

        api_key = await self._get_api_key(context)
        headers = self._get_headers(api_key)
        payload = {"content": self.content}
        url = f"{KIE_API_BASE_URL}/api/v1/style/generate"
        log.info(f"Requesting boosted style from {url} with payload: {payload}")

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as response:
                response_data = await response.json()
                if "code" in response_data:
                    self._check_response_status(response_data)

                if response.status != 200:
                    raise ValueError(
                        f"Failed to boost style: {response.status} - {response_data}"
                    )

                data = response_data.get("data", {}) or {}
                result_text = data.get("result") or ""
                return await context.text_from_str(result_text)


class GenerateMusicCover(KieBaseNode):
    """Generate cover images for a Suno music task.

    kie, suno, music, cover, image, artwork
    """

    _expose_as_tool: ClassVar[bool] = True
    _poll_interval: float = 4.0
    _max_poll_attempts: int = 120

    task_id: str = Field(
        default="",
        description="Original music task ID.",
    )

    @classmethod
    def get_title(cls) -> str:
        return "Generate Music Cover"

    def _get_model(self) -> str:
        return "suno"

    async def process(self, context: ProcessingContext) -> list[ImageRef]:
        if not self.task_id:
            raise ValueError("task_id is required")
        api_key = await self._get_api_key(context)
        headers = self._get_headers(api_key)
        payload = {"taskId": self.task_id, "callBackUrl": DUMMY_CALLBACK_URL}
        url = f"{KIE_API_BASE_URL}/api/v1/suno/cover/generate"
        log.info(f"Requesting cover generation from {url} with payload: {payload}")

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as response:
                response_data = await response.json()
                if "code" in response_data:
                    self._check_response_status(response_data)

                if response.status != 200:
                    raise ValueError(
                        f"Failed to generate cover: {response.status} - {response_data}"
                    )

                task_id = response_data.get("data", {}).get("taskId")
                if not task_id:
                    raise ValueError(f"No taskId in response: {response_data}")

            return await self._poll_cover_result(session, api_key, task_id, context)

    async def _poll_cover_result(
        self,
        session: aiohttp.ClientSession,
        api_key: str,
        task_id: str,
        context: ProcessingContext,
    ) -> list[ImageRef]:
        url = f"{KIE_API_BASE_URL}/api/v1/suno/cover/record-info?taskId={task_id}"
        headers = self._get_headers(api_key)

        for attempt in range(self._max_poll_attempts):
            log.debug(
                f"Polling cover task status (attempt {attempt + 1}/{self._max_poll_attempts})"
            )
            async with session.get(url, headers=headers) as response:
                status_data = await response.json()
                log.info("Cover record-info response: %s", status_data)
                if "code" in status_data:
                    self._check_response_status(status_data)

                data = status_data.get("data", {}) or {}
                success_flag = data.get("successFlag")
                if success_flag == 1:
                    images = data.get("response", {}).get("images", []) or []
                    if not images:
                        raise ValueError("No images found in cover response")
                    return list(
                        await asyncio.gather(
                            *[
                                context.image_from_url(image_url)
                                for image_url in images
                            ]
                        )
                    )
                if success_flag == 3:
                    error_message = data.get("errorMessage") or "Cover generation failed"
                    raise ValueError(error_message)

            await asyncio.sleep(self._poll_interval)

        raise TimeoutError(
            f"Cover generation did not complete within {self._max_poll_attempts * self._poll_interval} seconds"
        )


class ReplaceMusicSection(KieBaseNode):
    """Replace a section of a generated Suno track.

    kie, suno, music, replace, edit, infill

    Regenerates a time range and blends it into the original track.
    """

    _expose_as_tool: ClassVar[bool] = True
    _poll_interval: float = 4.0
    _max_poll_attempts: int = 120

    task_id: str = Field(
        default="",
        description="Original music task ID.",
    )

    audio_id: str = Field(
        default="",
        description="Audio ID to replace.",
    )

    prompt: str = Field(
        default="",
        description="Prompt describing the replacement segment content.",
    )

    tags: str = Field(
        default="",
        description="Music style tags.",
    )

    title: str = Field(
        default="",
        description="Music title.",
    )

    infill_start_s: float = Field(
        default=0.0,
        description="Start time point for replacement (seconds).",
        ge=0.0,
    )

    infill_end_s: float = Field(
        default=0.0,
        description="End time point for replacement (seconds).",
        ge=0.0,
    )

    negative_tags: str = Field(
        default="",
        description="Excluded music styles for the replacement segment.",
    )

    full_lyrics: str = Field(
        default="",
        description="Full lyrics after modification.",
    )

    @classmethod
    def get_title(cls) -> str:
        return "Replace Music Section"

    def _get_model(self) -> str:
        return "suno"

    def _validate_time_range(self) -> None:
        if self.infill_start_s <= 0 or self.infill_end_s <= 0:
            raise ValueError("infill_start_s and infill_end_s must be > 0")
        if self.infill_start_s >= self.infill_end_s:
            raise ValueError("infill_start_s must be less than infill_end_s")
        duration = self.infill_end_s - self.infill_start_s
        if duration < 6 or duration > 60:
            raise ValueError(
                "Replacement duration must be between 6 and 60 seconds"
            )

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if not self.task_id:
            raise ValueError("task_id is required")
        if not self.audio_id:
            raise ValueError("audio_id is required")
        if not self.prompt:
            raise ValueError("prompt is required")
        if not self.tags:
            raise ValueError("tags is required")
        if not self.title:
            raise ValueError("title is required")

        self._validate_time_range()

        payload: dict[str, Any] = {
            "taskId": self.task_id,
            "audioId": self.audio_id,
            "prompt": self.prompt,
            "tags": self.tags,
            "title": self.title,
            "infillStartS": round(self.infill_start_s, 2),
            "infillEndS": round(self.infill_end_s, 2),
        }

        if self.negative_tags:
            payload["negativeTags"] = self.negative_tags
        if self.full_lyrics:
            payload["fullLyrics"] = self.full_lyrics
        return payload

    async def _submit_task(
        self,
        session: aiohttp.ClientSession,
        api_key: str,
        context: ProcessingContext | None = None,
    ) -> dict[str, Any]:
        url = f"{KIE_API_BASE_URL}/api/v1/generate/replace-section"
        payload = await self._get_input_params(context)
        headers = self._get_headers(api_key)
        log.info(f"Submitting Suno replace-section task to {url} with payload: {payload}")
        async with session.post(url, json=payload, headers=headers) as response:
            response_data = await response.json()
            if "code" in response_data:
                self._check_response_status(response_data)

            if response.status != 200:
                raise ValueError(
                    f"Failed to submit task: {response.status} - {response_data}"
                )
            return response_data

    async def _poll_status(
        self, session: aiohttp.ClientSession, api_key: str, task_id: str
    ) -> dict[str, Any]:
        url = f"{KIE_API_BASE_URL}/api/v1/generate/record-info?taskId={task_id}"
        headers = self._get_headers(api_key)

        failed_statuses = {
            "CREATE_TASK_FAILED",
            "GENERATE_AUDIO_FAILED",
            "CALLBACK_EXCEPTION",
            "SENSITIVE_WORD_ERROR",
        }

        for attempt in range(self._max_poll_attempts):
            log.debug(
                f"Polling Suno task status (attempt {attempt + 1}/{self._max_poll_attempts})"
            )
            async with session.get(url, headers=headers) as response:
                status_data = await response.json()
                if "code" in status_data:
                    self._check_response_status(status_data)

                status = status_data.get("data", {}).get("status")
                if status == "SUCCESS":
                    return status_data
                if status in failed_statuses:
                    error_message = status_data.get("data", {}).get("errorMessage")
                    error_code = status_data.get("data", {}).get("errorCode")
                    raise ValueError(
                        f"Suno task failed: {status} ({error_code}) {error_message}"
                    )

            await asyncio.sleep(self._poll_interval)

        raise TimeoutError(
            f"Task did not complete within {self._max_poll_attempts * self._poll_interval} seconds"
        )

    async def _download_result(
        self, session: aiohttp.ClientSession, api_key: str, task_id: str
    ) -> bytes:
        url = f"{KIE_API_BASE_URL}/api/v1/generate/record-info?taskId={task_id}"
        headers = self._get_headers(api_key)

        async with session.get(url, headers=headers) as response:
            if response.status != 200:
                response_text = await response.text()
                raise ValueError(
                    f"Failed to get result: {response.status} - {response_text}"
                )

            status_data = await response.json()
            if "code" in status_data:
                self._check_response_status(status_data)

            response_payload = status_data.get("data", {}).get("response", {})
            if isinstance(response_payload, str):
                try:
                    import json

                    response_payload = json.loads(response_payload)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid response payload JSON: {response_payload}"
                    ) from exc

            suno_data = response_payload.get("sunoData", [])
            if not suno_data:
                raise ValueError("No sunoData found in response")

            first_track = suno_data[0]
            audio_candidates = [
                first_track.get("audioUrl"),
                first_track.get("audio_url"),
                first_track.get("sourceAudioUrl"),
                first_track.get("source_audio_url"),
                first_track.get("streamAudioUrl"),
                first_track.get("stream_audio_url"),
                first_track.get("sourceStreamAudioUrl"),
                first_track.get("source_stream_audio_url"),
            ]
            audio_urls = [url for url in audio_candidates if url]
            if not audio_urls:
                raise ValueError("No audio URL found in sunoData")

            for audio_url in audio_urls:
                log.debug(f"Downloading audio from {audio_url}")
                async with session.get(audio_url, headers=headers) as audio_response:
                    if audio_response.status != 200:
                        log.warning(
                            "Failed to download audio from %s: %s",
                            audio_url,
                            audio_response.status,
                        )
                        continue
                    audio_bytes = await audio_response.read()
                    if audio_bytes:
                        return audio_bytes

            raise ValueError("Downloaded audio was empty or inaccessible")

    async def process(self, context: ProcessingContext) -> AudioRef:
        audio_bytes, _ = await self._execute_task(context)
        return await context.audio_from_bytes(audio_bytes)


class GenerateLyrics(KieBaseNode):
    """Generate lyrics based on a text prompt via Kie.ai.

    kie, suno, lyrics, text, songwriting, prompt
    """

    _expose_as_tool: ClassVar[bool] = True
    _poll_interval: float = 4.0
    _max_poll_attempts: int = 120

    prompt: str = Field(
        default="",
        description="Prompt describing the theme, mood, or style of the lyrics.",
    )

    @classmethod
    def get_title(cls) -> str:
        return "Generate Lyrics"

    def _get_model(self) -> str:
        return "suno"

    def _validate_prompt_length(self) -> None:
        word_count = len(self.prompt.split())
        if word_count > 200:
            raise ValueError("prompt exceeds 200 words")

    async def process(self, context: ProcessingContext) -> list[dict[str, Any]]:
        if not self.prompt:
            raise ValueError("prompt is required")
        self._validate_prompt_length()

        api_key = await self._get_api_key(context)
        headers = self._get_headers(api_key)
        payload = {"prompt": self.prompt, "callBackUrl": DUMMY_CALLBACK_URL}
        url = f"{KIE_API_BASE_URL}/api/v1/lyrics"
        log.info(f"Requesting lyrics from {url} with payload: {payload}")

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as response:
                response_data = await response.json()
                if "code" in response_data:
                    self._check_response_status(response_data)

                if response.status != 200:
                    raise ValueError(
                        f"Failed to generate lyrics: {response.status} - {response_data}"
                    )

                task_id = response_data.get("data", {}).get("taskId")
                if not task_id:
                    raise ValueError(f"No taskId in response: {response_data}")

            return await self._poll_lyrics_result(session, api_key, task_id, context)

    async def _poll_lyrics_result(
        self,
        session: aiohttp.ClientSession,
        api_key: str,
        task_id: str,
        context: ProcessingContext,
    ) -> list[dict[str, Any]]:
        url = f"{KIE_API_BASE_URL}/api/v1/lyrics/record-info?taskId={task_id}"
        headers = self._get_headers(api_key)

        failed_statuses = {
            "CREATE_TASK_FAILED",
            "GENERATE_LYRICS_FAILED",
            "CALLBACK_EXCEPTION",
            "SENSITIVE_WORD_ERROR",
        }

        for attempt in range(self._max_poll_attempts):
            log.debug(
                f"Polling lyrics task status (attempt {attempt + 1}/{self._max_poll_attempts})"
            )
            async with session.get(url, headers=headers) as response:
                status_data = await response.json()
                log.info("Lyrics record-info response: %s", status_data)
                if "code" in status_data:
                    self._check_response_status(status_data)

                data = status_data.get("data", {}) or {}
                status = data.get("status")
                if status == "SUCCESS":
                    response_payload = data.get("response", {}) or {}
                    lyric_items = response_payload.get("data", []) or []
                    results: list[dict[str, Any]] = []
                    for item in lyric_items:
                        text = item.get("text") or ""
                        results.append(
                            {
                                "title": item.get("title") or "",
                                "status": item.get("status") or "",
                                "error_message": item.get("errorMessage") or "",
                                "lyrics": await context.text_from_str(text),
                            }
                        )
                    return results
                if status in failed_statuses:
                    error_message = data.get("errorMessage") or "Lyrics generation failed"
                    raise ValueError(error_message)

            await asyncio.sleep(self._poll_interval)

        raise TimeoutError(
            f"Lyrics generation did not complete within {self._max_poll_attempts * self._poll_interval} seconds"
        )


class ConvertToWav(KieBaseNode):
    """Convert a generated music track to WAV format.

    kie, suno, music, audio, wav, conversion
    """

    _expose_as_tool: ClassVar[bool] = True
    _poll_interval: float = 4.0
    _max_poll_attempts: int = 120

    task_id: str = Field(
        default="",
        description="Original music task ID.",
    )

    audio_id: str = Field(
        default="",
        description="Audio ID to convert to WAV.",
    )

    @classmethod
    def get_title(cls) -> str:
        return "Convert To WAV"

    def _get_model(self) -> str:
        return "suno"

    async def process(self, context: ProcessingContext) -> AudioRef:
        if not self.task_id:
            raise ValueError("task_id is required")
        if not self.audio_id:
            raise ValueError("audio_id is required")
        api_key = await self._get_api_key(context)
        headers = self._get_headers(api_key)
        payload = {
            "taskId": self.task_id,
            "audioId": self.audio_id,
            "callBackUrl": DUMMY_CALLBACK_URL,
        }
        url = f"{KIE_API_BASE_URL}/api/v1/wav/generate"
        log.info(f"Requesting WAV conversion from {url} with payload: {payload}")

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as response:
                response_data = await response.json()
                if "code" in response_data:
                    self._check_response_status(response_data)

                if response.status != 200:
                    raise ValueError(
                        f"Failed to convert to WAV: {response.status} - {response_data}"
                    )

                task_id = response_data.get("data", {}).get("taskId")
                if not task_id:
                    raise ValueError(f"No taskId in response: {response_data}")

            return await self._poll_wav_result(session, api_key, task_id, context)

    async def _poll_wav_result(
        self,
        session: aiohttp.ClientSession,
        api_key: str,
        task_id: str,
        context: ProcessingContext,
    ) -> AudioRef:
        url = f"{KIE_API_BASE_URL}/api/v1/wav/record-info?taskId={task_id}"
        headers = self._get_headers(api_key)

        failed_statuses = {
            "CREATE_TASK_FAILED",
            "GENERATE_WAV_FAILED",
            "CALLBACK_EXCEPTION",
        }

        for attempt in range(self._max_poll_attempts):
            log.debug(
                f"Polling WAV task status (attempt {attempt + 1}/{self._max_poll_attempts})"
            )
            async with session.get(url, headers=headers) as response:
                status_data = await response.json()
                log.info("WAV record-info response: %s", status_data)
                if "code" in status_data:
                    self._check_response_status(status_data)

                data = status_data.get("data", {}) or {}
                status = data.get("successFlag")
                if status == "SUCCESS":
                    audio_url = data.get("response", {}).get("audioWavUrl")
                    if not audio_url:
                        raise ValueError("No audioWavUrl found in response")
                    buffer = await context.download_file(audio_url)
                    return await context.audio_from_io(
                        buffer, content_type="audio/wav"
                    )
                if status in failed_statuses:
                    error_message = data.get("errorMessage") or "WAV conversion failed"
                    raise ValueError(error_message)

            await asyncio.sleep(self._poll_interval)

        raise TimeoutError(
            f"WAV conversion did not complete within {self._max_poll_attempts * self._poll_interval} seconds"
        )


class GenerateMusicVideo(KieBaseNode):
    """Create a music video visualization for a generated track.

    kie, suno, music, video, mp4, visualization
    """

    _expose_as_tool: ClassVar[bool] = True
    _poll_interval: float = 4.0
    _max_poll_attempts: int = 120

    task_id: str = Field(
        default="",
        description="Original music task ID.",
    )

    audio_id: str = Field(
        default="",
        description="Audio ID to visualize.",
    )

    author: str = Field(
        default="",
        description="Optional artist/creator name (max 50 chars).",
    )

    domain_name: str = Field(
        default="",
        description="Optional domain watermark (max 50 chars).",
    )

    @classmethod
    def get_title(cls) -> str:
        return "Generate Music Video"

    def _get_model(self) -> str:
        return "suno"

    async def process(self, context: ProcessingContext) -> VideoRef:
        if not self.task_id:
            raise ValueError("task_id is required")
        if not self.audio_id:
            raise ValueError("audio_id is required")
        api_key = await self._get_api_key(context)
        headers = self._get_headers(api_key)
        payload: dict[str, Any] = {
            "taskId": self.task_id,
            "audioId": self.audio_id,
            "callBackUrl": DUMMY_CALLBACK_URL,
        }
        if self.author:
            payload["author"] = self.author
        if self.domain_name:
            payload["domainName"] = self.domain_name

        url = f"{KIE_API_BASE_URL}/api/v1/mp4/generate"
        log.info(f"Requesting music video from {url} with payload: {payload}")

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as response:
                response_data = await response.json()
                if "code" in response_data:
                    self._check_response_status(response_data)

                if response.status != 200:
                    raise ValueError(
                        f"Failed to generate music video: {response.status} - {response_data}"
                    )

                task_id = response_data.get("data", {}).get("taskId")
                if not task_id:
                    raise ValueError(f"No taskId in response: {response_data}")

            return await self._poll_video_result(session, api_key, task_id, context)

    async def _poll_video_result(
        self,
        session: aiohttp.ClientSession,
        api_key: str,
        task_id: str,
        context: ProcessingContext,
    ) -> VideoRef:
        url = f"{KIE_API_BASE_URL}/api/v1/mp4/record-info?taskId={task_id}"
        headers = self._get_headers(api_key)

        failed_statuses = {
            "CREATE_TASK_FAILED",
            "GENERATE_MP4_FAILED",
        }

        for attempt in range(self._max_poll_attempts):
            log.debug(
                f"Polling MP4 task status (attempt {attempt + 1}/{self._max_poll_attempts})"
            )
            async with session.get(url, headers=headers) as response:
                status_data = await response.json()
                log.info("MP4 record-info response: %s", status_data)
                if "code" in status_data:
                    self._check_response_status(status_data)

                data = status_data.get("data", {}) or {}
                status = data.get("successFlag")
                if status == "SUCCESS":
                    video_url = data.get("response", {}).get("videoUrl")
                    if not video_url:
                        raise ValueError("No videoUrl found in response")
                    buffer = await context.download_file(video_url)
                    return await context.video_from_io(
                        buffer, content_type="video/mp4"
                    )
                if status in failed_statuses:
                    error_message = data.get("errorMessage") or "Video generation failed"
                    raise ValueError(error_message)

            await asyncio.sleep(self._poll_interval)

        raise TimeoutError(
            f"Video generation did not complete within {self._max_poll_attempts * self._poll_interval} seconds"
        )


class VocalStemSeparation(KieBaseNode):
    """Separate a track into vocal/instrument stems via Suno.

    kie, suno, music, stems, separation, vocals, instrumental
    """

    _expose_as_tool: ClassVar[bool] = True
    _poll_interval: float = 4.0
    _max_poll_attempts: int = 120

    task_id: str = Field(
        default="",
        description="Original music task ID.",
    )

    audio_id: str = Field(
        default="",
        description="Audio ID to separate.",
    )

    class SeparationType(str, Enum):
        SEPARATE_VOCAL = "separate_vocal"
        SPLIT_STEM = "split_stem"

    separation_type: SeparationType = Field(
        default=SeparationType.SEPARATE_VOCAL,
        description="Separation mode.",
    )

    @classmethod
    def get_title(cls) -> str:
        return "Vocal & Instrument Stem Separation"

    def _get_model(self) -> str:
        return "suno"

    async def process(self, context: ProcessingContext) -> dict[str, AudioRef]:
        if not self.task_id:
            raise ValueError("task_id is required")
        if not self.audio_id:
            raise ValueError("audio_id is required")
        api_key = await self._get_api_key(context)
        headers = self._get_headers(api_key)
        payload = {
            "taskId": self.task_id,
            "audioId": self.audio_id,
            "type": self.separation_type.value,
            "callBackUrl": DUMMY_CALLBACK_URL,
        }
        url = f"{KIE_API_BASE_URL}/api/v1/vocal-removal/generate"
        log.info(f"Requesting vocal separation from {url} with payload: {payload}")

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as response:
                response_data = await response.json()
                if "code" in response_data:
                    self._check_response_status(response_data)

                if response.status != 200:
                    raise ValueError(
                        f"Failed to start vocal separation: {response.status} - {response_data}"
                    )

                task_id = response_data.get("data", {}).get("taskId")
                if not task_id:
                    raise ValueError(f"No taskId in response: {response_data}")

            return await self._poll_separation_result(
                session, api_key, task_id, context
            )

    async def _poll_separation_result(
        self,
        session: aiohttp.ClientSession,
        api_key: str,
        task_id: str,
        context: ProcessingContext,
    ) -> dict[str, AudioRef]:
        url = f"{KIE_API_BASE_URL}/api/v1/vocal-removal/record-info?taskId={task_id}"
        headers = self._get_headers(api_key)

        failed_statuses = {
            "CREATE_TASK_FAILED",
            "GENERATE_AUDIO_FAILED",
            "CALLBACK_EXCEPTION",
        }

        for attempt in range(self._max_poll_attempts):
            log.debug(
                f"Polling vocal separation status (attempt {attempt + 1}/{self._max_poll_attempts})"
            )
            async with session.get(url, headers=headers) as response:
                status_data = await response.json()
                log.info("Vocal separation record-info response: %s", status_data)
                if "code" in status_data:
                    self._check_response_status(status_data)

                data = status_data.get("data", {}) or {}
                status = data.get("successFlag")
                if status == "SUCCESS":
                    response_payload = data.get("response", {}) or {}
                    return await self._convert_stems(response_payload, context)
                if status in failed_statuses:
                    error_message = data.get("errorMessage") or "Stem separation failed"
                    raise ValueError(error_message)

            await asyncio.sleep(self._poll_interval)

        raise TimeoutError(
            f"Stem separation did not complete within {self._max_poll_attempts * self._poll_interval} seconds"
        )

    async def _convert_stems(
        self, response_payload: dict[str, Any], context: ProcessingContext
    ) -> dict[str, AudioRef]:
        url_map = {
            "origin": response_payload.get("originUrl"),
            "vocal": response_payload.get("vocalUrl"),
            "instrumental": response_payload.get("instrumentalUrl"),
            "backing_vocals": response_payload.get("backingVocalsUrl"),
            "drums": response_payload.get("drumsUrl"),
            "bass": response_payload.get("bassUrl"),
            "guitar": response_payload.get("guitarUrl"),
            "piano": response_payload.get("pianoUrl"),
            "keyboard": response_payload.get("keyboardUrl"),
            "percussion": response_payload.get("percussionUrl"),
            "strings": response_payload.get("stringsUrl"),
            "synth": response_payload.get("synthUrl"),
            "fx": response_payload.get("fxUrl"),
            "brass": response_payload.get("brassUrl"),
            "woodwinds": response_payload.get("woodwindsUrl"),
        }

        stems: dict[str, AudioRef] = {}
        for key, url in url_map.items():
            if not url:
                continue
            buffer = await context.download_file(url)
            stems[key] = await context.audio_from_io(buffer)

        if not stems:
            raise ValueError("No stem URLs found in response")

        return stems


class ElevenLabsTextToSpeech(KieBaseNode):
    """Generate speech using ElevenLabs AI via Kie.ai.

    kie, elevenlabs, tts, text-to-speech, voice, audio, ai, speech synthesis

    Creates natural-sounding speech from text using ElevenLabs' voice models.
    Supports multiple voices, stability controls, and multilingual output.

    Use cases:
    - Generate voiceovers for videos and podcasts
    - Create audiobooks and narrated content
    - Produce natural-sounding speech for applications
    - Generate speech in multiple languages and voices
    """

    _expose_as_tool: ClassVar[bool] = True
    _poll_interval: float = 2.0
    _max_poll_attempts: int = 60

    text: str = Field(
        default="",
        description="The text to convert to speech.",
    )

    voice: str = Field(
        default="Rachel",
        description="The voice ID to use for synthesis. Common voices: Rachel, Adam, Bella, Antoni.",
    )

    stability: float = Field(
        default=0.5,
        description="Stability of the voice output. Lower values are more expressive, higher values are more consistent.",
        ge=0.0,
        le=1.0,
    )

    similarity_boost: float = Field(
        default=0.75,
        description="How closely to clone the voice characteristics. Higher values match the voice more closely.",
        ge=0.0,
        le=1.0,
    )

    style: float = Field(
        default=0.0,
        description="Style parameter for voice expression. Range 0.0 to 1.0.",
        ge=0.0,
        le=1.0,
    )

    speed: float = Field(
        default=1.0,
        description="Speed of the speech. Range 0.5 to 1.5.",
        ge=0.5,
        le=1.5,
    )

    language_code: str = Field(
        default="",
        description="Language code for multilingual TTS (e.g., 'en', 'es', 'fr', 'de'). Leave empty for auto-detection.",
    )

    class Model(str, Enum):
        TURBO_2_5 = "text-to-speech-turbo-2-5"
        MULTILINGUAL_V2 = "text-to-speech-multilingual-v2"

    model: Model = Field(
        default=Model.TURBO_2_5,
        description="ElevenLabs model version to use.",
    )

    def _get_model(self) -> str:
        return f"elevenlabs/{self.model.value}"

    @classmethod
    def get_title(cls) -> str:
        return "ElevenLabs Text To Speech"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if not self.text:
            raise ValueError("Text cannot be empty")
        payload: dict[str, Any] = {
            "text": self.text,
            "voice": self.voice,
            "stability": self.stability,
            "similarity_boost": self.similarity_boost,
            "style": self.style,
            "speed": self.speed,
        }
        if self.language_code:
            payload["language_code"] = self.language_code
        return payload

    async def process(self, context: ProcessingContext) -> AudioRef:
        audio_bytes, _ = await self._execute_task(context)
        return await context.audio_from_bytes(audio_bytes)
