"""Tests for _required_settings ClassVar on nodes that require API keys or secrets."""

import pytest
from typing import ClassVar

from nodetool.nodes.kie.image import KieBaseNode, GenerateImage
from nodetool.nodes.mistral.text import ChatComplete, CodeComplete
from nodetool.nodes.mistral.vision import ImageToText, OCR
from nodetool.nodes.mistral.embeddings import Embedding as MistralEmbedding
from nodetool.nodes.openai.agents import RealtimeAgent, RealtimeTranscription
from nodetool.nodes.openai.image import CreateImage, EditImage
from nodetool.nodes.openai.text import (
    Embedding as OpenAIEmbedding,
    WebSearch,
    Moderation,
)
from nodetool.nodes.openai.audio import (
    TextToSpeech as OpenAITTS,
    Translate,
    Transcribe as OpenAITranscribe,
)
from nodetool.nodes.anthropic.agents import ClaudeAgent
from nodetool.nodes.lib.mail import GmailSearch, MoveToArchive, AddLabel
from nodetool.nodes.lib.browser import BrowserUseNode
from nodetool.nodes.gemini.image import ImageGeneration
from nodetool.nodes.gemini.text import GroundedSearch, Embedding as GeminiEmbedding
from nodetool.nodes.gemini.video import TextToVideo, ImageToVideo
from nodetool.nodes.gemini.audio import TextToSpeech as GeminiTTS, Transcribe as GeminiTranscribe


class TestKieRequiredSettings:
    def test_kie_base_node_has_required_settings(self):
        assert KieBaseNode._required_settings == ["KIE_API_KEY"]

    def test_kie_base_node_required_settings_classmethod(self):
        assert KieBaseNode.required_settings() == ["KIE_API_KEY"]

    def test_kie_subclass_inherits_required_settings(self):
        assert GenerateImage._required_settings == ["KIE_API_KEY"]
        assert GenerateImage.required_settings() == ["KIE_API_KEY"]


class TestMistralRequiredSettings:
    def test_chat_complete(self):
        assert ChatComplete._required_settings == ["MISTRAL_API_KEY"]
        assert ChatComplete.required_settings() == ["MISTRAL_API_KEY"]

    def test_code_complete(self):
        assert CodeComplete._required_settings == ["MISTRAL_API_KEY"]
        assert CodeComplete.required_settings() == ["MISTRAL_API_KEY"]

    def test_image_to_text(self):
        assert ImageToText._required_settings == ["MISTRAL_API_KEY"]
        assert ImageToText.required_settings() == ["MISTRAL_API_KEY"]

    def test_ocr(self):
        assert OCR._required_settings == ["MISTRAL_API_KEY"]
        assert OCR.required_settings() == ["MISTRAL_API_KEY"]

    def test_embedding(self):
        assert MistralEmbedding._required_settings == ["MISTRAL_API_KEY"]
        assert MistralEmbedding.required_settings() == ["MISTRAL_API_KEY"]


class TestOpenAIRequiredSettings:
    def test_realtime_agent(self):
        assert RealtimeAgent._required_settings == ["OPENAI_API_KEY"]
        assert RealtimeAgent.required_settings() == ["OPENAI_API_KEY"]

    def test_realtime_transcription(self):
        assert RealtimeTranscription._required_settings == ["OPENAI_API_KEY"]
        assert RealtimeTranscription.required_settings() == ["OPENAI_API_KEY"]

    def test_create_image(self):
        assert CreateImage._required_settings == ["OPENAI_API_KEY"]
        assert CreateImage.required_settings() == ["OPENAI_API_KEY"]

    def test_edit_image(self):
        assert EditImage._required_settings == ["OPENAI_API_KEY"]
        assert EditImage.required_settings() == ["OPENAI_API_KEY"]

    def test_embedding(self):
        assert OpenAIEmbedding._required_settings == ["OPENAI_API_KEY"]
        assert OpenAIEmbedding.required_settings() == ["OPENAI_API_KEY"]

    def test_web_search(self):
        assert WebSearch._required_settings == ["OPENAI_API_KEY"]
        assert WebSearch.required_settings() == ["OPENAI_API_KEY"]

    def test_moderation(self):
        assert Moderation._required_settings == ["OPENAI_API_KEY"]
        assert Moderation.required_settings() == ["OPENAI_API_KEY"]

    def test_tts(self):
        assert OpenAITTS._required_settings == ["OPENAI_API_KEY"]
        assert OpenAITTS.required_settings() == ["OPENAI_API_KEY"]

    def test_translate(self):
        assert Translate._required_settings == ["OPENAI_API_KEY"]
        assert Translate.required_settings() == ["OPENAI_API_KEY"]

    def test_transcribe(self):
        assert OpenAITranscribe._required_settings == ["OPENAI_API_KEY"]
        assert OpenAITranscribe.required_settings() == ["OPENAI_API_KEY"]


class TestAnthropicRequiredSettings:
    def test_claude_agent(self):
        assert ClaudeAgent._required_settings == ["ANTHROPIC_API_KEY"]
        assert ClaudeAgent.required_settings() == ["ANTHROPIC_API_KEY"]


class TestGmailRequiredSettings:
    def test_gmail_search(self):
        assert GmailSearch._required_settings == ["GOOGLE_MAIL_USER", "GOOGLE_APP_PASSWORD"]
        assert GmailSearch.required_settings() == ["GOOGLE_MAIL_USER", "GOOGLE_APP_PASSWORD"]

    def test_move_to_archive(self):
        assert MoveToArchive._required_settings == ["GOOGLE_MAIL_USER", "GOOGLE_APP_PASSWORD"]
        assert MoveToArchive.required_settings() == ["GOOGLE_MAIL_USER", "GOOGLE_APP_PASSWORD"]

    def test_add_label(self):
        assert AddLabel._required_settings == ["GOOGLE_MAIL_USER", "GOOGLE_APP_PASSWORD"]
        assert AddLabel.required_settings() == ["GOOGLE_MAIL_USER", "GOOGLE_APP_PASSWORD"]


class TestBrowserRequiredSettings:
    def test_browser_use_node(self):
        assert BrowserUseNode._required_settings == ["OPENAI_API_KEY"]
        assert BrowserUseNode.required_settings() == ["OPENAI_API_KEY"]


class TestGeminiRequiredSettings:
    def test_image_generation(self):
        assert ImageGeneration._required_settings == ["GEMINI_API_KEY"]
        assert ImageGeneration.required_settings() == ["GEMINI_API_KEY"]

    def test_grounded_search(self):
        assert GroundedSearch._required_settings == ["GEMINI_API_KEY"]
        assert GroundedSearch.required_settings() == ["GEMINI_API_KEY"]

    def test_embedding(self):
        assert GeminiEmbedding._required_settings == ["GEMINI_API_KEY"]
        assert GeminiEmbedding.required_settings() == ["GEMINI_API_KEY"]

    def test_text_to_video(self):
        assert TextToVideo._required_settings == ["GEMINI_API_KEY"]
        assert TextToVideo.required_settings() == ["GEMINI_API_KEY"]

    def test_image_to_video(self):
        assert ImageToVideo._required_settings == ["GEMINI_API_KEY"]
        assert ImageToVideo.required_settings() == ["GEMINI_API_KEY"]

    def test_tts(self):
        assert GeminiTTS._required_settings == ["GEMINI_API_KEY"]
        assert GeminiTTS.required_settings() == ["GEMINI_API_KEY"]

    def test_transcribe(self):
        assert GeminiTranscribe._required_settings == ["GEMINI_API_KEY"]
        assert GeminiTranscribe.required_settings() == ["GEMINI_API_KEY"]
