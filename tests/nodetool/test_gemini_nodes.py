"""
Tests for Gemini nodes (Transcribe, Embedding)
These tests focus on node field validation and basic instantiation.
"""

import pytest


class TestTranscribeNode:
    """Tests for the Gemini Transcribe node."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Import node classes only if nodetool-core is available."""
        try:
            from nodetool.nodes.gemini.audio import Transcribe, TranscriptionModel
            from nodetool.metadata.types import AudioRef
            self.Transcribe = Transcribe
            self.TranscriptionModel = TranscriptionModel
            self.AudioRef = AudioRef
            self.skip = False
        except ImportError:
            self.skip = True

    def test_default_values(self):
        """Test default field values."""
        if self.skip:
            pytest.skip("nodetool-core not available")
        node = self.Transcribe()
        assert node.model == self.TranscriptionModel.GEMINI_2_5_FLASH
        assert node.audio == self.AudioRef()
        assert "Transcribe" in node.prompt

    def test_custom_model(self):
        """Test setting custom model."""
        if self.skip:
            pytest.skip("nodetool-core not available")
        node = self.Transcribe(model=self.TranscriptionModel.GEMINI_2_5_PRO)
        assert node.model == self.TranscriptionModel.GEMINI_2_5_PRO

    def test_custom_prompt(self):
        """Test setting custom prompt."""
        if self.skip:
            pytest.skip("nodetool-core not available")
        custom_prompt = "Please transcribe this audio in English."
        node = self.Transcribe(prompt=custom_prompt)
        assert node.prompt == custom_prompt

    def test_model_enum_values(self):
        """Test that all transcription model enum values are valid strings."""
        if self.skip:
            pytest.skip("nodetool-core not available")
        for model in self.TranscriptionModel:
            assert isinstance(model.value, str)
            assert "gemini" in model.value.lower()


class TestEmbeddingNode:
    """Tests for the Gemini Embedding node."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Import node classes only if nodetool-core is available."""
        try:
            from nodetool.nodes.gemini.text import Embedding, EmbeddingModel
            self.Embedding = Embedding
            self.EmbeddingModel = EmbeddingModel
            self.skip = False
        except ImportError:
            self.skip = True

    def test_default_values(self):
        """Test default field values."""
        if self.skip:
            pytest.skip("nodetool-core not available")
        node = self.Embedding()
        assert node.model == self.EmbeddingModel.TEXT_EMBEDDING_004
        assert node.input == ""

    def test_custom_model(self):
        """Test setting custom model."""
        if self.skip:
            pytest.skip("nodetool-core not available")
        node = self.Embedding(model=self.EmbeddingModel.TEXT_EMBEDDING_005)
        assert node.model == self.EmbeddingModel.TEXT_EMBEDDING_005

    def test_custom_input(self):
        """Test setting custom input."""
        if self.skip:
            pytest.skip("nodetool-core not available")
        node = self.Embedding(input="Hello, world!")
        assert node.input == "Hello, world!"

    def test_model_enum_values(self):
        """Test that all embedding model enum values are valid strings."""
        if self.skip:
            pytest.skip("nodetool-core not available")
        for model in self.EmbeddingModel:
            assert isinstance(model.value, str)
            assert "embedding" in model.value.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
