"""Lightweight stub for HuggingFace embeddings used in tests.

The upstream `llama_index` project provides a rich embedding interface backed by
Hugging Face models.  The test-suite in this kata only needs the class to exist
and to return deterministic vectors so that downstream nodes can operate.  The
implementation below keeps the public surface compatible while avoiding heavy
third-party dependencies.
"""

from __future__ import annotations

from typing import Iterable

Vector = list[float]


class HuggingFaceEmbedding:
    """Minimal stand-in for `llama_index`'s embedding class.

    Parameters
    ----------
    model_name:
        Kept for API compatibility.  The parameter is accepted but unused.
    """

    def __init__(self, model_name: str | None = None, **_: object) -> None:
        self.model_name = model_name or "stub-model"

    # The real implementation exposes both synchronous and asynchronous
    # embedding helpers.  Returning deterministic zero vectors is sufficient for
    # the unit tests because they only verify structural behaviour.

    def embed_documents(self, texts: Iterable[str]) -> list[Vector]:
        return [self._vector_for(text) for text in texts]

    async def aembed_documents(self, texts: Iterable[str]) -> list[Vector]:
        return self.embed_documents(texts)

    def embed_query(self, text: str) -> Vector:
        return self._vector_for(text)

    async def aembed_query(self, text: str) -> Vector:
        return self.embed_query(text)

    @staticmethod
    def _vector_for(text: str, size: int = 768) -> Vector:
        # Generate a deterministic pseudo-vector using a simple hash-based
        # scheme.  The exact values are irrelevant; the goal is to return the
        # expected dimensionality and ensure calls are repeatable.
        seed = hash(text) & 0xFFFFFFFF
        return [((seed >> (i % 32)) & 1) * 1.0 for i in range(size)]

