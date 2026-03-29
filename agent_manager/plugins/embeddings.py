"""Optional embedding provider abstraction and adapters."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any


class BaseEmbeddingProvider(ABC):
    """Contract for turning text into dense vectors."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Return a dense vector for *text*."""

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts.  Default loops over ``embed``."""
        return [self.embed(t) for t in texts]


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """Embed text using an OpenAI-compatible ``/v1/embeddings`` endpoint.

    Works with the official ``openai`` SDK or any OpenAI-compatible
    local server (LM Studio, vLLM, etc.).

    Usage::

        from openai import OpenAI
        client = OpenAI()
        embedder = OpenAIEmbeddingProvider(client=client, model="text-embedding-3-small")
    """

    def __init__(
        self,
        *,
        client: Any,
        model: str = "text-embedding-3-small",
    ) -> None:
        self._client = client
        self._model = model

    def embed(self, text: str) -> list[float]:
        response = self._client.embeddings.create(input=[text], model=self._model)
        return list(response.data[0].embedding)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(input=texts, model=self._model)
        sorted_data = sorted(response.data, key=lambda d: d.index)
        return [list(d.embedding) for d in sorted_data]


class SentenceTransformerEmbeddingProvider(BaseEmbeddingProvider):
    """Embed text using a ``sentence-transformers`` model.

    Usage::

        embedder = SentenceTransformerEmbeddingProvider(
            model_name="all-MiniLM-L6-v2",
        )
    """

    def __init__(self, *, model_name: str = "all-MiniLM-L6-v2", device: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, device=device)

    def embed(self, text: str) -> list[float]:
        vector = self._model.encode(text, convert_to_numpy=True)
        return vector.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, convert_to_numpy=True)
        return vectors.tolist()


class CallableEmbeddingProvider(BaseEmbeddingProvider):
    """Thin wrapper around any ``Callable[[str], list[float]]`` function.

    Useful when you already have a custom embedding function and want to
    plug it into adapters that expect a ``BaseEmbeddingProvider``.

    Usage::

        provider = CallableEmbeddingProvider(fn=my_embed_fn)
    """

    def __init__(self, fn: Any) -> None:
        self._fn = fn

    def embed(self, text: str) -> list[float]:
        return list(self._fn(text))
