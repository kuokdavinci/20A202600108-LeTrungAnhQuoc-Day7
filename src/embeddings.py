from __future__ import annotations

import hashlib
import math

LOCAL_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_PROVIDER_ENV = "EMBEDDING_PROVIDER"


class MockEmbedder:
    """Deterministic embedding backend used by tests and default classroom runs."""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim
        self._backend_name = "mock embeddings fallback"

    def __call__(self, text: str | list[str]) -> list[float] | list[list[float]]:
        if isinstance(text, str):
            return self._embed_single(text)
        return [self._embed_single(t) for t in text]

    def _embed_single(self, text: str) -> list[float]:
        digest = hashlib.md5(text.encode()).hexdigest()
        seed = int(digest, 16)
        vector = []
        for _ in range(self.dim):
            seed = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
            vector.append((seed / 0xFFFFFFFF) * 2 - 1)
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class LocalEmbedder:
    """Sentence Transformers-backed local embedder."""

    def __init__(self, model_name: str = LOCAL_EMBEDDING_MODEL) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._backend_name = model_name
        self.model = SentenceTransformer(model_name)

    def __call__(self, text: str | list[str]) -> list[float] | list[list[float]]:
        embeddings = self.model.encode(text, normalize_embeddings=True)
        if isinstance(text, list):
            return embeddings.tolist() if hasattr(embeddings, "tolist") else [list(map(float, e)) for e in embeddings]
        return embeddings.tolist() if hasattr(embeddings, "tolist") else [float(v) for v in embeddings]


class OpenAIEmbedder:
    """OpenAI embeddings API-backed embedder."""

    def __init__(self, model_name: str = OPENAI_EMBEDDING_MODEL, api_key: str | None = None) -> None:
        from openai import OpenAI
        import os

        self.model_name = model_name
        self._backend_name = model_name
        
        actual_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=actual_key)

    def __call__(self, text: str | list[str]) -> list[float] | list[list[float]]:
        response = self.client.embeddings.create(model=self.model_name, input=text)
        if isinstance(text, list):
            return [list(map(float, d.embedding)) for d in response.data]
        return [float(value) for value in response.data[0].embedding]


_mock_embed = MockEmbedder()
