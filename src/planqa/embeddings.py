from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Iterable, Protocol

import numpy as np
import requests

from .config import AppConfig


class EmbeddingProvider(Protocol):
    signature: str

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        ...


def normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix.astype(np.float32)
    matrix = matrix.astype(np.float32, copy=False)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


@dataclass
class LocalHashEmbeddingProvider:
    dims: int = 768
    signature: str = "local-hash:v1:768"

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        vectors = [self._embed_one(text) for text in texts]
        return normalize_matrix(np.vstack(vectors))

    def _embed_one(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dims, dtype=np.float32)
        compact = re.sub(r"\s+", "", text.lower())
        tokens = re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", text.lower())

        for token in tokens:
            self._add_feature(vector, token, 2.5)

        for n, weight in ((1, 0.6), (2, 1.0), (3, 1.2)):
            if len(compact) < n:
                continue
            for idx in range(0, len(compact) - n + 1):
                self._add_feature(vector, compact[idx : idx + n], weight)

        norm = math.sqrt(float(np.dot(vector, vector)))
        if norm > 0:
            vector /= norm
        return vector

    def _add_feature(self, vector: np.ndarray, feature: str, weight: float) -> None:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        raw = int.from_bytes(digest, "little", signed=False)
        index = raw % self.dims
        sign = 1.0 if (raw >> 63) == 0 else -1.0
        vector[index] += sign * weight


@dataclass
class OpenAIEmbeddingProvider:
    config: AppConfig

    @property
    def signature(self) -> str:
        return f"openai-compatible:{self.config.base_url}:{self.config.embedding_model}"

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        for batch in _batches(texts, self.config.embedding_batch_size):
            vectors.extend(self._embed_batch(batch))
        return normalize_matrix(np.asarray(vectors, dtype=np.float32))

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        url = f"{self.config.base_url}/embeddings"
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            json={"model": self.config.embedding_model, "input": texts},
            timeout=90,
        )
        if response.status_code >= 400:
            body = response.text[:800]
            raise RuntimeError(f"Embedding API failed: HTTP {response.status_code}: {body}")
        payload = response.json()
        data = sorted(payload.get("data", []), key=lambda item: item.get("index", 0))
        if len(data) != len(texts):
            raise RuntimeError("Embedding API returned an unexpected number of vectors.")
        return [item["embedding"] for item in data]


def make_embedding_provider(config: AppConfig) -> EmbeddingProvider:
    if config.has_api_key and not config.uses_local_embeddings:
        return OpenAIEmbeddingProvider(config)
    return LocalHashEmbeddingProvider()


def _batches(items: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), max(size, 1)):
        yield items[start : start + size]
