"""Embeddings + cosine-similarity retrieval.

We embed LOs and chunks once at startup (or load cached embeddings from
disk) and keep them in memory as numpy arrays. At query time we embed
the user's text and do an in-memory cosine similarity search.

For 33 LOs and 105 chunks this is trivially fast — no need for a real
vector DB. The on-disk JSON cache means we only pay the embedding cost
once per data file.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
from openai import OpenAI

from app.config import get_settings
from app.schemas import Chunk, LearningOutcome


def _content_hash(items: list[str]) -> str:
    """Stable hash so cache invalidates when source text changes."""
    h = hashlib.sha256()
    for s in items:
        h.update(s.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


class EmbeddingStore:
    """Holds embeddings for LOs and chunks plus a tiny vector search.

    LO embeddings combine the domain, subdomain and outcome text so
    semantic matches pick up the curriculum context. Chunk embeddings
    just use the raw content.
    """

    def __init__(self, client: OpenAI | None = None):
        self.settings = get_settings()
        self.client = client or OpenAI(api_key=self.settings.openai_api_key)
        self.lo_vectors: np.ndarray | None = None
        self.chunk_vectors: np.ndarray | None = None
        self.lo_ids: list[str] = []
        self.chunk_ids: list[str] = []

    # ------------------------------------------------------------------
    # Building / caching
    # ------------------------------------------------------------------

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        """Call OpenAI embeddings API and return an (N, D) ndarray."""
        resp = self.client.embeddings.create(
            model=self.settings.openai_embed_model,
            input=texts,
        )
        return np.array([d.embedding for d in resp.data], dtype=np.float32)

    def _load_cache(self) -> dict | None:
        path = Path(self.settings.embeddings_cache_path)
        if not path.exists():
            return None
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    def _save_cache(self, payload: dict) -> None:
        path = Path(self.settings.embeddings_cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(payload, f)

    def build(self, los: list[LearningOutcome], chunks: list[Chunk]) -> None:
        """Embed LOs and chunks, using cache when content hasn't changed."""
        lo_inputs = [
            f"{lo.full_domain_label} — {lo.text}" for lo in los
        ]
        chunk_inputs = [c.content for c in chunks]

        lo_hash = _content_hash(lo_inputs)
        chunk_hash = _content_hash(chunk_inputs)

        cache = self._load_cache()
        if (
            cache
            and cache.get("lo_hash") == lo_hash
            and cache.get("chunk_hash") == chunk_hash
            and cache.get("model") == self.settings.openai_embed_model
        ):
            self.lo_vectors = np.array(cache["lo_vectors"], dtype=np.float32)
            self.chunk_vectors = np.array(cache["chunk_vectors"], dtype=np.float32)
            self.lo_ids = cache["lo_ids"]
            self.chunk_ids = cache["chunk_ids"]
            return

        # Cold path: call the API
        self.lo_vectors = self._embed_batch(lo_inputs)
        self.chunk_vectors = self._embed_batch(chunk_inputs)
        self.lo_ids = [lo.lo_id for lo in los]
        self.chunk_ids = [c.chunk_id for c in chunks]

        self._save_cache(
            {
                "model": self.settings.openai_embed_model,
                "lo_hash": lo_hash,
                "chunk_hash": chunk_hash,
                "lo_ids": self.lo_ids,
                "chunk_ids": self.chunk_ids,
                "lo_vectors": self.lo_vectors.tolist(),
                "chunk_vectors": self.chunk_vectors.tolist(),
            }
        )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def embed_query(self, text: str) -> np.ndarray:
        v = self._embed_batch([text])[0]
        return v

    @staticmethod
    def _cosine(matrix: np.ndarray, query: np.ndarray) -> np.ndarray:
        # Both inputs may not be unit-normalized
        m_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)
        q_norm = query / (np.linalg.norm(query) + 1e-9)
        return m_norm @ q_norm

    def top_los(self, query_vec: np.ndarray, k: int) -> list[tuple[str, float]]:
        """Return top-k (lo_id, score) by cosine similarity."""
        if self.lo_vectors is None:
            raise RuntimeError("EmbeddingStore not built")
        scores = self._cosine(self.lo_vectors, query_vec)
        top_idx = np.argsort(-scores)[:k]
        return [(self.lo_ids[i], float(scores[i])) for i in top_idx]

    def top_chunks(self, query_vec: np.ndarray, k: int) -> list[tuple[str, float]]:
        if self.chunk_vectors is None:
            raise RuntimeError("EmbeddingStore not built")
        scores = self._cosine(self.chunk_vectors, query_vec)
        top_idx = np.argsort(-scores)[:k]
        return [(self.chunk_ids[i], float(scores[i])) for i in top_idx]
