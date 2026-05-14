"""Singleton service that owns the curriculum, chunks, and embedding store.

The agent nodes call into this for everything data-related: looking up
LOs by id, grouping by subdomain for nice presentation, and running
semantic searches.
"""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache

from app.config import get_settings
from app.data_loader import load_chunks, load_learning_outcomes
from app.schemas import Chunk, LearningOutcome
from app.services.embeddings import EmbeddingStore


class CurriculumService:
    """Holds all curriculum data and the embedding index."""

    def __init__(self) -> None:
        settings = get_settings()
        self.los: list[LearningOutcome] = load_learning_outcomes(settings.lo_xlsx_path)
        self.chunks: list[Chunk] = load_chunks(settings.chunks_json_path)

        self._lo_by_id: dict[str, LearningOutcome] = {lo.lo_id: lo for lo in self.los}
        self._chunk_by_id: dict[str, Chunk] = {c.chunk_id: c for c in self.chunks}

        self.embeddings = EmbeddingStore()
        self.embeddings.build(self.los, self.chunks)

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def get_lo(self, lo_id: str) -> LearningOutcome | None:
        return self._lo_by_id.get(lo_id)

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        return self._chunk_by_id.get(chunk_id)

    # ------------------------------------------------------------------
    # Presentation helpers
    # ------------------------------------------------------------------

    def group_by_subdomain(self, lo_ids: list[str]) -> dict[str, list[LearningOutcome]]:
        """Group selected LOs under their domain/subdomain headers."""
        grouped: dict[str, list[LearningOutcome]] = defaultdict(list)
        for lo_id in lo_ids:
            lo = self.get_lo(lo_id)
            if lo:
                key = f"{lo.domain} — {lo.subdomain}"
                grouped[key].append(lo)
        return dict(grouped)

    def full_catalog_markdown(self) -> str:
        """Render the entire curriculum as a markdown menu.

        Used when the teacher gives a vague prompt and the agent wants
        to show the full landscape of what's available.
        """
        grouped: dict[str, dict[str, list[LearningOutcome]]] = defaultdict(lambda: defaultdict(list))
        for lo in self.los:
            grouped[lo.domain][lo.subdomain].append(lo)

        lines: list[str] = []
        for domain, subdomains in grouped.items():
            lines.append(f"### {domain}")
            for subdomain, los in subdomains.items():
                lines.append(f"**{subdomain}**")
                for lo in los:
                    lines.append(f"- `{lo.lo_id}` — {lo.text}")
                lines.append("")
        return "\n".join(lines).strip()

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def find_matching_los(self, intent_text: str, k: int | None = None) -> list[tuple[LearningOutcome, float]]:
        """Semantic match teacher intent → LOs."""
        settings = get_settings()
        k = k or settings.top_k_lo_match
        qv = self.embeddings.embed_query(intent_text)
        matches = self.embeddings.top_los(qv, k)
        return [(self._lo_by_id[lo_id], score) for lo_id, score in matches if lo_id in self._lo_by_id]

    def retrieve_chunks_for_lo(
        self,
        lo_id: str,
        k: int | None = None,
        extra_context: str = "",
    ) -> list[tuple[Chunk, float]]:
        """Retrieve top-k chunks for an LO.

        extra_context is appended to the query — this is the hook the
        refinement loop uses to bias retrieval based on teacher feedback
        like "I need more on real-world applications".
        """
        settings = get_settings()
        k = k or settings.top_k_chunk_retrieval
        lo = self.get_lo(lo_id)
        if not lo:
            return []
        query = f"{lo.full_domain_label} — {lo.text}"
        if extra_context:
            query = f"{query}\nAdditional focus: {extra_context}"
        qv = self.embeddings.embed_query(query)
        matches = self.embeddings.top_chunks(qv, k)
        return [(self._chunk_by_id[c_id], score) for c_id, score in matches if c_id in self._chunk_by_id]


@lru_cache
def get_curriculum_service() -> CurriculumService:
    """Cached singleton — loaded once per process."""
    return CurriculumService()
