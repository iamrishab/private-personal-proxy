"""Retrieval utilities with corrective RAG retry logic."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from loguru import logger

from config import Settings
from models.grounding import GroundingInfo, RetrievedSource
from models.request import AdvisoryRequest
from observability.privacy import redact_pii
from rag.store import VectorStoreManager


@dataclass
class RetrievalOutcome:
    """Result of a retrieval attempt."""

    sources: list[RetrievedSource]
    quality: str
    contexts: list[str]


class AdvisoryRetriever:
    """Retrieve grounding context from the user's uploaded documents."""

    def __init__(self, settings: Settings, store: VectorStoreManager) -> None:
        """Wire settings and vector store."""
        self._settings = settings
        self._store = store

    async def retrieve(self, request: AdvisoryRequest) -> RetrievalOutcome:
        """Retrieve relevant chunks with CRAG-style retry.

        The underlying Chroma similarity search and Ollama embedding calls are
        synchronous and CPU/IO-bound, so they run in a worker thread to avoid
        blocking the event loop. A vector-store failure degrades gracefully to a
        ``missing`` outcome rather than failing the whole advisory request.
        """
        query = redact_pii(request.problem_description)
        try:
            primary = await asyncio.to_thread(
                self._search,
                query=query,
                request=request,
                k=self._settings.rag_retrieval_k,
                threshold=self._settings.rag_similarity_threshold,
            )
            if primary.quality == "good":
                return primary
            logger.info("Weak retrieval detected; retrying with relaxed parameters")
            retry_k = (
                self._settings.rag_retrieval_k * self._settings.rag_crag_k_multiplier
            )
            retry_threshold = max(
                self._settings.rag_crag_min_threshold,
                self._settings.rag_similarity_threshold
                - self._settings.rag_crag_threshold_reduction,
            )
            retry = await asyncio.to_thread(
                self._search,
                query=query,
                request=request,
                k=retry_k,
                threshold=retry_threshold,
            )
        except Exception as exc:  # noqa: BLE001 - retrieval is best-effort grounding
            logger.exception("Retrieval failed; proceeding without grounding: {}", exc)
            return RetrievalOutcome(sources=[], quality="missing", contexts=[])

        if retry.quality == "good":
            return retry
        if retry.sources:
            return RetrievalOutcome(
                sources=retry.sources,
                quality="weak",
                contexts=retry.contexts,
            )
        return RetrievalOutcome(sources=[], quality="missing", contexts=[])

    def _search(
        self,
        *,
        query: str,
        request: AdvisoryRequest,
        k: int,
        threshold: float,
    ) -> RetrievalOutcome:
        """Search the user's uploaded documents scoped to their session/client.

        With no session or client scope there are no documents to search, so the
        outcome is ``missing`` and the answer falls back to general knowledge.
        """
        if not (request.client_id or request.session_id):
            return RetrievalOutcome(sources=[], quality="missing", contexts=[])

        sources = self._store.search(
            query,
            k=k,
            threshold=threshold,
            client_id=request.client_id,
            session_id=request.session_id,
        )
        sources.sort(key=lambda item: item.score, reverse=True)
        sources = sources[: k * 2]

        if not sources:
            return RetrievalOutcome(sources=[], quality="missing", contexts=[])

        max_score = max(source.score for source in sources)
        quality = "good" if max_score >= threshold else "weak"
        contexts = [
            f"[source: {source.filename} p.{source.page}] {source.snippet}"
            for source in sources
        ]
        return RetrievalOutcome(sources=sources, quality=quality, contexts=contexts)

    @staticmethod
    def to_grounding(outcome: RetrievalOutcome) -> GroundingInfo | None:
        """Convert retrieval outcome to API grounding payload."""
        if not outcome.sources and outcome.quality == "missing":
            return None
        return GroundingInfo(
            sources=outcome.sources,
            retrieval_quality=outcome.quality,
        )
