"""Chroma vector store wrapper for the user's uploaded document collection."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from loguru import logger

from config import Settings
from models.grounding import RetrievedSource
from rag.embeddings import build_embeddings


def build_user_filter(
    client_id: str | None,
    session_id: str | None,
) -> dict[str, object] | None:
    """Build a Chroma metadata filter scoping user uploads.

    Documents match when they belong to EITHER the active session OR the active
    client (OR, not AND), so a document uploaded under a session is still found
    after the user later sets a client id, and a client's documents stay visible
    across their sessions. None means no filtering.
    """
    conditions: list[dict[str, object]] = []
    if session_id:
        conditions.append({"session_id": session_id})
    if client_id:
        conditions.append({"client_id": client_id})
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$or": conditions}


@dataclass
class IngestResult:
    """Result of a document ingestion operation."""

    document_id: str
    session_id: str
    client_id: str
    filename: str
    chunk_count: int


class DocumentRegistry:
    """Track ingestion status for uploaded documents."""

    def __init__(self) -> None:
        """Initialize an empty in-memory registry."""
        self._status: dict[str, dict[str, str | int]] = {}

    def set_status(
        self,
        document_id: str,
        *,
        filename: str,
        session_id: str,
        client_id: str,
        chunk_count: int,
        status: str,
    ) -> None:
        """Store document ingestion status."""
        self._status[document_id] = {
            "filename": filename,
            "session_id": session_id,
            "client_id": client_id,
            "chunk_count": chunk_count,
            "status": status,
        }

    def get_status(self, document_id: str) -> dict[str, str | int] | None:
        """Return status for a document id."""
        return self._status.get(document_id)


class VectorStoreManager:
    """Manage the Chroma collection for the user's uploaded documents."""

    def __init__(
        self, settings: Settings, embeddings: Embeddings | None = None
    ) -> None:
        """Initialize the persistent Chroma collection."""
        self._settings = settings
        self._persist_dir = settings.chroma_persist_dir
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._embeddings = embeddings or build_embeddings(settings)
        self.registry = DocumentRegistry()
        # nomic-embed-text is trained for cosine similarity, so the collection is
        # configured with the cosine HNSW space. This makes LangChain's relevance
        # score "1 - cosine_distance" (a calibrated 0..1 value) instead of the
        # default squared-L2 mapping, which can fall outside 0..1 and silently
        # filter out valid matches at a fixed threshold.
        self._user_store = Chroma(
            collection_name=settings.rag_user_docs_collection,
            embedding_function=self._embeddings,
            persist_directory=str(self._persist_dir),
            collection_metadata={"hnsw:space": "cosine"},
        )

    @property
    def settings(self) -> Settings:
        """Return vector store settings."""
        return self._settings

    @property
    def user_store(self) -> Chroma:
        """Return the user uploads Chroma collection."""
        return self._user_store

    def search(
        self,
        query: str,
        *,
        k: int,
        threshold: float,
        client_id: str | None = None,
        session_id: str | None = None,
    ) -> list[RetrievedSource]:
        """Run a relevance-scored similarity search over the user's uploads.

        Results are scoped to the active session/client and any scoring below
        ``threshold`` are dropped. This is the shared retrieval primitive used by
        both the retrieval tool and the CRAG retriever.
        """
        results = self._user_store.similarity_search_with_relevance_scores(
            query,
            k=k,
            filter=build_user_filter(client_id, session_id),
        )
        sources: list[RetrievedSource] = []
        for doc, score in results:
            if score < threshold:
                continue
            metadata = doc.metadata
            sources.append(
                RetrievedSource(
                    document_id=str(metadata.get("document_id", "unknown")),
                    filename=str(metadata.get("filename", "unknown")),
                    page=metadata.get("page"),
                    source="user_upload",
                    snippet=doc.page_content[: self._settings.rag_snippet_max_chars],
                    score=score,
                ),
            )
        return sources

    def ingest_user_document(
        self,
        documents: list[Document],
        *,
        filename: str,
        session_id: str | None = None,
        client_id: str | None = None,
    ) -> IngestResult:
        """Ingest a user-uploaded document scoped to a client and session."""
        document_id = str(uuid.uuid4())
        resolved_session = session_id or str(uuid.uuid4())
        resolved_client = client_id or ""
        for doc in documents:
            metadata = doc.metadata.copy()
            metadata["source"] = "user_upload"
            metadata["document_id"] = document_id
            metadata["filename"] = filename
            metadata["session_id"] = resolved_session
            metadata["client_id"] = resolved_client
            doc.metadata = metadata
        self._user_store.add_documents(documents)
        self.registry.set_status(
            document_id,
            filename=filename,
            session_id=resolved_session,
            client_id=resolved_client,
            chunk_count=len(documents),
            status="ready",
        )
        logger.info(
            "Ingested user document id={} client={} session={} chunks={}",
            document_id,
            resolved_client or "(none)",
            resolved_session,
            len(documents),
        )
        return IngestResult(
            document_id=document_id,
            session_id=resolved_session,
            client_id=resolved_client,
            filename=filename,
            chunk_count=len(documents),
        )
