"""Document ingestion orchestration."""

from pathlib import Path

from config import Settings
from rag.chunker import chunk_documents
from rag.loaders import load_documents
from rag.store import IngestResult, VectorStoreManager


class DocumentIngestionPipeline:
    """Load, chunk, embed, and store uploaded documents."""

    def __init__(self, store: VectorStoreManager, settings: Settings | None = None) -> None:
        """Wire vector store manager and optional settings."""
        self._store = store
        self._settings = settings or store.settings

    def ingest_upload(
        self,
        file_path: Path,
        *,
        filename: str,
        session_id: str | None = None,
        client_id: str | None = None,
    ) -> IngestResult:
        """Ingest a user upload from a temporary file path.

        Raises ValueError if no text could be extracted (e.g. scanned PDF without OCR).
        """
        documents = load_documents(file_path)
        chunks = chunk_documents(
            documents,
            chunk_size=self._settings.rag_chunk_size,
            chunk_overlap=self._settings.rag_chunk_overlap,
        )
        if not chunks:
            raise ValueError(
                f"No text could be extracted from '{filename}'. "
                "Scanned PDFs require OCR pre-processing before upload."
            )
        return self._store.ingest_user_document(
            chunks,
            filename=filename,
            session_id=session_id,
            client_id=client_id,
        )
