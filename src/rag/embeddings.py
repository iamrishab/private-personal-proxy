"""Ollama embedding model factory for RAG.

Uses nomic-embed-text which requires task prefixes for accurate retrieval:
  - "search_document: " prefix for chunks stored in the vector database.
  - "search_query: "   prefix for user queries at retrieval time.
"""

from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings

from config import Settings


class _NomicEmbeddings(Embeddings):
    """Thin wrapper around OllamaEmbeddings that prepends the required task prefixes.

    nomic-embed-text v1.5 produces meaningfully better retrieval accuracy when
    the embeddings are created with the correct task instruction prefix.
    """

    def __init__(self, base_url: str, model: str) -> None:
        """Wire the underlying Ollama embeddings client."""
        self._embed = OllamaEmbeddings(model=model, base_url=base_url)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed document chunks with the search_document prefix."""
        prefixed = [f"search_document: {t}" for t in texts]
        return self._embed.embed_documents(prefixed)

    def embed_query(self, text: str) -> list[float]:
        """Embed a retrieval query with the search_query prefix."""
        return self._embed.embed_query(f"search_query: {text}")

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Async embed document chunks with the search_document prefix."""
        prefixed = [f"search_document: {t}" for t in texts]
        return await self._embed.aembed_documents(prefixed)

    async def aembed_query(self, text: str) -> list[float]:
        """Async embed a retrieval query with the search_query prefix."""
        return await self._embed.aembed_query(f"search_query: {text}")


def build_embeddings(settings: Settings) -> Embeddings:
    """Create an Ollama-backed embeddings client for vector search."""
    return _NomicEmbeddings(
        base_url=settings.ollama_base_url,
        model=settings.ollama_embedding_model,
    )
