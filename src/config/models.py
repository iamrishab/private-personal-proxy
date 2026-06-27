"""Typed application settings mapped from Dynaconf."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, field_validator

from config.dynaconf import dynaconf_settings


def _env_or_none(key: str) -> str | None:
    """Return a non-empty environment variable value, or None."""
    value = os.environ.get(key)
    if value is None or value.strip() == "":
        return None
    return value


def _first_str(*values: object) -> str | None:
    """Return the first non-empty string from the provided values."""
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


class Settings(BaseModel):
    """Flat runtime settings for the advisory service."""

    # Ollama server and model configuration.
    ollama_base_url: str
    ollama_model: str
    ollama_embedding_model: str
    ollama_request_timeout_seconds: float

    # Circuit breaker resilience settings.
    circuit_breaker_failure_threshold: int
    circuit_breaker_timeout_seconds: int

    # FastAPI server binding.
    host: str
    port: int

    # RAG vector store settings.
    chroma_persist_dir: Path
    rag_retrieval_k: int
    rag_similarity_threshold: float
    rag_chunk_size: int
    rag_chunk_overlap: int
    rag_user_docs_collection: str
    rag_snippet_max_chars: int
    rag_crag_k_multiplier: int
    rag_crag_threshold_reduction: float
    rag_crag_min_threshold: float

    # Document upload constraints.
    max_upload_bytes: int
    allowed_extensions: set[str]

    # LLM retry policy (tenacity).
    llm_retry_max_attempts: int
    llm_retry_min_wait_seconds: int
    llm_retry_max_wait_seconds: int

    # LangGraph node retry policies.
    graph_retrieve_max_attempts: int
    graph_retrieve_initial_interval: float
    graph_analyze_max_attempts: int
    graph_analyze_initial_interval: float

    # Tool-calling agent loop cap.
    agent_max_tool_iterations: int

    # Logging verbosity.
    logging_level: str

    # Streamlit UI settings.
    streamlit_page_title: str
    streamlit_page_icon: str
    streamlit_chat_history_dir: Path

    @field_validator("allowed_extensions", mode="before")
    @classmethod
    def _normalize_extensions(cls, value: object) -> set[str]:
        """Normalize extension lists to lowercase dotted suffixes."""
        if isinstance(value, set):
            items = value
        elif isinstance(value, (list, tuple)):
            items = set(value)
        else:
            msg = "allowed_extensions must be a list or set"
            raise TypeError(msg)
        return {item if item.startswith(".") else f".{item}" for item in items}

    @classmethod
    def from_dynaconf(cls) -> Settings:
        """Build validated settings from Dynaconf and environment overrides."""
        dynaconf_settings.reload()
        cfg = dynaconf_settings

        return cls(
            ollama_base_url=_first_str(
                _env_or_none("APP_OLLAMA__BASE_URL"),
                cfg.get("llm.base_url"),
            )
            or "http://localhost:11434",
            ollama_model=_first_str(
                _env_or_none("APP_LLM__MODEL"),
                cfg.get("llm.model"),
            )
            or "llama3.2:3b",
            ollama_embedding_model=_first_str(
                _env_or_none("APP_LLM__EMBEDDING_MODEL"),
                cfg.get("llm.embedding_model"),
            )
            or "nomic-embed-text:latest",
            ollama_request_timeout_seconds=float(
                _env_or_none("APP_LLM__REQUEST_TIMEOUT_SECONDS")
                or cfg.get("llm.request_timeout_seconds")
                or 60,
            ),
            circuit_breaker_failure_threshold=int(
                cfg.get("resilience.circuit_breaker.failure_threshold")
                if cfg.get("resilience.circuit_breaker.failure_threshold") is not None
                else _env_or_none("CIRCUIT_BREAKER_FAILURE_THRESHOLD") or 3
            ),
            circuit_breaker_timeout_seconds=int(
                cfg.get("resilience.circuit_breaker.timeout_seconds")
                if cfg.get("resilience.circuit_breaker.timeout_seconds") is not None
                else _env_or_none("CIRCUIT_BREAKER_TIMEOUT_SECONDS") or 60
            ),
            host=_first_str(cfg.get("app.host"), _env_or_none("HOST")) or "0.0.0.0",
            port=int(cfg.get("app.port") or _env_or_none("PORT") or 8000),
            chroma_persist_dir=Path(
                _first_str(
                    cfg.get("rag.chroma_persist_dir"),
                    _env_or_none("CHROMA_PERSIST_DIR"),
                )
                or "data/chroma",
            ),
            rag_retrieval_k=int(
                cfg.get("rag.retrieval_k") or _env_or_none("RAG_RETRIEVAL_K") or 5,
            ),
            rag_similarity_threshold=float(
                cfg.get("rag.similarity_threshold")
                or _env_or_none("RAG_SIMILARITY_THRESHOLD")
                or 0.5,
            ),
            rag_chunk_size=int(cfg.get("rag.chunk_size", 512)),
            rag_chunk_overlap=int(cfg.get("rag.chunk_overlap", 77)),
            rag_user_docs_collection=str(
                cfg.get("rag.user_docs_collection", "user_docs")
            ),
            rag_snippet_max_chars=int(cfg.get("rag.snippet_max_chars", 500)),
            rag_crag_k_multiplier=int(cfg.get("rag.crag.k_multiplier", 2)),
            rag_crag_threshold_reduction=float(
                cfg.get("rag.crag.threshold_reduction", 0.2),
            ),
            rag_crag_min_threshold=float(cfg.get("rag.crag.min_threshold", 0.2)),
            max_upload_bytes=int(
                _env_or_none("MAX_UPLOAD_BYTES")
                or cfg.get("advisory.max_upload_bytes", 10485760),
            ),
            allowed_extensions=cfg.get(
                "advisory.allowed_extensions", [".pdf", ".docx", ".txt"]
            ),
            llm_retry_max_attempts=int(cfg.get("llm.retry.max_attempts", 3)),
            llm_retry_min_wait_seconds=int(cfg.get("llm.retry.min_wait_seconds", 1)),
            llm_retry_max_wait_seconds=int(cfg.get("llm.retry.max_wait_seconds", 8)),
            graph_retrieve_max_attempts=int(
                cfg.get("resilience.graph_retry.retrieve.max_attempts", 2),
            ),
            graph_retrieve_initial_interval=float(
                cfg.get("resilience.graph_retry.retrieve.initial_interval", 0.5),
            ),
            graph_analyze_max_attempts=int(
                cfg.get("resilience.graph_retry.analyze.max_attempts", 3),
            ),
            graph_analyze_initial_interval=float(
                cfg.get("resilience.graph_retry.analyze.initial_interval", 1.0),
            ),
            agent_max_tool_iterations=int(cfg.get("agent.max_tool_iterations", 5)),
            logging_level=str(cfg.get("logging.level", "INFO")),
            streamlit_page_title=str(
                cfg.get("streamlit.page_title", "Private Docs Assistant")
            ),
            streamlit_page_icon=str(cfg.get("streamlit.page_icon", "🔒")),
            streamlit_chat_history_dir=Path(
                _first_str(
                    cfg.get("streamlit.chat_history_dir"),
                    _env_or_none("STREAMLIT_CHAT_HISTORY_DIR"),
                )
                or "data/chat_history",
            ),
        )


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings.from_dynaconf()
