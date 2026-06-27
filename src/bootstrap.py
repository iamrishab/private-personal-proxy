"""Shared application service wiring for FastAPI and Streamlit."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_ollama import ChatOllama
from loguru import logger

from config import Settings, get_settings
from graph.advisory_graph import AdvisoryWorkflow
from rag.ingestion import DocumentIngestionPipeline
from rag.retriever import AdvisoryRetriever
from rag.store import VectorStoreManager
from rag.tools import build_retrieval_tools
from services.advisory import AdvisoryService
from services.circuit_breaker import CircuitBreaker
from services.llm_client import LLMClient
from services.post_processor import AdvisoryPostProcessor

# Model families that cannot drive Ollama tool calling. Binding tools to these
# makes the agent regurgitate (echo) its prompt instead of answering, so we warn
# loudly at startup rather than fail silently.
_NON_TOOL_MODEL_PREFIXES = ("gemma", "phi", "tinyllama", "orca", "vicuna")


@dataclass
class AppStack:
    """Shared runtime services for advisory processing."""

    settings: Settings
    advisory_service: AdvisoryService
    ingestion_pipeline: DocumentIngestionPipeline
    llm_client: LLMClient
    vector_store: VectorStoreManager
    circuit_breaker: CircuitBreaker


async def create_app_stack(settings: Settings | None = None) -> AppStack:
    """Initialize shared services used by API and Streamlit entrypoints."""
    resolved = settings or get_settings()
    circuit_breaker = CircuitBreaker(
        failure_threshold=resolved.circuit_breaker_failure_threshold,
        timeout_seconds=resolved.circuit_breaker_timeout_seconds,
    )
    llm_client = LLMClient(resolved)
    vector_store = VectorStoreManager(resolved)
    ingestion_pipeline = DocumentIngestionPipeline(vector_store, settings=resolved)
    retriever = AdvisoryRetriever(resolved, vector_store)
    post_processor = AdvisoryPostProcessor()
    # Surface a misconfigured model early: non-tool models cause the agent to
    # echo its prompt and feel slow, which is hard to diagnose from the UI alone.
    if (
        resolved.ollama_model.split(":", 1)[0]
        .lower()
        .startswith(_NON_TOOL_MODEL_PREFIXES)
    ):
        logger.warning(
            "Configured chat model '{}' does not support tool calling; the agent "
            "may echo its prompt and respond slowly. Use a tool-capable model such "
            "as llama3.2:3b (set APP_LLM__MODEL and restart).",
            resolved.ollama_model,
        )
    # Tool-calling chat model for the agentic-RAG flow; streams tokens for the UI
    # and supports tool binding for the retrieval tools. Temperature 0 keeps tool
    # selection and routing deterministic; num_predict caps answer length so
    # replies stay concise and fast.
    chat_model = ChatOllama(
        model=resolved.ollama_model,
        base_url=resolved.ollama_base_url,
        temperature=0,
        num_predict=512,
    )
    tools = build_retrieval_tools(vector_store, resolved)
    workflow = AdvisoryWorkflow(
        resolved,
        llm_client,
        retriever,
        post_processor,
        chat_model,
        tools,
    )
    advisory_service = AdvisoryService(resolved, workflow, circuit_breaker)
    return AppStack(
        settings=resolved,
        advisory_service=advisory_service,
        ingestion_pipeline=ingestion_pipeline,
        llm_client=llm_client,
        vector_store=vector_store,
        circuit_breaker=circuit_breaker,
    )


async def close_app_stack(stack: AppStack) -> None:
    """Release resources held by a service stack."""
    await stack.llm_client.aclose()
