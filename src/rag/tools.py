"""LangChain retrieval tools for the advisory agent.

These wrap the vector store as tools the tool-calling agent can invoke on demand
(agentic RAG). Each tool returns a :class:`~langgraph.types.Command` that both
feeds the formatted passages back to the model and records the retrieved sources
into graph state so the final response can cite them.
"""

from __future__ import annotations

from typing import Annotated, Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from config import Settings
from models.grounding import RetrievedSource
from rag.store import VectorStoreManager


def _format_sources(sources: list[RetrievedSource]) -> str:
    """Render retrieved sources as a readable block for the model."""
    if not sources:
        return "No matching passages were found."
    blocks: list[str] = []
    for source in sources:
        location = f" p.{source.page}" if source.page is not None else ""
        blocks.append(f"[{source.filename}{location}] {source.snippet}")
    return "\n\n".join(blocks)


def build_retrieval_tools(
    store: VectorStoreManager,
    settings: Settings,
) -> list[BaseTool]:
    """Build the retrieval tool bound to a vector store and settings.

    A factory is used so the tool closes over the shared store instance while
    the agent supplies only the natural-language ``query``; document scoping is
    injected from graph state, never from the model.
    """
    k = settings.rag_retrieval_k
    threshold = settings.rag_similarity_threshold

    @tool("search_uploaded_documents")
    def search_uploaded_documents(
        query: str,
        state: Annotated[dict[str, Any], InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command[Any]:
        """Search the documents the user uploaded in this conversation (their own
        notes, contracts, statements, policies, reports, or manuals) for relevant
        passages."""
        request = state["request"]
        sources = store.search(
            query,
            k=k,
            threshold=threshold,
            client_id=request.client_id,
            session_id=request.session_id,
        )
        return Command(
            update={
                "grounding_sources": sources,
                "messages": [
                    ToolMessage(_format_sources(sources), tool_call_id=tool_call_id),
                ],
            },
        )

    return [search_uploaded_documents]
