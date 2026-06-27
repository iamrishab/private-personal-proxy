"""LangGraph state definitions for advisory workflow."""

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from models.grounding import GroundingInfo, RetrievedSource
from models.request import AdvisoryRequest
from models.response import AdvisoryResponse
from services.llm_client import LLMResult
from services.topic_guardrail import TopicScopeResult


class AdvisoryState(TypedDict, total=False):
    """Shared state passed between LangGraph nodes."""

    request: AdvisoryRequest
    trace_id: str
    # Conversation passed to the tool-calling agent; add_messages appends turns
    # (agent replies and tool results) as the agent loop runs.
    messages: Annotated[list[AnyMessage], add_messages]
    # Sources retrieved by the tools, accumulated across tool calls for citations.
    grounding_sources: Annotated[list[RetrievedSource], operator.add]
    # Number of agent turns taken, used to cap the tool-calling loop.
    agent_steps: int
    # The grounded free-text answer produced by the agent (or fallback).
    answer: str
    retrieved_contexts: list[str]
    grounding: GroundingInfo | None
    llm_result: LLMResult | None
    response: AdvisoryResponse | None
    scope_result: TopicScopeResult | None
    errors: list[str]
