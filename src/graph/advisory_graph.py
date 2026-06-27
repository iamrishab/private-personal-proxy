"""LangGraph answer workflow.

A tool-calling agent (agentic RAG) answers the user's question, deciding when to
search the user's uploaded documents. The plain-language answer is then distilled
into a small structured record (next steps and any clarifying questions). A
non-tool fallback keeps the app working on weaker models.
"""

from __future__ import annotations

from typing import Any, Literal, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from loguru import logger

from config import Settings
from graph.state import AdvisoryState
from models.grounding import GroundingInfo, RetrievedSource
from models.llm_schema import LLMAdvisoryOutput
from models.request import AdvisoryRequest
from prompts.advisory_prompts import (
    AGENT_SYSTEM_PROMPT,
    AGENT_USER_PROMPT_TEMPLATE,
    FALLBACK_ANSWER_SYSTEM_PROMPT,
    RAG_CONTEXT_TEMPLATE,
)
from rag.retriever import AdvisoryRetriever
from services.llm_client import LLMClient, LLMResult
from services.post_processor import AdvisoryPostProcessor
from services.topic_guardrail import TopicScopeResult, classify_intake

# Tail line of AGENT_USER_PROMPT_TEMPLATE. A non-tool model that echoes its input
# reproduces the seeded prompt up to this marker before its real answer begins;
# detecting it lets us strip the echo defensively.
_PROMPT_ECHO_MARKER = "searching their uploaded documents where they add value."


class AdvisoryWorkflow:
    """LangGraph workflow: agentic RAG answer, synthesis, and post-processing."""

    def __init__(
        self,
        settings: Settings,
        llm_client: LLMClient,
        retriever: AdvisoryRetriever,
        post_processor: AdvisoryPostProcessor,
        chat_model: BaseChatModel,
        tools: list[BaseTool],
    ) -> None:
        """Wire workflow dependencies and bind the tool-calling agent model."""
        self._settings = settings
        self._llm_client = llm_client
        self._retriever = retriever
        self._post_processor = post_processor
        self._chat_model = chat_model
        self._tools = tools
        # The agent model is the chat model with the retrieval tools bound.
        self._agent_model: Runnable[list[AnyMessage], AIMessage] = (
            chat_model.bind_tools(
                tools,
            )
        )
        self._graph = self._build_graph()

    @property
    def graph(self) -> CompiledStateGraph[Any, Any, Any, Any]:
        """Return the compiled graph (used for streaming token output)."""
        return self._graph

    async def run(self, state: AdvisoryState) -> AdvisoryState:
        """Execute the compiled advisory graph.

        Binds the request trace id to the logging context so every node-level log
        line emitted during this run is correlated to the originating request.
        """
        trace_id = state.get("trace_id") or "-"
        with logger.contextualize(trace_id=trace_id):
            result = await self._graph.ainvoke(state)
        return cast(AdvisoryState, result)

    def _build_graph(self) -> CompiledStateGraph[Any, Any, Any, Any]:
        """Compile the advisory state graph."""
        graph = StateGraph(AdvisoryState)
        graph.add_node("check_scope", self._check_scope)
        graph.add_node("agent", self._agent)
        graph.add_node("tools", ToolNode(self._tools))
        graph.add_node("fallback_answer", self._fallback_answer)
        graph.add_node("synthesize", self._synthesize)
        graph.add_node("post_process", self._post_process)
        graph.add_node("degraded", self._degraded)
        graph.add_node("greeting_response", self._greeting_response)

        graph.add_edge(START, "check_scope")
        graph.add_conditional_edges(
            "check_scope",
            self._route_after_scope_check,
            {
                "agent": "agent",
                "greeting_response": "greeting_response",
            },
        )
        graph.add_conditional_edges(
            "agent",
            self._route_after_agent,
            {
                "tools": "tools",
                "synthesize": "synthesize",
                "fallback_answer": "fallback_answer",
            },
        )
        graph.add_edge("tools", "agent")
        graph.add_edge("fallback_answer", "synthesize")
        graph.add_conditional_edges(
            "synthesize",
            self._route_after_synthesize,
            {
                "post_process": "post_process",
                "degraded": "degraded",
            },
        )
        graph.add_edge("post_process", END)
        graph.add_edge("degraded", END)
        graph.add_edge("greeting_response", END)
        return graph.compile()

    # --- Conversational intake -------------------------------------------

    async def _check_scope(self, state: AdvisoryState) -> AdvisoryState:
        """Detect a plain greeting before the agent answers."""
        request = state["request"]
        scope_result = classify_intake(request.problem_description)
        return {"scope_result": scope_result}

    def _route_after_scope_check(
        self,
        state: AdvisoryState,
    ) -> Literal["agent", "greeting_response"]:
        """Route to the greeting handler for a plain greeting, else the agent."""
        scope_result = state.get("scope_result")
        if isinstance(scope_result, TopicScopeResult) and scope_result.source == "greeting":
            return "greeting_response"
        return "agent"

    async def _greeting_response(self, state: AdvisoryState) -> AdvisoryState:
        """Return a friendly welcome when the user opens with a pure greeting."""
        response = self._post_processor.build_greeting_response()
        logger.info("Greeting detected; returning welcome response")
        return {"response": response}

    # --- Agentic answer --------------------------------------------------

    async def _agent(self, state: AdvisoryState) -> AdvisoryState:
        """Run one turn of the tool-calling agent.

        Seeds the conversation with the system and problem messages on the first
        turn, then invokes the tool-bound model. The model either requests a tool
        call or produces the final grounded answer.
        """
        messages: list[AnyMessage] = list(state.get("messages") or [])
        seeded: list[AnyMessage] = []
        if not messages:
            seeded = [
                SystemMessage(content=AGENT_SYSTEM_PROMPT),
                HumanMessage(content=self._build_agent_user_prompt(state["request"])),
            ]
            messages = seeded
        response = await self._agent_model.ainvoke(messages)
        steps = state.get("agent_steps", 0) + 1
        # Append any seeded messages plus the model reply via the add_messages reducer.
        return {"messages": [*seeded, response], "agent_steps": steps}

    def _route_after_agent(
        self,
        state: AdvisoryState,
    ) -> Literal["tools", "synthesize", "fallback_answer"]:
        """Decide whether to run tools, synthesise the answer, or fall back."""
        messages = state.get("messages") or []
        last = messages[-1] if messages else None
        tool_calls = getattr(last, "tool_calls", None) if last is not None else None
        steps = state.get("agent_steps", 0)
        if tool_calls and steps < self._settings.agent_max_tool_iterations:
            return "tools"
        if self._message_text(last):
            return "synthesize"
        # No usable answer (e.g. only tool calls, or the model returned nothing).
        return "fallback_answer"

    async def _fallback_answer(self, state: AdvisoryState) -> AdvisoryState:
        """Generate a grounded answer without tools for weak/non-tool models."""
        request = state["request"]
        outcome = await self._retriever.retrieve(request)
        user_prompt = self._build_agent_user_prompt(request)
        if outcome.contexts:
            context_block = "\n\n".join(outcome.contexts)
            user_prompt = f"{user_prompt}\n\n{RAG_CONTEXT_TEMPLATE.format(context_block=context_block)}"
        reply = await self._chat_model.ainvoke(
            [
                SystemMessage(content=FALLBACK_ANSWER_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ],
        )
        logger.info("Used deterministic fallback to generate the advisory answer")
        return {
            "answer": self._clean_answer(self._message_text(reply)),
            "grounding": self._retriever.to_grounding(outcome),
            "grounding_sources": outcome.sources,
        }

    # --- Synthesis + routing --------------------------------------------

    async def _synthesize(self, state: AdvisoryState) -> AdvisoryState:
        """Turn the grounded answer into the structured analysis."""
        request = state["request"]
        answer = self._clean_answer(
            state.get("answer") or self._last_ai_text(state.get("messages") or []),
        )
        sources = state.get("grounding_sources") or []
        contexts = [self._format_source(source) for source in sources]
        llm_result = await self._llm_client.analyze(
            request,
            answer=answer,
            rag_contexts=contexts or None,
        )
        grounding = state.get("grounding") or self._build_grounding(sources)
        update = self._with_llm_result(state, llm_result)
        update["answer"] = answer
        update["grounding"] = grounding
        return update

    @staticmethod
    def _with_llm_result(
        state: AdvisoryState,
        llm_result: LLMResult,
    ) -> AdvisoryState:
        """Store an LLM result and accumulate any parse error onto state.errors."""
        update: AdvisoryState = {"llm_result": llm_result}
        if llm_result.error:
            errors = list(state.get("errors") or [])
            errors.append(f"synthesize: {llm_result.error}")
            update["errors"] = errors
        return update

    def _route_after_synthesize(
        self,
        state: AdvisoryState,
    ) -> Literal["post_process", "degraded"]:
        """Route to post-processing when the analysis parsed, else degrade."""
        llm_result = state.get("llm_result")
        if llm_result and isinstance(llm_result.output, LLMAdvisoryOutput):
            return "post_process"
        return "degraded"

    async def _post_process(self, state: AdvisoryState) -> AdvisoryState:
        """Apply deterministic routing rules to the synthesised analysis."""
        llm_result = state.get("llm_result")
        output = llm_result.output if llm_result is not None else None
        if not isinstance(output, LLMAdvisoryOutput):
            return await self._degraded(state)
        response = self._post_processor.post_process(
            output,
            grounding=state.get("grounding"),
            answer=state.get("answer") or "",
        )
        return {"response": response}

    async def _degraded(self, state: AdvisoryState) -> AdvisoryState:
        """Build a safe degraded response."""
        llm_result = state.get("llm_result")
        error = llm_result.error if isinstance(llm_result, LLMResult) else "unknown"
        response = self._post_processor.build_degraded_response(error)
        # Surface the answer text even when structured analysis failed.
        answer = self._clean_answer(
            state.get("answer") or self._last_ai_text(state.get("messages") or []),
        )
        if answer:
            response.answer = answer
        logger.warning("Returning degraded advisory response")
        return {"response": response}

    # --- Helpers ---------------------------------------------------------

    def _build_agent_user_prompt(self, request: AdvisoryRequest) -> str:
        """Format the first human turn for the agent from the request."""
        return AGENT_USER_PROMPT_TEMPLATE.format(
            problem_description=request.problem_description,
        )

    @staticmethod
    def _format_source(source: RetrievedSource) -> str:
        """Render a retrieved source as a grounding context line."""
        return f"[source: {source.filename} p.{source.page}] {source.snippet}"

    @staticmethod
    def _build_grounding(sources: list[RetrievedSource]) -> GroundingInfo | None:
        """Build grounding metadata from accumulated tool sources (deduplicated)."""
        if not sources:
            return None
        seen: set[tuple[str, object, str]] = set()
        unique: list[RetrievedSource] = []
        for source in sources:
            key = (source.filename, source.page, source.snippet)
            if key in seen:
                continue
            seen.add(key)
            unique.append(source)
        return GroundingInfo(sources=unique, retrieval_quality="good")

    @classmethod
    def _last_ai_text(cls, messages: list[AnyMessage]) -> str:
        """Return the text of the most recent AI message, if any."""
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                text = cls._message_text(message)
                if text:
                    return text
        return ""

    @staticmethod
    def _clean_answer(answer: str) -> str:
        """Strip any prompt a weak/non-tool model regurgitated before its answer.

        Conservative: only fires when the text literally begins with the system
        prompt's opening and still contains the user-prompt tail marker, so a
        legitimate answer can never be truncated by coincidence.
        """
        if not answer:
            return answer
        if (
            answer.lstrip().startswith(AGENT_SYSTEM_PROMPT[:30])
            and _PROMPT_ECHO_MARKER in answer
        ):
            return answer.rsplit(_PROMPT_ECHO_MARKER, 1)[1].strip()
        return answer

    @staticmethod
    def _message_text(message: AnyMessage | None) -> str:
        """Extract plain text content from a message."""
        if message is None:
            return ""
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content.strip()
        # Some providers return content as a list of parts; join text parts.
        if isinstance(content, list):
            parts = [
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            ]
            return "".join(parts).strip()
        return ""
