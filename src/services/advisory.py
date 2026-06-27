"""Advisory orchestration via LangGraph workflow."""

import re
import time
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass

import httpx
import ollama
from loguru import logger

from config import Settings
from graph.advisory_graph import AdvisoryWorkflow
from models.request import AdvisoryRequest
from models.response import AdvisoryResponse
from services.circuit_breaker import CircuitBreaker


@dataclass
class ServiceUnavailableError(Exception):
    """Raised when the LLM provider cannot be reached."""

    retry_after_seconds: int
    message: str


class AdvisoryService:
    """Coordinates LangGraph workflow and provider resilience."""

    def __init__(
        self,
        settings: Settings,
        workflow: AdvisoryWorkflow,
        circuit_breaker: CircuitBreaker,
    ) -> None:
        """Wire dependencies for advisory processing."""
        self._settings = settings
        self._workflow = workflow
        self._circuit_breaker = circuit_breaker

    async def advise(
        self,
        request: AdvisoryRequest,
        *,
        trace_id: str | None = None,
    ) -> AdvisoryResponse:
        """Process a request and return a structured advisory response."""
        if not await self._circuit_breaker.is_available():
            raise ServiceUnavailableError(
                retry_after_seconds=await self._circuit_breaker.retry_after_seconds(),
                message="LLM provider temporarily unavailable due to repeated failures.",
            )
        start = time.perf_counter()
        try:
            result_state = await self._workflow.run(
                {
                    "request": request,
                    "trace_id": trace_id or "",
                    "retrieved_contexts": [],
                    "errors": [],
                },
            )
        except ollama.ResponseError as exc:
            await self._circuit_breaker.record_failure()
            logger.error("Ollama server error after retries: {}", exc)
            raise ServiceUnavailableError(
                retry_after_seconds=await self._circuit_breaker.retry_after_seconds(),
                message="LLM provider returned a server error.",
            ) from exc
        except (
            ConnectionError,
            TimeoutError,
            httpx.TimeoutException,
            httpx.ConnectError,
        ) as exc:
            await self._circuit_breaker.record_failure()
            logger.error("Ollama connection or timeout error after retries: {}", exc)
            raise ServiceUnavailableError(
                retry_after_seconds=await self._circuit_breaker.retry_after_seconds(),
                message=(
                    "Cannot reach the Ollama server. "
                    "Ensure Ollama is running at the configured base URL."
                ),
            ) from exc

        response = result_state.get("response")
        if response is None:
            await self._circuit_breaker.record_failure()
            raise ServiceUnavailableError(
                retry_after_seconds=await self._circuit_breaker.retry_after_seconds(),
                message="Advisory workflow did not produce a response.",
            )

        # A degraded response still means the provider responded (the model simply
        # produced unparseable output); the in-app fallback already handles it, so
        # it must NOT trip the circuit breaker and turn soft fallbacks into hard
        # 503s for every other caller. Only transport-level failures (handled in
        # the except blocks above) count against the breaker.
        await self._circuit_breaker.record_success()
        # Stamp the end-to-end wall-clock time so the UI/API can surface it.
        response.elapsed_seconds = round(time.perf_counter() - start, 2)
        return response

    async def advise_stream(
        self,
        request: AdvisoryRequest,
        *,
        trace_id: str | None = None,
    ) -> AsyncIterator[str | AdvisoryResponse]:
        """Stream the advisory answer word-by-word, then the full response.

        The workflow runs to completion first; the agentic tool loop and the
        structured synthesis pass are deliberately NOT streamed directly, because
        small local models can echo their prompt or emit tool-call noise in the
        middle of a generation. Only the finished, cleaned answer is streamed in
        word-sized chunks (for a live typing effect), followed by the assembled
        :class:`AdvisoryResponse` (grounding, elapsed time) as the last item.
        Transport-level failures raise :class:`ServiceUnavailableError`,
        mirroring :meth:`advise`.
        """
        if not await self._circuit_breaker.is_available():
            raise ServiceUnavailableError(
                retry_after_seconds=await self._circuit_breaker.retry_after_seconds(),
                message="LLM provider temporarily unavailable due to repeated failures.",
            )
        start = time.perf_counter()
        try:
            result_state = await self._workflow.run(
                {
                    "request": request,
                    "trace_id": trace_id or "",
                    "retrieved_contexts": [],
                    "errors": [],
                },
            )
        except ollama.ResponseError as exc:
            await self._circuit_breaker.record_failure()
            logger.error("Ollama server error during stream: {}", exc)
            raise ServiceUnavailableError(
                retry_after_seconds=await self._circuit_breaker.retry_after_seconds(),
                message="LLM provider returned a server error.",
            ) from exc
        except (
            ConnectionError,
            TimeoutError,
            httpx.TimeoutException,
            httpx.ConnectError,
        ) as exc:
            await self._circuit_breaker.record_failure()
            logger.error("Ollama connection or timeout error during stream: {}", exc)
            raise ServiceUnavailableError(
                retry_after_seconds=await self._circuit_breaker.retry_after_seconds(),
                message=(
                    "Cannot reach the Ollama server. "
                    "Ensure Ollama is running at the configured base URL."
                ),
            ) from exc

        response = result_state.get("response")
        if response is None:
            await self._circuit_breaker.record_failure()
            raise ServiceUnavailableError(
                retry_after_seconds=await self._circuit_breaker.retry_after_seconds(),
                message="Advisory workflow did not produce a response.",
            )
        await self._circuit_breaker.record_success()
        response.elapsed_seconds = round(time.perf_counter() - start, 2)
        # Stream the finished, already-cleaned answer in word-sized chunks so the
        # UI shows a live typing effect without any prompt echo or tool noise.
        answer = (response.answer or response.message).strip()
        for token in self._iter_answer_tokens(answer):
            yield token
        yield response

    @staticmethod
    def _iter_answer_tokens(answer: str) -> Iterator[str]:
        """Split an answer into word-sized chunks, preserving all whitespace.

        Keeping the original spacing and newlines intact lets the UI render the
        Markdown structure (headings, bullet lists) correctly while streaming.
        """
        for match in re.finditer(r"\S+\s*", answer):
            yield match.group(0)
