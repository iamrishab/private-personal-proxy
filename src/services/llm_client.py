"""Ollama async client wrapper with structured output, retries, and repair."""

from dataclasses import dataclass

import httpx
import ollama
from loguru import logger
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from config import Settings
from models.llm_schema import LLMAdvisoryOutput
from models.request import AdvisoryRequest
from prompts.advisory_prompts import (
    RAG_CONTEXT_TEMPLATE,
    REPAIR_SYSTEM_PROMPT,
    REPAIR_USER_PROMPT_TEMPLATE,
    SYNTHESIS_SYSTEM_PROMPT,
    SYNTHESIS_USER_PROMPT_TEMPLATE,
)


def _is_retryable_ollama_error(exc: BaseException) -> bool:
    """Return whether an Ollama error warrants a retry.

    Retries on transient server errors (503 overload) and network issues.
    Permanent errors such as model-not-found (404) are not retried.
    """
    if isinstance(exc, ollama.ResponseError):
        # 5xx errors are transient; 4xx (e.g. model not found) are permanent.
        return exc.status_code >= 500
    # Catch generic connection/timeout problems when Ollama is momentarily
    # unreachable, including the httpx-level errors raised by the ollama client.
    return isinstance(
        exc,
        (ConnectionError, TimeoutError, httpx.TimeoutException, httpx.ConnectError),
    )


@dataclass
class LLMResult:
    """Result of an LLM structured parse attempt."""

    output: LLMAdvisoryOutput | None
    refused: bool
    error: str | None = None


class LLMClient:
    """Calls the Ollama chat API with structured JSON output and retries."""

    def __init__(self, settings: Settings) -> None:
        """Create async Ollama client pointed at the configured server."""
        self._settings = settings
        # timeout bounds every underlying httpx call so a hung Ollama server
        # surfaces as a TimeoutError (retryable) instead of blocking forever.
        self._client = ollama.AsyncClient(
            host=settings.ollama_base_url,
            timeout=settings.ollama_request_timeout_seconds,
        )

    async def aclose(self) -> None:
        """No-op — ollama.AsyncClient has no persistent connection to close."""

    async def parse_primary(
        self,
        request: AdvisoryRequest,
        *,
        answer: str,
        rag_contexts: list[str] | None = None,
    ) -> LLMResult:
        """Synthesise the structured analysis from the advisor's answer."""
        user_content = self._build_synthesis_prompt(
            request, answer, rag_contexts=rag_contexts
        )
        return await self._call_structured(
            SYNTHESIS_SYSTEM_PROMPT, user_content, schema=LLMAdvisoryOutput
        )

    async def parse_repair(
        self,
        request: AdvisoryRequest,
        *,
        answer: str,
        rag_contexts: list[str] | None = None,
    ) -> LLMResult:
        """Repair the structured record when the primary parse fails.

        Grounding context is carried into the repair pass so the second attempt
        does not lose the retrieved evidence the primary attempt had access to.
        """
        repair_content = REPAIR_USER_PROMPT_TEMPLATE.format(
            problem_description=request.problem_description,
            answer=answer,
        )
        if rag_contexts:
            context_block = "\n\n".join(rag_contexts)
            repair_content = f"{repair_content}\n\n{RAG_CONTEXT_TEMPLATE.format(context_block=context_block)}"
        return await self._call_structured(
            REPAIR_SYSTEM_PROMPT, repair_content, schema=LLMAdvisoryOutput
        )

    async def analyze(
        self,
        request: AdvisoryRequest,
        *,
        answer: str,
        rag_contexts: list[str] | None = None,
    ) -> LLMResult:
        """Synthesise the structured record via primary then repair output."""
        primary = await self.parse_primary(
            request, answer=answer, rag_contexts=rag_contexts
        )
        if primary.output is not None:
            return primary
        logger.warning(
            "LLM primary synthesis returned no output (error={}); attempting repair",
            primary.error,
        )
        return await self.parse_repair(
            request, answer=answer, rag_contexts=rag_contexts
        )

    def _build_synthesis_prompt(
        self,
        request: AdvisoryRequest,
        answer: str,
        *,
        rag_contexts: list[str] | None = None,
    ) -> str:
        """Format the synthesis user prompt from the request, answer, and context."""
        base = SYNTHESIS_USER_PROMPT_TEMPLATE.format(
            problem_description=request.problem_description,
            answer=answer,
        )
        if not rag_contexts:
            return base
        context_block = "\n\n".join(rag_contexts)
        return f"{base}\n\n{RAG_CONTEXT_TEMPLATE.format(context_block=context_block)}"

    async def _call_structured(
        self,
        system_prompt: str,
        user_content: str,
        *,
        schema: type[LLMAdvisoryOutput],
    ) -> LLMResult:
        """Execute a structured JSON chat call with async retries.

        Ollama enforces the JSON schema via the `format` parameter.
        The model response is validated against the Pydantic schema before returning.
        Temperature is set to 0 for deterministic structured output.
        """
        retrying = AsyncRetrying(
            retry=retry_if_exception(_is_retryable_ollama_error),
            stop=stop_after_attempt(self._settings.llm_retry_max_attempts),
            wait=wait_exponential(
                multiplier=1,
                min=self._settings.llm_retry_min_wait_seconds,
                max=self._settings.llm_retry_max_wait_seconds,
            ),
            reraise=True,
        )
        async for attempt in retrying:
            with attempt:
                try:
                    response = await self._client.chat(
                        model=self._settings.ollama_model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_content},
                        ],
                        # Pass the Pydantic model JSON schema to enforce structured output.
                        format=schema.model_json_schema(),
                        options={"temperature": 0},
                    )
                except ollama.ResponseError as exc:
                    if exc.status_code < 500:
                        # Permanent error (e.g. model not pulled); return error immediately.
                        logger.error(
                            "Ollama permanent error status={} error={}",
                            exc.status_code,
                            exc.error,
                        )
                        return LLMResult(output=None, refused=False, error=str(exc))
                    logger.exception("Transient Ollama server error during chat")
                    raise
                except (
                    ConnectionError,
                    TimeoutError,
                    httpx.TimeoutException,
                    httpx.ConnectError,
                ):
                    logger.exception("Ollama connection or timeout error during chat")
                    raise
                except Exception as exc:
                    logger.exception("Unexpected error during Ollama chat call")
                    return LLMResult(output=None, refused=False, error=str(exc))

                raw_content = response.message.content or ""
                if not raw_content.strip():
                    return LLMResult(output=None, refused=False, error="empty_response")

                try:
                    # Validate the JSON response against the Pydantic schema.
                    parsed = schema.model_validate_json(raw_content)
                except Exception as exc:
                    logger.warning(
                        "Failed to parse structured output from model: {}",
                        exc,
                    )
                    return LLMResult(output=None, refused=False, error="parsed_none")

                return LLMResult(output=parsed, refused=False)

        return LLMResult(output=None, refused=False, error="retry_exhausted")
