"""Build the final response from the synthesised structured record."""

from loguru import logger

from models.enums import AdvisoryStatus
from models.grounding import GroundingInfo
from models.llm_schema import LLMAdvisoryOutput
from models.response import AdvisoryResponse
from prompts.guardrail_prompts import GREETING_RESPONSE_MESSAGE

# Default clarifying question used when the model flags that it needs more detail
# but does not supply its own questions.
_DEFAULT_CLARIFICATION = "Could you share a bit more detail about what you need?"


class AdvisoryPostProcessor:
    """Assemble the user-facing response from the structured synthesis record."""

    def post_process(
        self,
        output: LLMAdvisoryOutput,
        *,
        grounding: GroundingInfo | None = None,
        answer: str = "",
    ) -> AdvisoryResponse:
        """Assemble the response from the grounded answer and synthesis record.

        ``answer`` is the plain-language reply produced by the agent; it is
        attached so the UI can stream it. ``output`` carries the optional next
        steps and any clarifying questions.
        """
        clarification_questions = list(output.clarification_questions)
        if output.needs_clarification and not clarification_questions:
            clarification_questions = [_DEFAULT_CLARIFICATION]

        status = (
            AdvisoryStatus.NEEDS_CLARIFICATION
            if output.needs_clarification
            else AdvisoryStatus.SUCCESS
        )
        logger.info(
            "Built response status={} recommendations={} sources={}",
            status.value,
            len(output.recommendations),
            len(grounding.sources) if grounding else 0,
        )
        return AdvisoryResponse(
            answer=answer.strip(),
            recommendations=list(output.recommendations),
            clarification_questions=clarification_questions,
            status=status,
            grounding=grounding,
        )

    def build_degraded_response(
        self,
        error: str | None,
    ) -> AdvisoryResponse:
        """Return a safe fallback when structured synthesis cannot be parsed."""
        logger.warning("Building degraded response error={}", error or "unknown")
        return AdvisoryResponse(
            answer="",
            recommendations=[],
            clarification_questions=[],
            status=AdvisoryStatus.DEGRADED,
            message=(
                "I wasn't able to fully process your request. "
                "Could you try rephrasing it with a little more detail?"
            ),
        )

    def build_greeting_response(self) -> AdvisoryResponse:
        """Return a friendly welcome when the user opens with a plain greeting."""
        return AdvisoryResponse(
            answer="",
            recommendations=[],
            clarification_questions=[],
            status=AdvisoryStatus.GREETING,
            message=GREETING_RESPONSE_MESSAGE,
        )
