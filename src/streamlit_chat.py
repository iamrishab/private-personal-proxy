"""Pure helpers for the Streamlit chat experience."""

from __future__ import annotations

import re
from typing import TypedDict

from models.enums import AdvisoryStatus
from models.grounding import GroundingInfo
from models.response import AdvisoryResponse
from prompts.streamlit_chat_prompts import (
    CHAT_HISTORY_LINE,
    CHAT_MULTI_TURN_TEMPLATE,
)

# Statuses that carry no streamed answer — the message is shown as-is.
_PLAIN_STATUSES = {
    AdvisoryStatus.GREETING,
}

# Matches the folded <details> metadata block appended to assistant messages.
_DETAILS_PATTERN = re.compile(r"<details>.*?</details>", re.DOTALL)


class ChatMessage(TypedDict):
    """One message in the Streamlit chat transcript."""

    role: str
    content: str


def _strip_html_details(content: str) -> str:
    """Remove the folded metadata HTML block before including a message in LLM history."""
    return _DETAILS_PATTERN.sub("", content).strip()


def truncate_history_content(content: str, *, max_chars: int = 1200) -> str:
    """Strip metadata HTML then trim long assistant replies for LLM context."""
    trimmed = _strip_html_details(content)
    if len(trimmed) <= max_chars:
        return trimmed
    return f"{trimmed[:max_chars].rstrip()}..."


def build_problem_description(
    messages: list[ChatMessage],
    latest_user_message: str,
    *,
    max_history_chars: int = 1200,
) -> str:
    """Combine prior chat turns with the latest user message into one prompt."""
    trimmed = latest_user_message.strip()
    history_lines = [
        CHAT_HISTORY_LINE.format(
            role=message["role"],
            content=(
                truncate_history_content(
                    message["content"], max_chars=max_history_chars
                )
                if message["role"] == "assistant"
                else message["content"].strip()
            ),
        )
        for message in messages
        if message["role"] in {"user", "assistant"}
    ]

    if not history_lines:
        return trimmed

    return CHAT_MULTI_TURN_TEMPLATE.format(
        history="\n".join(history_lines),
        latest_message=trimmed,
    )


def get_answer_text(response: AdvisoryResponse) -> str:
    """Return the streamable narrative answer, falling back to the routing message."""
    return (response.answer or response.message).strip()


def render_recommendations(recommendations: list[str]) -> str:
    """Render crisp recommendations as bold-led Markdown bullets."""
    if not recommendations:
        return ""
    bullets = "\n".join(f"- **{rec.strip().rstrip('.')}**" for rec in recommendations)
    return f"**Recommended next steps**\n\n{bullets}"


def render_clarifications(questions: list[str]) -> str:
    """Render any clarifying questions as a Markdown bullet list."""
    if not questions:
        return ""
    bullets = "\n".join(f"- {question.strip()}" for question in questions)
    return f"**To help you better, could you clarify:**\n\n{bullets}"


def render_sources(grounding: GroundingInfo | None) -> str:
    """Render cited document sources as a Markdown list."""
    if grounding is None or not grounding.sources:
        return ""
    seen: set[str] = set()
    lines: list[str] = []
    for source in grounding.sources:
        location = f" (p.{source.page})" if source.page is not None else ""
        label = f"{source.filename}{location}"
        if label in seen:
            continue
        seen.add(label)
        lines.append(f"- `{label}`")
    return "**Sources**\n\n" + "\n".join(lines)


def render_elapsed(elapsed_seconds: float) -> str:
    """Render the elapsed-time footer line."""
    if elapsed_seconds <= 0:
        return ""
    return f"*Answered in {elapsed_seconds:.1f}s*"


def format_response_details(response: AdvisoryResponse) -> str:
    """Render everything that follows the narrative answer.

    Includes the recommendation bullets, cited sources, and the elapsed-time
    footer. Used after the answer has been streamed so the rest of the reply can
    be appended in one block.
    """
    if response.status in _PLAIN_STATUSES:
        return ""
    # When the assistant needs clarification, its clarifying questions and the
    # recommendations overlap, so show only the questions to avoid duplication.
    next_steps = (
        ""
        if response.clarification_questions
        else render_recommendations(response.recommendations)
    )
    blocks = [
        next_steps,
        render_clarifications(response.clarification_questions),
        render_sources(response.grounding),
        render_elapsed(response.elapsed_seconds),
    ]
    rendered = [block for block in blocks if block]
    if not rendered:
        return ""
    return "\n\n---\n\n" + "\n\n".join(rendered)


def format_advisory_response(response: AdvisoryResponse) -> str:
    """Render a full reply as Markdown for history and non-streaming UI.

    Greeting status returns its message as-is. Every other status renders the
    grounded answer followed by recommendations, any clarifying questions, cited
    sources, and elapsed time.
    """
    if response.status in _PLAIN_STATUSES:
        return response.message
    answer = get_answer_text(response)
    return f"{answer}{format_response_details(response)}"
