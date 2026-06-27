"""Lightweight conversational intake.

The assistant answers questions on any topic, so there is no topic restriction.
The only intake behaviour is detecting when the user opens with a plain greeting
and nothing else, so the UI can reply with a friendly welcome instead of running
the full answer pipeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from loguru import logger

from prompts.guardrail_prompts import GREETING_RESPONSE_MESSAGE

_LATEST_MESSAGE_MARKER = "Latest user message:"

# Matches a message that is ONLY a conversational greeting with no other content.
# Anchored so "hi, where does my lease end?" is NOT matched and falls through to
# the normal answer pipeline.
_GREETING_PATTERN = re.compile(
    r"^(hi+|hello+|hey+|howdy|good\s+(morning|afternoon|evening|day)|"
    r"what'?s\s+up|greetings|hiya|yo|sup)[.!?,\s]*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TopicScopeResult:
    """Outcome of the conversational intake check."""

    on_topic: bool
    reason: str
    source: str


def extract_user_message_for_scope(problem_description: str) -> str:
    """Return the latest user turn when chat history is embedded in the prompt."""
    if _LATEST_MESSAGE_MARKER in problem_description:
        return problem_description.rsplit(_LATEST_MESSAGE_MARKER, maxsplit=1)[-1].strip()
    return problem_description.strip()


def classify_intake(problem_description: str) -> TopicScopeResult:
    """Classify a message as a plain greeting or a normal question.

    Every non-greeting message is always treated as in-scope, since the
    assistant answers questions on any topic. Greeting detection fires only on
    the first turn: a casual "hi" mid-thread is a continuation, not a fresh
    opener, so it is routed through the normal answer pipeline.
    """
    latest_normalized = extract_user_message_for_scope(problem_description).lower().strip()
    is_multi_turn = "Conversation so far:" in problem_description
    if not is_multi_turn and _GREETING_PATTERN.fullmatch(latest_normalized):
        logger.info("Greeting detected; returning welcome response without pipeline")
        return TopicScopeResult(
            on_topic=True,
            reason=GREETING_RESPONSE_MESSAGE,
            source="greeting",
        )
    return TopicScopeResult(on_topic=True, reason="", source="proceed")
