"""Enumerations for the assistant's processing status."""

from enum import StrEnum


class AdvisoryStatus(StrEnum):
    """Overall processing status returned to the client."""

    # A grounded answer was produced normally.
    SUCCESS = "success"
    # The assistant needs more detail before it can answer well.
    NEEDS_CLARIFICATION = "needs_clarification"
    # Structured post-processing failed; a best-effort answer is still returned.
    DEGRADED = "degraded"
    # The local model provider could not be reached.
    UNAVAILABLE = "unavailable"
    # The user opened with a plain greeting; no answer pipeline was needed.
    GREETING = "greeting"
