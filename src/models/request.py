"""Inbound API request models."""

from pydantic import BaseModel, Field


class AdvisoryRequest(BaseModel):
    """User question plus the document scope to search."""

    problem_description: str = Field(
        ...,
        min_length=10,
        max_length=10000,
        description="Free-text question or the information the user is looking for.",
    )
    session_id: str | None = Field(
        default=None,
        max_length=200,
        description="Session id scoping uploaded documents for retrieval.",
    )
    client_id: str | None = Field(
        default=None,
        max_length=200,
        description="Optional identifier scoping documents and chat history.",
    )
