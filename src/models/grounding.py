"""Grounding metadata returned with answers."""

from pydantic import BaseModel, Field


class RetrievedSource(BaseModel):
    """A single retrieved document chunk used for grounding."""

    document_id: str = Field(description="Source document identifier.")
    filename: str = Field(description="Original filename.")
    page: int | None = Field(default=None, description="Page number when available.")
    source: str = Field(description="Origin of the chunk (always user_upload).")
    snippet: str = Field(description="Retrieved text snippet.")
    score: float = Field(description="Similarity score for the retrieved chunk.")


class GroundingInfo(BaseModel):
    """Grounding metadata attached to an answer."""

    sources: list[RetrievedSource] = Field(
        default_factory=list,
        description="Retrieved chunks used to ground the answer.",
    )
    retrieval_quality: str = Field(
        default="none",
        description="none, good, weak, or missing.",
    )
