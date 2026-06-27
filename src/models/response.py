"""Outbound API response models."""

from pydantic import BaseModel, Field

from models.enums import AdvisoryStatus
from models.grounding import GroundingInfo


class AdvisoryResponse(BaseModel):
    """Grounded answer returned to the client."""

    answer: str = Field(
        default="",
        description=(
            "The plain-language answer to the user's question, grounded in their "
            "uploaded documents when relevant."
        ),
    )
    recommendations: list[str] = Field(
        default_factory=list,
        description="Optional crisp, actionable next-step bullet points.",
    )
    clarification_questions: list[str] = Field(
        default_factory=list,
        description="Questions to ask when more detail is needed to answer well.",
    )
    status: AdvisoryStatus = Field(description="Processing outcome status.")
    grounding: GroundingInfo | None = Field(
        default=None,
        description="Cited document sources when the answer used uploaded documents.",
    )
    message: str = Field(
        default="",
        description=(
            "Pre-formatted conversational text shown directly to the user for "
            "greeting or degraded statuses that carry no streamed answer."
        ),
    )
    elapsed_seconds: float = Field(
        default=0.0,
        description="Wall-clock time taken to produce this response, in seconds.",
    )


class HealthResponse(BaseModel):
    """Health check payload."""

    status: str
    llm_available: bool


class UnavailableResponse(BaseModel):
    """Response when the local model provider is temporarily unavailable."""

    status: AdvisoryStatus
    message: str
    retry_after_seconds: int


class DocumentIngestResponse(BaseModel):
    """Response after uploading a document for retrieval."""

    document_id: str
    session_id: str
    client_id: str | None
    filename: str
    chunk_count: int
    status: str


class DocumentStatusResponse(BaseModel):
    """Status payload for an ingested document."""

    document_id: str
    filename: str
    session_id: str
    client_id: str | None
    chunk_count: int
    status: str
