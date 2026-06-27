"""FastAPI route definitions."""

import asyncio
import tempfile
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from loguru import logger

from config import Settings, get_settings
from models.enums import AdvisoryStatus
from models.request import AdvisoryRequest
from models.response import (
    AdvisoryResponse,
    DocumentIngestResponse,
    DocumentStatusResponse,
    HealthResponse,
    UnavailableResponse,
)
from rag.ingestion import DocumentIngestionPipeline
from services.advisory import AdvisoryService, ServiceUnavailableError

router = APIRouter()


def get_advisory_service(request: Request) -> AdvisoryService:
    """Resolve advisory service from application state."""
    service: AdvisoryService = request.app.state.advisory_service
    return service


def get_ingestion_pipeline(request: Request) -> DocumentIngestionPipeline:
    """Resolve document ingestion pipeline from application state."""
    pipeline: DocumentIngestionPipeline = request.app.state.ingestion_pipeline
    return pipeline


def get_app_settings() -> Settings:
    """Return application settings."""
    return get_settings()


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Return service health and LLM availability."""
    circuit_breaker = request.app.state.circuit_breaker
    return HealthResponse(
        status="ok",
        llm_available=await circuit_breaker.is_available(),
    )


@router.post(
    "/api/v1/advise",
    response_model=AdvisoryResponse,
    responses={503: {"model": UnavailableResponse}},
)
async def advise(
    problem_description: str = Form(
        ...,
        min_length=10,
        max_length=10000,
        description="Free-text question or the information the user is looking for.",
    ),  # noqa: E501
    session_id: str | None = Form(
        ...,
        max_length=200,
        description="Session id scoping uploaded documents; send empty when no session exists.",
    ),  # noqa: E501
    client_id: str | None = Form(
        ...,
        max_length=200,
        description="Optional identifier scoping documents and chat history; send empty when not applicable.",
    ),  # noqa: E501
    documents: list[UploadFile] = File(  # noqa: B008
        default=[],  # noqa: B006
        description="Optional documents to ingest and use as grounding context.",
    ),
    service: AdvisoryService = Depends(get_advisory_service),  # noqa: B008
    pipeline: DocumentIngestionPipeline = Depends(get_ingestion_pipeline),  # noqa: B008
    settings: Settings = Depends(get_app_settings),  # noqa: B008
) -> AdvisoryResponse | JSONResponse:
    """Answer the user's question and return a grounded response.

    Optionally accepts one or more documents that are ingested on-the-fly and
    used as grounding context for the current request.
    """
    request_id = str(uuid.uuid4())
    start = time.perf_counter()
    logger.bind(request_id=request_id).info(
        "Received question chars={} documents={}",
        len(problem_description),
        len(documents),
    )

    # Ingest any uploaded documents; carry the ingested session_id forward when
    # the caller provided an empty string (no pre-existing session).
    resolved_session = session_id
    for upload in documents:
        if upload.filename is None:
            continue
        suffix = Path(upload.filename).suffix.lower()
        if suffix not in settings.allowed_extensions:
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported file type '{suffix}'. Allowed: {', '.join(sorted(settings.allowed_extensions))}",
            )
        content = await upload.read()
        if len(content) > settings.max_upload_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File '{upload.filename}' exceeds maximum size of {settings.max_upload_bytes} bytes",
            )
        if not content.strip():
            raise HTTPException(
                status_code=422, detail=f"Uploaded file '{upload.filename}' is empty"
            )
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            temp_path = Path(tmp.name)
        try:
            # Ingestion (parse + embed + Chroma write) is synchronous and
            # IO/CPU-bound; run it in a worker thread so it does not block the
            # event loop and stall concurrent requests.
            result = await asyncio.to_thread(
                pipeline.ingest_upload,
                temp_path,
                filename=upload.filename,
                session_id=resolved_session,
                client_id=client_id,
            )
            # Carry the resolved session forward so all docs share the same session.
            resolved_session = result.session_id
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        finally:
            temp_path.unlink(missing_ok=True)

    payload = AdvisoryRequest(
        problem_description=problem_description,
        session_id=resolved_session,
        client_id=client_id,
    )

    try:
        result_response = await service.advise(payload, trace_id=request_id)
    except ServiceUnavailableError as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.bind(request_id=request_id).error(
            "Service unavailable after {:.0f}ms: {}",
            elapsed_ms,
            exc.message,
        )
        body = UnavailableResponse(
            status=AdvisoryStatus.UNAVAILABLE,
            message=exc.message,
            retry_after_seconds=exc.retry_after_seconds,
        )
        return JSONResponse(
            status_code=503,
            content=body.model_dump(mode="json"),
            headers={"Retry-After": str(exc.retry_after_seconds)},
        )

    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.bind(request_id=request_id).info(
        "Completed answer status={} latency_ms={:.0f}",
        result_response.status.value,
        elapsed_ms,
    )
    return result_response


async def _ingest_one(
    upload: UploadFile,
    *,
    session_id: str | None,
    client_id: str | None,
    pipeline: DocumentIngestionPipeline,
    settings: Settings,
) -> DocumentIngestResponse:
    """Validate, ingest, and return a response for a single uploaded file."""
    if upload.filename is None:
        raise HTTPException(status_code=422, detail="Filename is required")
    suffix = Path(upload.filename).suffix.lower()
    if suffix not in settings.allowed_extensions:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type '{suffix}'. Allowed: {', '.join(sorted(settings.allowed_extensions))}",
        )
    content = await upload.read()
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File '{upload.filename}' exceeds maximum size of {settings.max_upload_bytes} bytes",
        )
    if not content.strip():
        raise HTTPException(
            status_code=422, detail=f"Uploaded file '{upload.filename}' is empty"
        )
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        temp_path = Path(tmp.name)
    try:
        result = await asyncio.to_thread(
            pipeline.ingest_upload,
            temp_path,
            filename=upload.filename,
            session_id=session_id,
            client_id=client_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        temp_path.unlink(missing_ok=True)
    return DocumentIngestResponse(
        document_id=result.document_id,
        session_id=result.session_id,
        client_id=result.client_id or None,
        filename=result.filename,
        chunk_count=result.chunk_count,
        status="ready",
    )


@router.post("/api/v1/documents/ingest", response_model=list[DocumentIngestResponse])
async def ingest_documents(
    files: list[UploadFile] = File(...),  # noqa: B008
    session_id: str | None = None,
    client_id: str | None = None,
    pipeline: DocumentIngestionPipeline = Depends(get_ingestion_pipeline),  # noqa: B008
    settings: Settings = Depends(get_app_settings),  # noqa: B008
) -> list[DocumentIngestResponse]:
    """Upload and ingest one or more PDF, DOCX, or TXT documents for RAG retrieval.

    All files are scoped to the same client_id and session_id.
    """
    if not files:
        raise HTTPException(status_code=422, detail="At least one file is required")
    results: list[DocumentIngestResponse] = []
    # Carry the session forward across files so a batch uploaded without an
    # explicit session is grouped under one session instead of one per file.
    resolved_session = session_id
    for upload in files:
        result = await _ingest_one(
            upload,
            session_id=resolved_session,
            client_id=client_id,
            pipeline=pipeline,
            settings=settings,
        )
        resolved_session = result.session_id
        results.append(result)
    return results


@router.get(
    "/api/v1/documents/{document_id}/status",
    response_model=DocumentStatusResponse,
)
async def document_status(
    document_id: str,
    request: Request,
) -> DocumentStatusResponse:
    """Return ingestion status for an uploaded document."""
    registry = request.app.state.vector_store.registry
    status = registry.get_status(document_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Document not found")
    raw_client_id = str(status.get("client_id", ""))
    return DocumentStatusResponse(
        document_id=document_id,
        filename=str(status["filename"]),
        session_id=str(status["session_id"]),
        client_id=raw_client_id if raw_client_id else None,
        chunk_count=int(status["chunk_count"]),
        status=str(status["status"]),
    )
