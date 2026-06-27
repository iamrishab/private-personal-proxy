"""FastAPI application entrypoint and CLI runners."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from loguru import logger

from api.routes import router
from bootstrap import close_app_stack, create_app_stack
from config import get_settings
from logging_setup import configure_logging


def _is_binary_string_schema(node: Any) -> bool:
    """Return True when a schema node represents an uploaded binary file."""
    # Starlette's UploadFile renders as a binary-flagged string in OpenAPI 3.1.
    return (
        isinstance(node, dict)
        and node.get("type") == "string"
        and node.get("contentMediaType") == "application/octet-stream"
    )


def _fix_upload_schemas(node: Any) -> None:
    """Recursively normalize binary upload schemas for Swagger UI rendering, in place.

    Two problems are corrected so Swagger UI shows a real file-picker widget:

    1. Starlette's ``UploadFile`` emits ``{"type": "string",
       "contentMediaType": "application/octet-stream"}`` with no ``format`` keyword.
       Swagger UI only renders a file input when it sees ``format: binary``, so the
       marker is added here while leaving the valid ``contentMediaType`` intact.
    2. An optional ``list[UploadFile]`` parameter carries a ``default`` (e.g. ``[]``)
       on its array schema. Swagger UI falls back to a plain "add string item" editor
       whenever a ``default`` is present, hiding the file picker. The ``default`` is
       stripped from upload schemas; optionality is still honored because FastAPI keeps
       the field out of the request body's ``required`` list.
    """
    # Walk dictionaries: fix binary nodes and binary arrays, then recurse.
    if isinstance(node, dict):
        if _is_binary_string_schema(node):
            node["format"] = "binary"
            node.pop("default", None)
        # Drop the default on arrays whose items are uploaded files.
        if node.get("type") == "array" and _is_binary_string_schema(node.get("items")):
            node.pop("default", None)
        for value in node.values():
            _fix_upload_schemas(value)
    # Walk lists (e.g. anyOf / oneOf / allOf members) and recurse into each entry.
    elif isinstance(node, list):
        for item in node:
            _fix_upload_schemas(item)


def _build_openapi(app: FastAPI) -> dict[str, Any]:
    """Build the OpenAPI schema with binary upload fields marked for Swagger UI."""
    # Reuse the previously generated schema so repeated /openapi.json hits are cheap.
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title=app.title,
        version=app.version,
        openapi_version=app.openapi_version,
        description=app.description,
        routes=app.routes,
    )
    # Patch binary upload schemas so Swagger UI shows file pickers for uploads.
    _fix_upload_schemas(schema)
    app.openapi_schema = schema
    return schema


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize shared services on startup and release resources on shutdown."""
    stack = await create_app_stack()
    app.state.circuit_breaker = stack.circuit_breaker
    app.state.advisory_service = stack.advisory_service
    app.state.llm_client = stack.llm_client
    app.state.vector_store = stack.vector_store
    app.state.ingestion_pipeline = stack.ingestion_pipeline
    app.state.app_stack = stack
    yield
    await close_app_stack(stack)


def create_app() -> FastAPI:
    """Build and return the FastAPI application."""
    settings = get_settings()
    configure_logging(settings.logging_level)
    app = FastAPI(
        title="Private Docs Assistant",
        description=(
            "Local, private question-answering over your own documents. "
            "Runs entirely on a local Ollama model — no API keys, no external calls."
        ),
        version="0.2.0",
        lifespan=lifespan,
    )
    app.include_router(router)

    @app.exception_handler(Exception)
    async def _unhandled_exception_handler(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        """Convert any unhandled error into a structured 500 without leaking internals."""
        logger.exception(
            "Unhandled error on {} {}: {}", request.method, request.url.path, exc
        )
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "An internal error occurred while processing the request.",
            },
        )

    # Override schema generation so UploadFile fields render as Swagger UI file pickers.
    app.openapi = lambda: _build_openapi(app)  # type: ignore[method-assign]
    return app


app = create_app()


def run() -> None:
    """CLI entrypoint to start the uvicorn server."""
    settings = get_settings()
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )
