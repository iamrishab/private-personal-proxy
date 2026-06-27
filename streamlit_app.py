"""Streamlit chat UI for advisory triage."""

from __future__ import annotations

import asyncio
import os
import tempfile
import threading
import time
import uuid
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any, cast

import streamlit as st
from loguru import logger

os.environ.setdefault("ENV_FOR_DYNACONF", "streamlit")

from bootstrap import AppStack, create_app_stack
from config import Settings, get_settings
from logging_setup import configure_logging
from models.request import AdvisoryRequest
from models.response import AdvisoryResponse
from services.advisory import ServiceUnavailableError
from streamlit_chat import (
    ChatMessage,
    build_problem_description,
    format_advisory_response,
    format_response_details,
)
from streamlit_chat_history import (
    ChatHistoryStore,
    apply_snapshot_to_state,
    build_snapshot_from_state,
    format_saved_chat_label,
    sync_chat_from_saved_session,
)

WELCOME_MESSAGE = (
    "Ask me anything. I run entirely on your own computer, so your questions and "
    "documents stay private. Upload your documents in the sidebar and I'll answer "
    "questions about them with crisp next steps and cited sources."
)


class _BackgroundLoop:
    """A single, long-lived asyncio event loop running on a daemon thread.

    All async work (stack creation, advisory calls, streaming) is submitted to
    this one loop. The httpx/ollama clients cached inside the :class:`AppStack`
    bind to whichever loop first drives them; reusing them on a fresh per-call
    loop (Streamlit reruns each turn) leaves stale pooled connections from a now
    closed loop, which crashes uvloop with "the handler is closed". Pinning every
    call to this persistent loop keeps those connections valid for the whole
    process.
    """

    def __init__(self) -> None:
        """Start the background event loop on its own daemon thread."""
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="advisory-asyncio",
        )
        self._thread.start()

    def _run(self) -> None:
        """Bind the loop to this thread and run it until the process exits."""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def run(self, coro: Any) -> Any:
        """Run a coroutine to completion on the background loop and return it."""
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def iter_stream(
        self,
        agen: AsyncIterator[str | AdvisoryResponse],
    ) -> Iterator[str | AdvisoryResponse]:
        """Drive an async generator on the background loop as a sync iterator.

        Each ``__anext__`` is scheduled on the always-running loop, so the loop is
        never started/stopped between items and pooled connections stay alive.
        """
        while True:
            try:
                item = asyncio.run_coroutine_threadsafe(
                    agen.__anext__(),
                    self._loop,
                ).result()
            except StopAsyncIteration:
                return
            yield item


@st.cache_resource
def get_event_loop_runner() -> _BackgroundLoop:
    """Initialise and cache the process-wide background event loop."""
    return _BackgroundLoop()


@st.cache_resource
def get_stack() -> AppStack:
    """Initialize and cache the advisory service stack."""
    settings = get_settings()
    configure_logging(settings.logging_level)
    return cast(AppStack, get_event_loop_runner().run(create_app_stack(settings)))


def _init_session_state() -> None:
    """Ensure session-scoped chat and upload tracking keys exist."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "client_id" not in st.session_state:
        st.session_state.client_id = ""
    if "document_ids" not in st.session_state:
        st.session_state.document_ids = []
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "save_chat_locally" not in st.session_state:
        st.session_state.save_chat_locally = False
    # Holds the user message queued for LLM processing in the next script pass.
    if "pending_advisory" not in st.session_state:
        st.session_state.pending_advisory = None


def _sync_saved_session_context(settings: Settings) -> bool:
    """Restore locally saved chat context for the active session id."""
    store = _get_chat_history_store(settings)
    return sync_chat_from_saved_session(cast(dict[str, Any], st.session_state), store)


def _get_chat_history_store(settings: Settings) -> ChatHistoryStore:
    """Return the configured local chat history store."""
    return ChatHistoryStore(settings.streamlit_chat_history_dir)


def _save_chat_history(settings: Settings) -> Path | None:
    """Persist the current chat session to local disk."""
    messages = cast(list[ChatMessage], st.session_state.chat_messages)
    if not messages:
        return None
    store = _get_chat_history_store(settings)
    snapshot = build_snapshot_from_state(cast(dict[str, Any], st.session_state))
    try:
        return store.save(snapshot)
    except OSError as exc:
        # Disk full, permissions, bad path — never let a save failure crash a turn.
        logger.warning("Could not save chat history: {}", exc)
        return None


def _render_chat_history_controls(settings: Settings) -> None:
    """Render save, load, and delete controls for local chat history."""
    client_id = st.session_state.get("client_id") or None
    store = _get_chat_history_store(settings)
    saved_chats = store.list_saved(client_id=client_id)

    st.subheader("Chat history")
    st.checkbox(
        "Save chat history locally",
        key="save_chat_locally",
        help=(
            "When enabled, each completed chat turn is saved under "
            f"`{settings.streamlit_chat_history_dir}`."
        ),
    )
    st.caption(f"Storage: `{settings.streamlit_chat_history_dir}`")

    if st.button("Save chat now", disabled=not st.session_state.chat_messages):
        saved_path = _save_chat_history(settings)
        if saved_path is None:
            st.warning("Nothing to save yet.")
        else:
            st.success(f"Saved to `{saved_path.name}`.")

    if saved_chats:
        labels = {
            format_saved_chat_label(snapshot): snapshot.session_id
            for snapshot in saved_chats
        }
        selected_label = st.selectbox(
            "Saved chats",
            options=list(labels.keys()),
            key="saved_chat_selector",
        )
        selected_session_id = labels[selected_label]
        load_col, delete_col = st.columns(2)
        with load_col:
            if st.button("Load selected"):
                snapshot = store.load(selected_session_id, client_id=client_id)
                if snapshot is None:
                    st.error("Could not load the selected chat.")
                else:
                    apply_snapshot_to_state(
                        cast(dict[str, Any], st.session_state), snapshot
                    )
                    st.session_state._chat_synced_session_id = snapshot.session_id
                    st.success("Chat loaded.")
                    st.rerun()
        with delete_col:
            if st.button("Delete selected"):
                if store.delete(selected_session_id, client_id=client_id):
                    st.success("Saved chat deleted.")
                    st.rerun()
                else:
                    st.error("Could not delete the selected chat.")
    else:
        st.caption("No saved chats yet.")


def _render_sidebar(settings: Settings) -> None:
    """Render sidebar controls for the session, uploads, and chat history."""
    with st.sidebar:
        st.subheader("Session")
        st.text_input(
            "Client ID (optional)",
            key="client_id",
            help="Scope documents and chat history to a client identifier.",
        )
        # Display-only — changing session_id mid-session would break document retrieval.
        st.text_input("Session ID", key="session_id", disabled=True)

        uploaded = st.file_uploader(
            "Upload context (PDF, DOCX, TXT)",
            type=[ext.lstrip(".") for ext in sorted(settings.allowed_extensions)],
        )
        st.caption(
            "Uploaded documents are searched by the assistant to ground answers "
            "for this session.",
        )
        if uploaded is not None and st.button("Ingest upload"):
            stack = get_stack()
            suffix = Path(uploaded.name).suffix.lower()
            if suffix not in settings.allowed_extensions:
                st.error(f"Unsupported file type: {suffix}")
            elif len(uploaded.getvalue()) > settings.max_upload_bytes:
                st.error(f"File exceeds {settings.max_upload_bytes} bytes.")
            elif not uploaded.getvalue().strip():
                st.error("Uploaded file is empty.")
            else:
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(uploaded.getvalue())
                    temp_path = Path(tmp.name)
                try:
                    result = stack.ingestion_pipeline.ingest_upload(
                        temp_path,
                        filename=uploaded.name,
                        session_id=st.session_state.session_id,
                        client_id=st.session_state.client_id or None,
                    )
                    st.session_state.document_ids.append(result.document_id)
                    st.success(
                        f"Ingested {result.filename} ({result.chunk_count} chunks).",
                    )
                except ValueError as exc:
                    # Expected validation problem (bad/empty/unreadable file).
                    st.error(str(exc))
                except Exception as exc:  # noqa: BLE001 - keep the sidebar alive
                    logger.exception("Document ingest failed: {}", exc)
                    st.error(
                        "Could not ingest the document. Please try again or use a different file.",
                    )
                finally:
                    temp_path.unlink(missing_ok=True)

        if st.session_state.document_ids:
            st.write("Document IDs:", st.session_state.document_ids)

        _render_chat_history_controls(settings)

        if st.button("Clear chat"):
            st.session_state.chat_messages = []
            st.rerun()


def _render_chat_history(messages: list[ChatMessage]) -> None:
    """Render stored chat messages."""
    # Always show the welcome message as the assistant's opening turn so it stays
    # visible after the user sends their first message.
    with st.chat_message("assistant"):
        st.markdown(WELCOME_MESSAGE)

    for message in messages:
        with st.chat_message(message["role"]):
            # unsafe_allow_html required for the collapsible <details> metadata block.
            st.markdown(message["content"], unsafe_allow_html=True)


def _build_advisory_request(user_message: str) -> AdvisoryRequest | str:
    """Build the advisory request for a chat turn, or an error message to show."""
    # The latest user message is already persisted as the last entry, so prior
    # context is everything before it.
    prior_messages = cast(list[ChatMessage], list(st.session_state.chat_messages[:-1]))
    problem = build_problem_description(prior_messages, user_message)
    if len(problem.strip()) < 10:
        return "Please share at least 10 characters so I can help you properly."
    return AdvisoryRequest(
        problem_description=problem.strip(),
        session_id=str(st.session_state.session_id) or None,
        client_id=st.session_state.get("client_id") or None,
    )


def _stream_assistant_reply(stack: AppStack, request: AdvisoryRequest) -> str:
    """Stream the answer live, render the details, and return persisted Markdown.

    The underlying workflow runs to completion before the first token is
    emitted, so the spinner only covers that compute wait. Once tokens start
    arriving they are typed into a placeholder so the answer is visibly
    streamed; the final :class:`AdvisoryResponse` (captured from the stream) is
    then used to render the recommendations, any clarifying questions, cited
    sources, and elapsed time.
    """
    holder: dict[str, AdvisoryResponse] = {}
    # Live area the answer is typed into so streaming stays visible.
    answer_area = st.empty()
    streamed = ""

    agen = stack.advisory_service.advise_stream(request, trace_id=str(uuid.uuid4()))
    stream = get_event_loop_runner().iter_stream(agen)

    try:
        # The first item only arrives once the workflow has finished computing,
        # so keep the spinner up until then to signal the assistant is working.
        with st.spinner("Thinking…"):
            item = next(stream, None)

        # Type the answer out token-by-token for a visible streaming effect.
        while item is not None:
            if isinstance(item, AdvisoryResponse):
                holder["response"] = item
            else:
                streamed += item
                answer_area.markdown(streamed + "▌")
                time.sleep(0.01)
            item = next(stream, None)
    except ServiceUnavailableError as exc:
        retry_hint = (
            f" Try again in {exc.retry_after_seconds} seconds."
            if exc.retry_after_seconds
            else ""
        )
        reply = f"**Service unavailable:** {exc.message}{retry_hint}"
        answer_area.markdown(reply)
        return reply
    except Exception as exc:  # noqa: BLE001 - never crash the chat UI on a turn
        logger.exception("Unexpected error processing chat turn: {}", exc)
        reply = (
            "**Something went wrong** while analysing your request. "
            "Please try again in a moment, and rephrase if it keeps happening."
        )
        answer_area.markdown(reply)
        return reply

    # Re-render once more without the typing cursor.
    if streamed:
        answer_area.markdown(streamed)

    response = holder.get("response")
    if response is None:
        reply = (
            "**Something went wrong** while analysing your request. "
            "Please try again in a moment."
        )
        answer_area.markdown(reply)
        return reply

    # Greeting status streams no tokens — show its message directly.
    if not streamed:
        answer_area.markdown(response.message, unsafe_allow_html=True)
    else:
        details = format_response_details(response)
        if details:
            st.markdown(details, unsafe_allow_html=True)
    return format_advisory_response(response)


def main() -> None:
    """Render the Streamlit advisory chat application."""
    settings = get_settings()
    st.set_page_config(
        page_title=settings.streamlit_page_title,
        page_icon=settings.streamlit_page_icon,
        layout="wide",
    )
    _init_session_state()

    _render_sidebar(settings)
    restored = _sync_saved_session_context(settings)

    # Pre-warm the service stack so the first chat turn has no cold-start delay.
    # A startup failure (e.g. Ollama not running) is surfaced calmly instead of
    # crashing the whole page.
    try:
        get_stack()
    except Exception as exc:  # noqa: BLE001 - show a friendly banner, not a stack trace
        logger.exception("Advisory service stack failed to initialise: {}", exc)
        st.error(
            "The advisory service could not start. Please ensure the local Ollama "
            "server is running, then reload this page.",
        )
        return

    st.title(settings.streamlit_page_title)
    st.caption(
        "Private, local assistant. Ask anything — answers about your own documents "
        "are grounded in the files you upload, and nothing leaves your computer.",
    )
    if restored:
        st.info("Restored saved chat history for this session.")

    messages = cast(list[ChatMessage], st.session_state.chat_messages)
    _render_chat_history(messages)

    user_prompt = st.chat_input("Ask a question or describe what you need...")

    # Pass 1: user pressed Enter — persist the user message immediately and
    # re-render so the input clears, the user bubble shows, and the message is
    # retained even if the LLM call later fails.
    if user_prompt:
        st.session_state.chat_messages.append({"role": "user", "content": user_prompt})
        st.session_state.pending_advisory = user_prompt
        st.rerun()

    # Pass 2: process the queued message with the LLM.
    if st.session_state.pending_advisory:
        pending = st.session_state.pending_advisory
        # Clear before processing so a page reload does not re-trigger the call.
        st.session_state.pending_advisory = None

        stack = get_stack()

        with st.chat_message("assistant"):
            request = _build_advisory_request(pending)
            if isinstance(request, str):
                # Validation message (e.g. too short) — show it as-is.
                assistant_reply = request
                st.markdown(assistant_reply)
            else:
                assistant_reply = _stream_assistant_reply(stack, request)

        st.session_state.chat_messages.append(
            {"role": "assistant", "content": assistant_reply},
        )

        if st.session_state.get("save_chat_locally"):
            _save_chat_history(settings)


if __name__ == "__main__":
    main()
