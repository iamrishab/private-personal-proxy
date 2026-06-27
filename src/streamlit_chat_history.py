"""Local persistence for Streamlit chat transcripts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from streamlit_chat import ChatMessage


class ChatHistorySnapshot(BaseModel):
    """Serialized chat session stored on disk."""

    session_id: str
    client_id: str | None = None
    saved_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    document_ids: list[str] = Field(default_factory=list)
    messages: list[ChatMessage] = Field(default_factory=list)


class ChatHistoryStore:
    """Read and write chat history JSON files in a local directory.

    When a snapshot has a client_id, files are stored under a per-client
    sub-directory: ``{history_dir}/{client_id}/{session_id}.json``.
    Snapshots without a client_id use the flat layout for backward compatibility:
    ``{history_dir}/{session_id}.json``.
    """

    def __init__(self, history_dir: Path) -> None:
        """Bind the store to a directory path."""
        self._history_dir = history_dir

    @property
    def history_dir(self) -> Path:
        """Return the configured chat history directory."""
        return self._history_dir

    def ensure_directory(self) -> None:
        """Create the chat history directory when it does not exist."""
        self._history_dir.mkdir(parents=True, exist_ok=True)

    def save(self, snapshot: ChatHistorySnapshot) -> Path:
        """Persist one chat snapshot keyed by client and session id."""
        path = self._path_for_session(snapshot.session_id, client_id=snapshot.client_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = snapshot.model_dump(mode="json")
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def load(self, session_id: str, *, client_id: str | None = None) -> ChatHistorySnapshot | None:
        """Load a saved chat snapshot for the given session and optional client id."""
        path = self._path_for_session(session_id, client_id=client_id)
        if not path.is_file():
            return None
        return self._read_snapshot_file(path)

    def list_saved(self, *, client_id: str | None = None) -> list[ChatHistorySnapshot]:
        """Return saved snapshots for the given client, sorted by most recent first.

        When client_id is supplied only that client's sub-directory is scanned.
        Without a client_id the flat top-level files are returned.
        """
        if not self._history_dir.is_dir():
            return []
        if client_id:
            safe_client = self._sanitize(client_id)
            client_dir = self._history_dir / safe_client
            files = sorted(client_dir.glob("*.json")) if client_dir.is_dir() else []
        else:
            # Flat files only — skips client sub-directories.
            files = sorted(f for f in self._history_dir.glob("*.json") if f.is_file())
        snapshots: list[ChatHistorySnapshot] = []
        for path in files:
            snapshot = self._read_snapshot_file(path)
            if snapshot is not None:
                snapshots.append(snapshot)
        snapshots.sort(key=lambda item: item.saved_at, reverse=True)
        return snapshots

    def delete(self, session_id: str, *, client_id: str | None = None) -> bool:
        """Delete a saved chat snapshot when it exists."""
        path = self._path_for_session(session_id, client_id=client_id)
        if not path.is_file():
            return False
        path.unlink()
        return True

    def _path_for_session(self, session_id: str, *, client_id: str | None = None) -> Path:
        """Build a safe file path for one session id, optionally namespaced by client."""
        safe_session = self._sanitize(session_id)
        if client_id:
            safe_client = self._sanitize(client_id)
            return self._history_dir / safe_client / f"{safe_session}.json"
        return self._history_dir / f"{safe_session}.json"

    @staticmethod
    def _sanitize(value: str) -> str:
        """Replace unsafe file-system characters with underscores."""
        return "".join(
            character if character.isalnum() or character in {"-", "_"} else "_"
            for character in value
        )

    def _read_snapshot_file(self, path: Path) -> ChatHistorySnapshot | None:
        """Parse one snapshot file and ignore invalid records."""
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(raw, dict):
            return None
        try:
            return ChatHistorySnapshot.model_validate(raw)
        except ValidationError:
            return None


def build_snapshot_from_state(state: dict[str, Any]) -> ChatHistorySnapshot:
    """Create a snapshot from Streamlit session state values."""
    return ChatHistorySnapshot(
        session_id=str(state["session_id"]),
        client_id=state.get("client_id") or None,
        document_ids=list(state.get("document_ids", [])),
        messages=list(state.get("chat_messages", [])),
    )


def sync_chat_from_saved_session(
    state: dict[str, Any],
    store: ChatHistoryStore,
) -> bool:
    """Load saved chat context when the session id changes or on first sync."""
    session_id = str(state.get("session_id", ""))
    client_id = state.get("client_id") or None
    previous_session_id = state.get("_chat_synced_session_id")
    if session_id == previous_session_id:
        return False

    snapshot = store.load(session_id, client_id=client_id)
    if snapshot is not None:
        apply_snapshot_to_state(state, snapshot)
    elif previous_session_id is not None:
        state["chat_messages"] = []
        state["document_ids"] = []

    state["_chat_synced_session_id"] = session_id
    return snapshot is not None


def apply_snapshot_to_state(state: dict[str, Any], snapshot: ChatHistorySnapshot) -> None:
    """Restore Streamlit session state fields from a saved snapshot."""
    state["session_id"] = snapshot.session_id
    state["client_id"] = snapshot.client_id or ""
    state["document_ids"] = list(snapshot.document_ids)
    state["chat_messages"] = list(snapshot.messages)


def format_saved_chat_label(snapshot: ChatHistorySnapshot) -> str:
    """Render a compact label for saved chat selectors."""
    saved_at = snapshot.saved_at.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")
    preview = next(
        (message["content"][:40] for message in snapshot.messages if message["role"] == "user"),
        "Empty chat",
    )
    client_prefix = f"[{snapshot.client_id}] " if snapshot.client_id else ""
    return f"{client_prefix}{snapshot.session_id[:8]}… · {saved_at} · {preview}"
