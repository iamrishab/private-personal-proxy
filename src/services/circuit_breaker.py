"""Circuit breaker for LLM provider failures."""

import asyncio
import time
from enum import StrEnum

from loguru import logger


class CircuitState(StrEnum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Tracks consecutive LLM failures and blocks calls when open."""

    def __init__(self, failure_threshold: int, timeout_seconds: int) -> None:
        """Initialize breaker with threshold and cooldown."""
        self._failure_threshold = failure_threshold
        self._timeout_seconds = timeout_seconds
        self._failure_count = 0
        self._state = CircuitState.CLOSED
        self._opened_at: float | None = None
        self._lock = asyncio.Lock()

    async def current_state(self) -> CircuitState:
        """Return current state, transitioning from open to half-open when timed out."""
        async with self._lock:
            if self._state == CircuitState.OPEN and self._opened_at is not None:
                elapsed = time.monotonic() - self._opened_at
                if elapsed >= self._timeout_seconds:
                    self._state = CircuitState.HALF_OPEN
                    logger.info("Circuit breaker transitioned to half-open")
            return self._state

    async def is_available(self) -> bool:
        """Return whether new LLM calls are allowed."""
        state = await self.current_state()
        return state != CircuitState.OPEN

    async def record_success(self) -> None:
        """Reset failures after a successful LLM call."""
        async with self._lock:
            self._failure_count = 0
            self._state = CircuitState.CLOSED
            self._opened_at = None

    async def record_failure(self) -> None:
        """Increment failures and open the circuit when threshold is reached."""
        async with self._lock:
            self._failure_count += 1
            if self._state == CircuitState.HALF_OPEN:
                self._open_circuit()
                return
            if self._failure_count >= self._failure_threshold:
                self._open_circuit()

    def _open_circuit(self) -> None:
        """Open the circuit and start the cooldown timer."""
        self._state = CircuitState.OPEN
        self._opened_at = time.monotonic()
        logger.warning(
            "Circuit breaker opened after {} failures",
            self._failure_count,
        )

    async def retry_after_seconds(self) -> int:
        """Seconds until the circuit may accept a probe request."""
        async with self._lock:
            if self._state != CircuitState.OPEN or self._opened_at is None:
                return 0
            remaining = self._timeout_seconds - int(time.monotonic() - self._opened_at)
            return max(remaining, 1)
