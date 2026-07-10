from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from tts_playground.common.errors import ServiceError


class BoundedInferenceGate:
    def __init__(self, *, max_waiting: int, queue_timeout_seconds: float) -> None:
        if max_waiting < 0:
            raise ValueError("max_waiting must not be negative")
        self._max_waiting = max_waiting
        self._queue_timeout_seconds = queue_timeout_seconds
        self._active_lock = asyncio.Lock()
        self._waiting_lock = asyncio.Lock()
        self._waiting_count = 0
        self._occupied = False

    @property
    def waiting_count(self) -> int:
        return self._waiting_count

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[float]:
        started_at = time.monotonic()
        waiting = await self._reserve()
        acquired = False
        try:
            if waiting:
                try:
                    async with asyncio.timeout(self._queue_timeout_seconds):
                        await self._active_lock.acquire()
                except TimeoutError:
                    raise ServiceError(
                        status_code=504,
                        code="queue_timeout",
                        message="Qwen request queue timed out.",
                    ) from None
            else:
                await self._active_lock.acquire()
            acquired = True
        finally:
            if waiting:
                await self._finish_waiting(acquired=acquired)
            elif not acquired:
                await self._abandon_reservation()

        try:
            yield time.monotonic() - started_at
        finally:
            self._active_lock.release()
            await self._release_reservation_if_idle()

    async def _reserve(self) -> bool:
        async with self._waiting_lock:
            if not self._occupied:
                self._occupied = True
                return False
            if self._waiting_count >= self._max_waiting:
                raise ServiceError(
                    status_code=429,
                    code="queue_full",
                    message="Qwen request queue is full.",
                )
            self._waiting_count += 1
            return True

    async def _finish_waiting(self, *, acquired: bool) -> None:
        async with self._waiting_lock:
            self._waiting_count -= 1
            if not acquired and self._waiting_count == 0 and not self._active_lock.locked():
                self._occupied = False

    async def _abandon_reservation(self) -> None:
        async with self._waiting_lock:
            if self._waiting_count == 0 and not self._active_lock.locked():
                self._occupied = False

    async def _release_reservation_if_idle(self) -> None:
        async with self._waiting_lock:
            if self._waiting_count == 0 and not self._active_lock.locked():
                self._occupied = False
