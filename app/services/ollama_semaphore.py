"""Priority-based semaphore for serializing Ollama calls.

Ollama on a laptop is single-threaded — concurrent requests just queue internally.
This semaphore makes the queue explicit and priority-aware so user-facing queries
jump ahead of background embedding/tagging work.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from enum import IntEnum

logger = logging.getLogger(__name__)


class Priority(IntEnum):
    QUERY = 0       # user waiting — highest priority
    EMBEDDING = 1   # chunk embedding during ingestion
    TAGGING = 2     # auto-tag generation — nice-to-have


class OllamaSemaphore:
    def __init__(self):
        self._queue: asyncio.PriorityQueue | None = None
        self._worker_task: asyncio.Task | None = None
        self._counter = 0  # tie-breaker for same-priority items

    def start(self):
        self._queue = asyncio.PriorityQueue()
        self._worker_task = asyncio.create_task(self._worker())
        logger.info("Ollama semaphore started")

    async def stop(self):
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        logger.info("Ollama semaphore stopped")

    async def _worker(self):
        """Pull highest-priority item, let it run, wait for it to finish."""
        while True:
            _, _, granted, done = await self._queue.get()
            granted.set()
            await done.wait()
            self._queue.task_done()

    async def execute(self, priority: Priority, fn, *args, **kwargs):
        """Submit a one-shot async callable and wait for its result."""
        granted = asyncio.Event()
        done = asyncio.Event()
        self._counter += 1
        await self._queue.put((priority, self._counter, granted, done))

        await granted.wait()
        try:
            return await fn(*args, **kwargs)
        finally:
            done.set()

    @asynccontextmanager
    async def acquire(self, priority: Priority):
        """Hold the semaphore slot for the duration of the context (e.g. streaming)."""
        granted = asyncio.Event()
        done = asyncio.Event()
        self._counter += 1
        await self._queue.put((priority, self._counter, granted, done))

        await granted.wait()
        try:
            yield
        finally:
            done.set()


ollama_semaphore = OllamaSemaphore()
