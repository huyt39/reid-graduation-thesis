# just serve for post /embedding endpoint
from __future__ import annotations

import asyncio
from dataclasses import dataclass

import structlog
import torch

log = structlog.get_logger()


@dataclass
class _Item:
    tensor: torch.Tensor          # shape [1, 3, H, W] already on device, also preprocessed image
    future: asyncio.Future         # resolved with list[float]
    model: str


class EmbeddingBatchQueue:

    def __init__(self, registry, max_batch_size: int = 32, timeout_ms: int = 10) -> None:
        self._registry = registry
        self._max = max_batch_size
        self._timeout = timeout_ms / 1000.0
        self._q: asyncio.Queue[_Item] = asyncio.Queue()
        self._task: asyncio.Task | None = None

    # lifecycle
    def start(self) -> None:
        self._task = asyncio.create_task(self._processor(), name="embedding-batch-queue")
        log.info("batch_queue.started", max_batch=self._max, timeout_ms=int(self._timeout * 1000))

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    # public api
    async def enqueue(self, image_bytes: bytes, model: str = "osnet") -> list[float]:
        """Preprocess image and enqueue; returns when the batch is executed."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future[list[float]] = loop.create_future()

        # Preprocessing is CPU-bound (PIL decode + torchvision transforms)
        tensor = await asyncio.to_thread(self._registry.preprocess_embedding, image_bytes)
        await self._q.put(_Item(tensor=tensor, future=future, model=model))
        return await future

    # background processor
    async def _processor(self) -> None:
        while True:
            # Block until at least one item is available
            try:
                first = await self._q.get()
            except asyncio.CancelledError:
                return

            batch: list[_Item] = [first]

            # Drain more items up to max_batch_size within the timeout window
            deadline = asyncio.get_event_loop().time() + self._timeout
            while len(batch) < self._max:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self._q.get(), timeout=remaining)
                    batch.append(item)
                except asyncio.TimeoutError:
                    break
                except asyncio.CancelledError:
                    self._reject_batch(batch, asyncio.CancelledError())
                    return

            await self._run_batch(batch)

    async def _run_batch(self, batch: list[_Item]) -> None:
        model_name = batch[0].model
        try:
            tensors = torch.cat([item.tensor for item in batch], dim=0)
            # Off-load the GPU/CPU forward pass to a thread so the event loop stays free
            results: list[list[float]] = await asyncio.to_thread(
                self._registry.extract_embedding_from_tensors, tensors, model_name,
            )
            for i, item in enumerate(batch):
                if not item.future.done():
                    item.future.set_result(results[i])
            log.debug("batch_queue.batch_done", size=len(batch), model=model_name)
        except Exception as exc:
            log.error("batch_queue.batch_error", exc_info=True)
            self._reject_batch(batch, exc)

    @staticmethod
    def _reject_batch(batch: list[_Item], exc: BaseException) -> None:
        for item in batch:
            if not item.future.done():
                item.future.set_exception(exc)
