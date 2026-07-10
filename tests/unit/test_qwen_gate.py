import asyncio

import pytest

from tts_playground.common.errors import ServiceError
from tts_playground.qwen.gate import BoundedInferenceGate


async def _hold_slot(
    gate: BoundedInferenceGate,
    entered: asyncio.Event,
    release: asyncio.Event,
) -> None:
    async with gate.slot():
        entered.set()
        await release.wait()


async def _wait_for_queue_depth(gate: BoundedInferenceGate, expected: int) -> None:
    for _ in range(100):
        if gate.waiting_count == expected:
            return
        await asyncio.sleep(0)
    pytest.fail(f"queue depth did not become {expected}; got {gate.waiting_count}")


async def test_gate_allows_one_active_and_only_three_waiters() -> None:
    gate = BoundedInferenceGate(max_waiting=3, queue_timeout_seconds=1.0)
    entered = asyncio.Event()
    release = asyncio.Event()
    holder = asyncio.create_task(_hold_slot(gate, entered, release))
    await entered.wait()

    waiters = [
        asyncio.create_task(_hold_slot(gate, asyncio.Event(), release))
        for _ in range(3)
    ]
    await _wait_for_queue_depth(gate, 3)

    with pytest.raises(ServiceError) as error:
        async with gate.slot():
            pytest.fail("a fourth waiter must not enter the gate")

    assert (error.value.status_code, error.value.code) == (429, "queue_full")
    assert gate.waiting_count == 3

    release.set()
    await asyncio.gather(holder, *waiters)
    assert gate.waiting_count == 0


async def test_gate_times_out_waiter_and_decrements_queue_depth() -> None:
    gate = BoundedInferenceGate(max_waiting=1, queue_timeout_seconds=0.01)
    entered = asyncio.Event()
    release = asyncio.Event()
    holder = asyncio.create_task(_hold_slot(gate, entered, release))
    await entered.wait()

    with pytest.raises(ServiceError) as error:
        async with gate.slot():
            pytest.fail("a timed-out waiter must not enter the gate")

    assert (error.value.status_code, error.value.code) == (504, "queue_timeout")
    assert gate.waiting_count == 0

    release.set()
    await holder


async def test_gate_cancellation_decrements_queue_depth() -> None:
    gate = BoundedInferenceGate(max_waiting=1, queue_timeout_seconds=1.0)
    entered = asyncio.Event()
    release = asyncio.Event()
    holder = asyncio.create_task(_hold_slot(gate, entered, release))
    await entered.wait()

    waiter = asyncio.create_task(_hold_slot(gate, asyncio.Event(), release))
    await _wait_for_queue_depth(gate, 1)
    waiter.cancel()

    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert gate.waiting_count == 0

    release.set()
    await holder
