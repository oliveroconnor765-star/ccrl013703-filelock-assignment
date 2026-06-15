from __future__ import annotations

import asyncio
import contextlib
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import pytest

pytest.importorskip("sqlite3")

import sqlite3

from filelock import AsyncReadWriteLock, Timeout
from filelock._read_write import ReadWriteLock

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_singleton_cache() -> Generator[None]:
    ReadWriteLock._instances.clear()
    yield
    for ref in list(ReadWriteLock._instances.valuerefs()):
        if (lock := ref()) is not None:
            lock.close()
    ReadWriteLock._instances.clear()


@pytest.fixture
def lock_file(tmp_path: Path) -> str:
    return str(tmp_path / "test_lock.db")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _wait_for_executor_blocking(alock: AsyncReadWriteLock, *, poll: float = 0.01, limit: float = 2.0) -> None:
    """Poll until the executor thread holds the internal transaction lock.

    This confirms that the executor is inside the blocking SQLite section, making
    the subsequent task.cancel() deterministic.
    """
    elapsed = 0.0
    while not alock._lock._transaction_lock.locked():
        await asyncio.sleep(poll)
        elapsed += poll
        if elapsed >= limit:  # pragma: no cover
            msg = "executor thread did not enter blocking section within time limit"
            raise AssertionError(msg)


async def _cancel_and_join(task: asyncio.Task[object]) -> None:
    """Cancel *task* and await it, suppressing the resulting CancelledError."""
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


# ---------------------------------------------------------------------------
# 1. Cancellation must not leak lock ownership (read & write)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_read_acquire_does_not_leak(lock_file: str) -> None:
    """Cancelling a task blocked on acquire_read must not retain the lock."""
    holder = ReadWriteLock(lock_file, is_singleton=False)
    holder.acquire_write()
    alock = AsyncReadWriteLock(lock_file, is_singleton=False)
    try:
        task = asyncio.create_task(alock.acquire_read())
        await _wait_for_executor_blocking(alock)
        await _cancel_and_join(task)

        # Release the holder so the executor can complete in the background.
        holder.release()
        # Give the executor time to finish and the done-callback to fire.
        await asyncio.sleep(0.3)

        assert alock._lock._lock_level == 0
        assert alock._lock._current_mode is None
    finally:
        holder.release(force=True)
        await alock.close()


@pytest.mark.asyncio
async def test_cancel_write_acquire_does_not_leak(lock_file: str) -> None:
    """Cancelling a task blocked on acquire_write must not retain the lock."""
    holder = ReadWriteLock(lock_file, is_singleton=False)
    holder.acquire_read()
    alock = AsyncReadWriteLock(lock_file, is_singleton=False)
    try:
        task = asyncio.create_task(alock.acquire_write())
        await _wait_for_executor_blocking(alock)
        await _cancel_and_join(task)

        holder.release()
        await asyncio.sleep(0.3)

        assert alock._lock._lock_level == 0
        assert alock._lock._current_mode is None
    finally:
        holder.release(force=True)
        await alock.close()


# ---------------------------------------------------------------------------
# 2. Fresh lock users make progress after cancellation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fresh_read_after_cancelled_read(lock_file: str) -> None:
    """A new read acquire succeeds cleanly after a cancelled read waiter."""
    holder = ReadWriteLock(lock_file, is_singleton=False)
    holder.acquire_write()
    alock = AsyncReadWriteLock(lock_file, is_singleton=False)
    try:
        task = asyncio.create_task(alock.acquire_read())
        await _wait_for_executor_blocking(alock)
        await _cancel_and_join(task)
        holder.release()
        await asyncio.sleep(0.3)

        # A completely fresh lock instance should acquire without issues.
        fresh = AsyncReadWriteLock(lock_file, is_singleton=False)
        await fresh.acquire_read(timeout=2.0)
        assert fresh._lock._lock_level == 1
        assert fresh._lock._current_mode == "read"
        await fresh.release()
        assert fresh._lock._lock_level == 0
        await fresh.close()
    finally:
        holder.release(force=True)
        await alock.close()


@pytest.mark.asyncio
async def test_fresh_write_after_cancelled_write(lock_file: str) -> None:
    """A new write acquire succeeds cleanly after a cancelled write waiter."""
    holder = ReadWriteLock(lock_file, is_singleton=False)
    holder.acquire_read()
    alock = AsyncReadWriteLock(lock_file, is_singleton=False)
    try:
        task = asyncio.create_task(alock.acquire_write())
        await _wait_for_executor_blocking(alock)
        await _cancel_and_join(task)
        holder.release()
        await asyncio.sleep(0.3)

        fresh = AsyncReadWriteLock(lock_file, is_singleton=False)
        await fresh.acquire_write(timeout=2.0)
        assert fresh._lock._lock_level == 1
        assert fresh._lock._current_mode == "write"
        await fresh.release()
        assert fresh._lock._lock_level == 0
        await fresh.close()
    finally:
        holder.release(force=True)
        await alock.close()


@pytest.mark.asyncio
async def test_no_stale_sqlite_state_after_cancel(lock_file: str) -> None:
    """After cancellation and cleanup, the SQLite DB has no active transactions."""
    holder = ReadWriteLock(lock_file, is_singleton=False)
    holder.acquire_write()
    alock = AsyncReadWriteLock(lock_file, is_singleton=False)
    try:
        task = asyncio.create_task(alock.acquire_read())
        await _wait_for_executor_blocking(alock)
        await _cancel_and_join(task)
        holder.release()
        await asyncio.sleep(0.3)

        # Verify no active transaction on the cancelled lock's connection.
        assert alock._lock._con.in_transaction is False
    finally:
        holder.release(force=True)
        await alock.close()


# ---------------------------------------------------------------------------
# 3. Context-manager flows clean up on cancellation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_read_lock_context_manager(lock_file: str) -> None:
    """Cancelling during ``async with lock.read_lock()`` acquire does not leak."""
    holder = ReadWriteLock(lock_file, is_singleton=False)
    holder.acquire_write()
    alock = AsyncReadWriteLock(lock_file, is_singleton=False)
    try:

        async def ctx_waiter() -> None:
            async with alock.read_lock():
                pass  # pragma: no cover -- should not reach here

        task = asyncio.create_task(ctx_waiter())
        await _wait_for_executor_blocking(alock)
        await _cancel_and_join(task)
        holder.release()
        await asyncio.sleep(0.3)

        assert alock._lock._lock_level == 0
        assert alock._lock._current_mode is None
    finally:
        holder.release(force=True)
        await alock.close()


@pytest.mark.asyncio
async def test_cancel_write_lock_context_manager(lock_file: str) -> None:
    """Cancelling during ``async with lock.write_lock()`` acquire does not leak."""
    holder = ReadWriteLock(lock_file, is_singleton=False)
    holder.acquire_read()
    alock = AsyncReadWriteLock(lock_file, is_singleton=False)
    try:

        async def ctx_waiter() -> None:
            async with alock.write_lock():
                pass  # pragma: no cover

        task = asyncio.create_task(ctx_waiter())
        await _wait_for_executor_blocking(alock)
        await _cancel_and_join(task)
        holder.release()
        await asyncio.sleep(0.3)

        assert alock._lock._lock_level == 0
        assert alock._lock._current_mode is None
    finally:
        holder.release(force=True)
        await alock.close()


@pytest.mark.asyncio
async def test_cancel_acquire_return_proxy_context(lock_file: str) -> None:
    """Cancelling ``async with await lock.acquire_read()`` during acquire does not leak."""
    holder = ReadWriteLock(lock_file, is_singleton=False)
    holder.acquire_write()
    alock = AsyncReadWriteLock(lock_file, is_singleton=False)
    try:

        async def proxy_waiter() -> None:
            async with await alock.acquire_read():
                pass  # pragma: no cover

        task = asyncio.create_task(proxy_waiter())
        await _wait_for_executor_blocking(alock)
        await _cancel_and_join(task)
        holder.release()
        await asyncio.sleep(0.3)

        assert alock._lock._lock_level == 0
    finally:
        holder.release(force=True)
        await alock.close()


# ---------------------------------------------------------------------------
# 4. Cancellation is distinct from timeout and non-blocking failures
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_timeout_still_works_after_cancel(lock_file: str) -> None:
    """Timeout behaviour is preserved when attempted after a cancellation scenario."""
    holder = ReadWriteLock(lock_file, is_singleton=False)
    holder.acquire_write()
    alock = AsyncReadWriteLock(lock_file, is_singleton=False)
    try:
        task = asyncio.create_task(alock.acquire_read())
        await _wait_for_executor_blocking(alock)
        await _cancel_and_join(task)
        holder.release()
        await asyncio.sleep(0.3)

        # Re-block with the holder.
        holder.acquire_write()
        with pytest.raises(Timeout):
            await alock.acquire_read(timeout=0.2)
    finally:
        holder.release(force=True)
        await alock.close()


@pytest.mark.asyncio
async def test_non_blocking_still_works_after_cancel(lock_file: str) -> None:
    """Non-blocking failure is preserved when attempted after a cancellation scenario."""
    holder = ReadWriteLock(lock_file, is_singleton=False)
    holder.acquire_write()
    alock = AsyncReadWriteLock(lock_file, is_singleton=False)
    try:
        task = asyncio.create_task(alock.acquire_read())
        await _wait_for_executor_blocking(alock)
        await _cancel_and_join(task)
        holder.release()
        await asyncio.sleep(0.3)

        # Re-block with the holder.
        holder.acquire_write()
        with pytest.raises(Timeout):
            await alock.acquire_read(blocking=False)
    finally:
        holder.release(force=True)
        await alock.close()


@pytest.mark.asyncio
async def test_cancel_raises_cancelled_not_timeout(lock_file: str) -> None:
    """A cancelled acquire raises CancelledError, never Timeout."""
    holder = ReadWriteLock(lock_file, is_singleton=False)
    holder.acquire_write()
    alock = AsyncReadWriteLock(lock_file, is_singleton=False)
    try:
        task = asyncio.create_task(alock.acquire_read(timeout=10.0))
        await _wait_for_executor_blocking(alock)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        holder.release()
        await asyncio.sleep(0.2)
        await alock.close()


# ---------------------------------------------------------------------------
# 5. Reentrant state and force release after cancellation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reentrant_read_preserved_after_outer_cancel(lock_file: str) -> None:
    """Holding a read lock, then cancelling a second acquire on a *different*
    instance competing for the same file, leaves the first instance intact."""
    holder = AsyncReadWriteLock(lock_file, is_singleton=False)
    await holder.acquire_read()
    assert holder._lock._lock_level == 1

    # alock tries exclusive write -- blocked by holder's shared read
    alock = AsyncReadWriteLock(lock_file, is_singleton=False)

    task = asyncio.create_task(alock.acquire_write())
    await _wait_for_executor_blocking(alock)
    await _cancel_and_join(task)

    # holder's read lock must be completely intact
    assert holder._lock._lock_level == 1
    assert holder._lock._current_mode == "read"

    await holder.release()
    assert holder._lock._lock_level == 0
    await asyncio.sleep(0.2)
    await holder.close()
    await alock.close()


@pytest.mark.asyncio
async def test_force_release_after_cancel(lock_file: str) -> None:
    """Force release cleanly resets state after a cancelled acquire."""
    holder = ReadWriteLock(lock_file, is_singleton=False)
    holder.acquire_write()
    alock = AsyncReadWriteLock(lock_file, is_singleton=False)
    try:
        task = asyncio.create_task(alock.acquire_read())
        await _wait_for_executor_blocking(alock)
        await _cancel_and_join(task)
        holder.release()
        await asyncio.sleep(0.3)

        # Force release should be a harmless no-op (level is already 0).
        await alock.release(force=True)
        assert alock._lock._lock_level == 0
        assert alock._lock._current_mode is None
    finally:
        holder.release(force=True)
        await alock.close()


@pytest.mark.asyncio
async def test_acquire_after_cancel_on_same_instance(lock_file: str) -> None:
    """The same async lock instance can acquire successfully after cancellation."""
    holder = ReadWriteLock(lock_file, is_singleton=False)
    holder.acquire_write()
    alock = AsyncReadWriteLock(lock_file, is_singleton=False)
    try:
        task = asyncio.create_task(alock.acquire_read())
        await _wait_for_executor_blocking(alock)
        await _cancel_and_join(task)
        holder.release()
        await asyncio.sleep(0.3)

        # The same instance should be usable for a new acquire.
        await alock.acquire_write(timeout=2.0)
        assert alock._lock._lock_level == 1
        assert alock._lock._current_mode == "write"
        await alock.release()
        assert alock._lock._lock_level == 0
    finally:
        holder.release(force=True)
        await alock.close()


@pytest.mark.asyncio
async def test_mode_not_inherited_after_cancel(lock_file: str) -> None:
    """A cancelled read acquire does not leave a stale mode that blocks writes."""
    holder = ReadWriteLock(lock_file, is_singleton=False)
    holder.acquire_write()
    alock = AsyncReadWriteLock(lock_file, is_singleton=False)
    try:
        task = asyncio.create_task(alock.acquire_read())
        await _wait_for_executor_blocking(alock)
        await _cancel_and_join(task)
        holder.release()
        await asyncio.sleep(0.3)

        # Must be able to acquire write (not read-stuck) on the same instance.
        await alock.acquire_write(timeout=2.0)
        assert alock._lock._current_mode == "write"
        await alock.release()

        # And switch back to read.
        await alock.acquire_read(timeout=2.0)
        assert alock._lock._current_mode == "read"
        await alock.release()
    finally:
        holder.release(force=True)
        await alock.close()


# ---------------------------------------------------------------------------
# 6. Close after cancellation and executor ownership
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_close_after_cancel_owned_executor(lock_file: str) -> None:
    """Close after cancellation shuts down an owned executor."""
    holder = ReadWriteLock(lock_file, is_singleton=False)
    holder.acquire_write()
    alock = AsyncReadWriteLock(lock_file, is_singleton=False)
    try:
        assert alock._owns_executor is True
        executor = alock.executor

        task = asyncio.create_task(alock.acquire_read())
        await _wait_for_executor_blocking(alock)
        await _cancel_and_join(task)
        holder.release()
        await asyncio.sleep(0.3)

        await alock.close()
        with pytest.raises(RuntimeError):
            executor.submit(int)
    finally:
        holder.release(force=True)


@pytest.mark.asyncio
async def test_close_after_cancel_provided_executor(lock_file: str) -> None:
    """Close after cancellation does not shut down a caller-provided executor."""
    holder = ReadWriteLock(lock_file, is_singleton=False)
    holder.acquire_write()
    executor = ThreadPoolExecutor(max_workers=1)
    alock = AsyncReadWriteLock(lock_file, is_singleton=False, executor=executor)
    try:
        assert alock._owns_executor is False

        task = asyncio.create_task(alock.acquire_read())
        await _wait_for_executor_blocking(alock)
        await _cancel_and_join(task)
        holder.release()
        await asyncio.sleep(0.3)

        await alock.close()
        # Provided executor must remain functional.
        assert executor.submit(int).result(timeout=5) == 0
    finally:
        holder.release(force=True)
        executor.shutdown(wait=False)


@pytest.mark.asyncio
async def test_close_deterministic_after_cancel_with_sqlite(lock_file: str) -> None:
    """SQLite connection is closed cleanly after cancellation + close."""
    holder = ReadWriteLock(lock_file, is_singleton=False)
    holder.acquire_write()
    alock = AsyncReadWriteLock(lock_file, is_singleton=False)
    try:
        task = asyncio.create_task(alock.acquire_read())
        await _wait_for_executor_blocking(alock)
        await _cancel_and_join(task)
        holder.release()
        await asyncio.sleep(0.3)

        await alock.close()
        with pytest.raises(sqlite3.ProgrammingError, match="Cannot operate on a closed database"):
            alock._lock._con.execute("SELECT 1;")
    finally:
        holder.release(force=True)


# ---------------------------------------------------------------------------
# 7. Determinism: multiple sequential cancellations
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multiple_sequential_cancellations(lock_file: str) -> None:
    """Several tasks cancelled in sequence leave the lock fully clean."""
    holder = ReadWriteLock(lock_file, is_singleton=False)
    alock = AsyncReadWriteLock(lock_file, is_singleton=False)
    try:
        for _ in range(3):
            holder.acquire_write()
            task = asyncio.create_task(alock.acquire_read())
            await _wait_for_executor_blocking(alock)
            await _cancel_and_join(task)
            holder.release()
            await asyncio.sleep(0.3)

            assert alock._lock._lock_level == 0
            assert alock._lock._current_mode is None

        # Still usable after repeated cancellations.
        await alock.acquire_write(timeout=2.0)
        assert alock._lock._lock_level == 1
        await alock.release()
    finally:
        holder.release(force=True)
        await alock.close()


@pytest.mark.asyncio
async def test_cancel_read_then_cancel_write(lock_file: str) -> None:
    """Cancelling a read waiter then a write waiter leaves clean state."""
    holder = ReadWriteLock(lock_file, is_singleton=False)
    alock = AsyncReadWriteLock(lock_file, is_singleton=False)
    try:
        # Cancel a read acquire.
        holder.acquire_write()
        task = asyncio.create_task(alock.acquire_read())
        await _wait_for_executor_blocking(alock)
        await _cancel_and_join(task)
        holder.release()
        await asyncio.sleep(0.3)
        assert alock._lock._lock_level == 0

        # Cancel a write acquire.
        holder.acquire_read()
        task = asyncio.create_task(alock.acquire_write())
        await _wait_for_executor_blocking(alock)
        await _cancel_and_join(task)
        holder.release()
        await asyncio.sleep(0.3)
        assert alock._lock._lock_level == 0
        assert alock._lock._current_mode is None
    finally:
        holder.release(force=True)
        await alock.close()
