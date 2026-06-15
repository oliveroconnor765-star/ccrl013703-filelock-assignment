from __future__ import annotations

import asyncio
import gc
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


@pytest.mark.asyncio
async def test_acquire_release_read(lock_file: str) -> None:
    lock = AsyncReadWriteLock(lock_file, is_singleton=False)
    proxy = await lock.acquire_read()
    assert lock._lock._lock_level == 1
    assert lock._lock._current_mode == "read"
    await lock.release()
    assert lock._lock._lock_level == 0
    assert lock._lock._current_mode is None
    assert isinstance(proxy, object)


@pytest.mark.asyncio
async def test_acquire_release_write(lock_file: str) -> None:
    lock = AsyncReadWriteLock(lock_file, is_singleton=False)
    await lock.acquire_write()
    assert lock._lock._lock_level == 1
    assert lock._lock._current_mode == "write"
    await lock.release()
    assert lock._lock._lock_level == 0
    assert lock._lock._current_mode is None


@pytest.mark.asyncio
async def test_read_lock_context_manager(lock_file: str) -> None:
    lock = AsyncReadWriteLock(lock_file, is_singleton=False)
    async with lock.read_lock():
        assert lock._lock._lock_level == 1
        assert lock._lock._current_mode == "read"
    assert lock._lock._lock_level == 0
    assert lock._lock._current_mode is None


@pytest.mark.asyncio
async def test_write_lock_context_manager(lock_file: str) -> None:
    lock = AsyncReadWriteLock(lock_file, is_singleton=False)
    async with lock.write_lock():
        assert lock._lock._lock_level == 1
        assert lock._lock._current_mode == "write"
    assert lock._lock._lock_level == 0
    assert lock._lock._current_mode is None


@pytest.mark.asyncio
async def test_reentrant_read(lock_file: str) -> None:
    lock = AsyncReadWriteLock(lock_file, is_singleton=False)
    await lock.acquire_read()
    await lock.acquire_read()
    assert lock._lock._lock_level == 2
    await lock.release()
    assert lock._lock._lock_level == 1
    assert lock._lock._current_mode == "read"
    await lock.release()
    assert lock._lock._lock_level == 0


@pytest.mark.asyncio
async def test_reentrant_write(lock_file: str) -> None:
    lock = AsyncReadWriteLock(lock_file, is_singleton=False)
    await lock.acquire_write()
    await lock.acquire_write()
    assert lock._lock._lock_level == 2
    await lock.release()
    assert lock._lock._lock_level == 1
    assert lock._lock._current_mode == "write"
    await lock.release()
    assert lock._lock._lock_level == 0


@pytest.mark.asyncio
async def test_upgrade_prohibited(lock_file: str) -> None:
    lock = AsyncReadWriteLock(lock_file, is_singleton=False)
    await lock.acquire_read()
    with pytest.raises(RuntimeError, match=r"already holding a read lock.*upgrade not allowed"):
        await lock.acquire_write()
    await lock.release()


@pytest.mark.asyncio
async def test_downgrade_prohibited(lock_file: str) -> None:
    lock = AsyncReadWriteLock(lock_file, is_singleton=False)
    await lock.acquire_write()
    with pytest.raises(RuntimeError, match=r"already holding a write lock.*downgrade not allowed"):
        await lock.acquire_read()
    await lock.release()


@pytest.mark.asyncio
async def test_non_blocking_read(lock_file: str) -> None:
    holder = ReadWriteLock(lock_file, is_singleton=False)
    holder.acquire_write()
    try:
        lock = AsyncReadWriteLock(lock_file, is_singleton=False)
        with pytest.raises(Timeout):
            await lock.acquire_read(blocking=False)
    finally:
        holder.release()


@pytest.mark.asyncio
async def test_non_blocking_write(lock_file: str) -> None:
    holder = ReadWriteLock(lock_file, is_singleton=False)
    holder.acquire_read()
    try:
        lock = AsyncReadWriteLock(lock_file, is_singleton=False)
        with pytest.raises(Timeout):
            await lock.acquire_write(blocking=False)
    finally:
        holder.release()


@pytest.mark.asyncio
async def test_timeout_expires(lock_file: str) -> None:
    holder = ReadWriteLock(lock_file, is_singleton=False)
    holder.acquire_write()
    try:
        lock = AsyncReadWriteLock(lock_file, is_singleton=False)
        with pytest.raises(Timeout):
            await lock.acquire_read(timeout=0.2)
    finally:
        holder.release()


@pytest.mark.asyncio
async def test_release_unheld_raises(lock_file: str) -> None:
    lock = AsyncReadWriteLock(lock_file, is_singleton=False)
    with pytest.raises(RuntimeError, match="not held"):
        await lock.release()


@pytest.mark.asyncio
async def test_release_force(lock_file: str) -> None:
    lock = AsyncReadWriteLock(lock_file, is_singleton=False)
    await lock.acquire_write()
    await lock.acquire_write()
    assert lock._lock._lock_level == 2
    await lock.release(force=True)
    assert lock._lock._lock_level == 0
    assert lock._lock._current_mode is None


@pytest.mark.asyncio
async def test_release_force_unheld_is_noop(lock_file: str) -> None:
    lock = AsyncReadWriteLock(lock_file, is_singleton=False)
    await lock.release(force=True)


@pytest.mark.asyncio
async def test_close(lock_file: str) -> None:
    lock = AsyncReadWriteLock(lock_file, is_singleton=False)
    await lock.acquire_write()
    assert lock._lock._lock_level == 1
    await lock.close()
    assert lock._lock._lock_level == 0
    with pytest.raises(sqlite3.ProgrammingError, match="Cannot operate on a closed database"):
        lock._lock._con.execute("SELECT 1;")


@pytest.mark.asyncio
async def test_close_on_unheld_lock(lock_file: str) -> None:
    lock = AsyncReadWriteLock(lock_file, is_singleton=False)
    await lock.close()
    with pytest.raises(sqlite3.ProgrammingError, match="Cannot operate on a closed database"):
        lock._lock._con.execute("SELECT 1;")


def test_properties(lock_file: str) -> None:
    executor = ThreadPoolExecutor(max_workers=1)
    lock = AsyncReadWriteLock(lock_file, timeout=5.0, blocking=False, is_singleton=False, executor=executor)
    assert lock.lock_file == lock_file
    assert lock.timeout == pytest.approx(5.0)
    assert lock.blocking is False
    assert lock.loop is None
    assert lock.executor is executor
    executor.shutdown(wait=False)


@pytest.mark.asyncio
async def test_custom_executor(lock_file: str) -> None:
    executor = ThreadPoolExecutor(max_workers=1)
    lock = AsyncReadWriteLock(lock_file, is_singleton=False, executor=executor)
    async with lock.read_lock():
        assert lock._lock._current_mode == "read"
    assert lock._lock._lock_level == 0
    executor.shutdown(wait=False)


@pytest.mark.asyncio
async def test_close_shuts_down_owned_executor(lock_file: str) -> None:
    lock = AsyncReadWriteLock(lock_file, is_singleton=False)
    assert lock._owns_executor is True
    executor = lock.executor
    await lock.close()
    with pytest.raises(RuntimeError):  # submitting after shutdown is rejected
        executor.submit(int)


@pytest.mark.asyncio
async def test_close_keeps_provided_executor_open(lock_file: str) -> None:
    executor = ThreadPoolExecutor(max_workers=1)
    lock = AsyncReadWriteLock(lock_file, is_singleton=False, executor=executor)
    assert lock._owns_executor is False
    await lock.close()
    assert executor.submit(int).result(timeout=5) == 0  # still usable
    executor.shutdown(wait=False)


def test_del_shuts_down_owned_executor(lock_file: str) -> None:
    lock = AsyncReadWriteLock(lock_file, is_singleton=False)
    executor = lock.executor
    lock._lock.close()  # close the connection so only the executor lifecycle is under test
    del lock
    gc.collect()
    with pytest.raises(RuntimeError):
        executor.submit(int)


def test_del_keeps_provided_executor_open(lock_file: str) -> None:
    executor = ThreadPoolExecutor(max_workers=1)
    lock = AsyncReadWriteLock(lock_file, is_singleton=False, executor=executor)
    lock._lock.close()
    del lock
    gc.collect()
    assert executor.submit(int).result(timeout=5) == 0
    executor.shutdown(wait=False)


@pytest.mark.asyncio
async def test_acquire_return_proxy_context_manager(lock_file: str) -> None:
    lock = AsyncReadWriteLock(lock_file, is_singleton=False)
    async with await lock.acquire_read() as ctx:
        assert ctx is lock
        assert lock._lock._lock_level == 1
    assert lock._lock._lock_level == 0


@pytest.mark.asyncio
async def test_nested_read_context_managers(lock_file: str) -> None:
    lock = AsyncReadWriteLock(lock_file, is_singleton=False)
    async with lock.read_lock():
        assert lock._lock._lock_level == 1
        async with lock.read_lock():
            assert lock._lock._lock_level == 2
        assert lock._lock._lock_level == 1
    assert lock._lock._lock_level == 0


@pytest.mark.asyncio
async def test_nested_write_context_managers(lock_file: str) -> None:
    lock = AsyncReadWriteLock(lock_file, is_singleton=False)
    async with lock.write_lock():
        assert lock._lock._lock_level == 1
        async with lock.write_lock():
            assert lock._lock._lock_level == 2
        assert lock._lock._lock_level == 1
    assert lock._lock._lock_level == 0


@pytest.mark.asyncio
async def test_context_manager_uses_instance_defaults(lock_file: str) -> None:
    lock = AsyncReadWriteLock(lock_file, timeout=3.0, blocking=True, is_singleton=False)
    async with lock.read_lock():
        assert lock._lock._current_mode == "read"
    async with lock.write_lock():
        assert lock._lock._current_mode == "write"


@pytest.mark.asyncio
async def test_sequential_mode_switch(lock_file: str) -> None:
    lock = AsyncReadWriteLock(lock_file, is_singleton=False)
    async with lock.read_lock():
        pass
    async with lock.write_lock():
        pass
    async with lock.read_lock():
        pass


# --- Cancellation and cleanup regression tests ---


@pytest.mark.asyncio
async def test_cancel_read_waiter_does_not_leak_lock(lock_file: str) -> None:
    """Cancelling a task waiting for a read lock must not leave the lock acquired in the background."""
    holder = AsyncReadWriteLock(lock_file, is_singleton=False)
    await holder.acquire_write()

    waiter = AsyncReadWriteLock(lock_file, is_singleton=False)
    task = asyncio.create_task(waiter.acquire_read(timeout=30))

    # Allow executor thread to start and block on SQLite
    await asyncio.sleep(0.1)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # Release holder so the waiter's executor thread can complete (and our cleanup can release)
    await holder.release()

    # Wait for background cleanup to complete
    await asyncio.sleep(0.2)

    # Waiter's internal state should be clean
    assert waiter._lock._lock_level == 0
    assert waiter._lock._current_mode is None

    # A new user should be able to acquire the lock
    newcomer = AsyncReadWriteLock(lock_file, is_singleton=False)
    await newcomer.acquire_write(timeout=2)
    assert newcomer._lock._lock_level == 1
    await newcomer.release()

    await holder.close()
    await waiter.close()
    await newcomer.close()


@pytest.mark.asyncio
async def test_cancel_write_waiter_does_not_leak_lock(lock_file: str) -> None:
    """Cancelling a task waiting for a write lock must not leave the lock acquired in the background."""
    holder = AsyncReadWriteLock(lock_file, is_singleton=False)
    await holder.acquire_read()

    waiter = AsyncReadWriteLock(lock_file, is_singleton=False)
    task = asyncio.create_task(waiter.acquire_write(timeout=30))

    await asyncio.sleep(0.1)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    await holder.release()
    await asyncio.sleep(0.2)

    assert waiter._lock._lock_level == 0
    assert waiter._lock._current_mode is None

    newcomer = AsyncReadWriteLock(lock_file, is_singleton=False)
    await newcomer.acquire_write(timeout=2)
    assert newcomer._lock._lock_level == 1
    await newcomer.release()

    await holder.close()
    await waiter.close()
    await newcomer.close()


@pytest.mark.asyncio
async def test_cancel_in_read_lock_context_manager(lock_file: str) -> None:
    """Cancelling while waiting in read_lock context manager should clean up."""
    holder = AsyncReadWriteLock(lock_file, is_singleton=False)
    await holder.acquire_write()

    waiter = AsyncReadWriteLock(lock_file, is_singleton=False)

    async def try_read_lock() -> None:
        async with waiter.read_lock(timeout=30):
            pass  # should never reach here

    task = asyncio.create_task(try_read_lock())
    await asyncio.sleep(0.1)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    await holder.release()
    await asyncio.sleep(0.2)

    # New user should be able to acquire
    newcomer = AsyncReadWriteLock(lock_file, is_singleton=False)
    await newcomer.acquire_write(timeout=2)
    await newcomer.release()

    await holder.close()
    await waiter.close()
    await newcomer.close()


@pytest.mark.asyncio
async def test_cancel_in_write_lock_context_manager(lock_file: str) -> None:
    """Cancelling while waiting in write_lock context manager should clean up."""
    holder = AsyncReadWriteLock(lock_file, is_singleton=False)
    await holder.acquire_read()

    waiter = AsyncReadWriteLock(lock_file, is_singleton=False)

    async def try_write_lock() -> None:
        async with waiter.write_lock(timeout=30):
            pass  # should never reach here

    task = asyncio.create_task(try_write_lock())
    await asyncio.sleep(0.1)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    await holder.release()
    await asyncio.sleep(0.2)

    newcomer = AsyncReadWriteLock(lock_file, is_singleton=False)
    await newcomer.acquire_write(timeout=2)
    await newcomer.release()

    await holder.close()
    await waiter.close()
    await newcomer.close()


@pytest.mark.asyncio
async def test_timeout_distinct_from_cancellation(lock_file: str) -> None:
    """Timeout should raise Timeout, not CancelledError, and not trigger cancellation cleanup."""
    holder = AsyncReadWriteLock(lock_file, is_singleton=False)
    await holder.acquire_write()

    waiter = AsyncReadWriteLock(lock_file, is_singleton=False)
    with pytest.raises(Timeout):
        await waiter.acquire_read(timeout=0.2)

    # Waiter should not have acquired the lock
    assert waiter._lock._lock_level == 0
    assert waiter._lock._current_mode is None

    await holder.release()
    await holder.close()
    await waiter.close()


@pytest.mark.asyncio
async def test_non_blocking_failure_distinct_from_cancellation(lock_file: str) -> None:
    """Non-blocking acquire should raise Timeout immediately when unavailable."""
    holder = AsyncReadWriteLock(lock_file, is_singleton=False)
    await holder.acquire_write()

    waiter = AsyncReadWriteLock(lock_file, is_singleton=False)
    with pytest.raises(Timeout):
        await waiter.acquire_read(blocking=False)

    assert waiter._lock._lock_level == 0
    assert waiter._lock._current_mode is None

    await holder.release()
    await holder.close()
    await waiter.close()


@pytest.mark.asyncio
async def test_reentrant_state_coherent_after_cancellation(lock_file: str) -> None:
    """After cancelling a waiter, reentrant state should be clean and allow normal operation."""
    holder = AsyncReadWriteLock(lock_file, is_singleton=False)
    await holder.acquire_write()

    waiter = AsyncReadWriteLock(lock_file, is_singleton=False)
    task = asyncio.create_task(waiter.acquire_write(timeout=30))
    await asyncio.sleep(0.1)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # Waiter's internal state should be clean
    assert waiter._lock._lock_level == 0
    assert waiter._lock._current_mode is None

    await holder.release()
    await asyncio.sleep(0.2)

    # Waiter should be able to acquire normally after holder releases
    await waiter.acquire_write(timeout=2)
    assert waiter._lock._lock_level == 1
    assert waiter._lock._current_mode == "write"

    # Test reentrant acquire
    await waiter.acquire_write()
    assert waiter._lock._lock_level == 2

    await waiter.release()
    assert waiter._lock._lock_level == 1
    await waiter.release()
    assert waiter._lock._lock_level == 0

    await holder.close()
    await waiter.close()


@pytest.mark.asyncio
async def test_forced_release_after_cancellation(lock_file: str) -> None:
    """Forced release should work correctly even after a cancelled acquire operation."""
    holder = AsyncReadWriteLock(lock_file, is_singleton=False)
    await holder.acquire_write()

    waiter = AsyncReadWriteLock(lock_file, is_singleton=False)
    task = asyncio.create_task(waiter.acquire_read(timeout=30))
    await asyncio.sleep(0.1)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    await holder.release()
    await asyncio.sleep(0.2)

    # Force release should be a no-op on unheld lock
    await waiter.release(force=True)
    assert waiter._lock._lock_level == 0

    # Now acquire and force release
    await waiter.acquire_read(timeout=2)
    await waiter.acquire_read()
    assert waiter._lock._lock_level == 2

    await waiter.release(force=True)
    assert waiter._lock._lock_level == 0
    assert waiter._lock._current_mode is None

    await holder.close()
    await waiter.close()


@pytest.mark.asyncio
async def test_close_after_cancellation_owned_executor(lock_file: str) -> None:
    """Closing a lock after cancellation should clean up the owned executor."""
    holder = AsyncReadWriteLock(lock_file, is_singleton=False)
    await holder.acquire_write()

    waiter = AsyncReadWriteLock(lock_file, is_singleton=False)
    assert waiter._owns_executor
    executor = waiter.executor

    task = asyncio.create_task(waiter.acquire_read(timeout=30))
    await asyncio.sleep(0.1)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    await holder.release()
    await asyncio.sleep(0.2)

    await waiter.close()

    # Owned executor should be shut down
    with pytest.raises(RuntimeError):
        executor.submit(int)

    await holder.close()


@pytest.mark.asyncio
async def test_close_after_cancellation_provided_executor(lock_file: str) -> None:
    """Closing a lock after cancellation should keep a caller-provided executor usable."""
    executor = ThreadPoolExecutor(max_workers=1)
    holder = AsyncReadWriteLock(lock_file, is_singleton=False)
    await holder.acquire_write()

    waiter = AsyncReadWriteLock(lock_file, is_singleton=False, executor=executor)
    assert not waiter._owns_executor

    task = asyncio.create_task(waiter.acquire_read(timeout=30))
    await asyncio.sleep(0.1)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    await holder.release()
    await asyncio.sleep(0.2)

    await waiter.close()

    # Provided executor should still be usable
    assert executor.submit(int).result(timeout=5) == 0
    executor.shutdown(wait=False)

    await holder.close()


@pytest.mark.asyncio
async def test_multiple_cancelled_waiters(lock_file: str) -> None:
    """Multiple cancelled waiters should all clean up without leaking."""
    holder = AsyncReadWriteLock(lock_file, is_singleton=False)
    await holder.acquire_write()

    waiters = [AsyncReadWriteLock(lock_file, is_singleton=False) for _ in range(3)]
    tasks = [asyncio.create_task(w.acquire_read(timeout=30)) for w in waiters]

    await asyncio.sleep(0.15)

    for task in tasks:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    await holder.release()
    await asyncio.sleep(0.3)

    # All waiters should have clean state
    for waiter in waiters:
        assert waiter._lock._lock_level == 0
        assert waiter._lock._current_mode is None

    # New user should be able to acquire
    newcomer = AsyncReadWriteLock(lock_file, is_singleton=False)
    await newcomer.acquire_write(timeout=2)
    await newcomer.release()

    await holder.close()
    for waiter in waiters:
        await waiter.close()
    await newcomer.close()


@pytest.mark.asyncio
async def test_cancel_before_executor_starts(lock_file: str) -> None:
    """Cancelling before the executor thread starts should not leak."""
    holder = AsyncReadWriteLock(lock_file, is_singleton=False)
    await holder.acquire_write()

    waiter = AsyncReadWriteLock(lock_file, is_singleton=False)
    task = asyncio.create_task(waiter.acquire_read(timeout=30))

    # Cancel immediately without waiting
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    await holder.release()
    await asyncio.sleep(0.2)

    # Should still be able to acquire
    newcomer = AsyncReadWriteLock(lock_file, is_singleton=False)
    await newcomer.acquire_write(timeout=2)
    await newcomer.release()

    await holder.close()
    await waiter.close()
    await newcomer.close()


@pytest.mark.asyncio
async def test_cancel_during_context_manager_yield(lock_file: str) -> None:
    """Cancelling while inside the context manager (after acquire) should release properly."""
    lock = AsyncReadWriteLock(lock_file, is_singleton=False)

    async def use_lock() -> None:
        async with lock.read_lock():
            assert lock._lock._lock_level == 1
            await asyncio.sleep(10)  # will be cancelled here

    task = asyncio.create_task(use_lock())
    await asyncio.sleep(0.05)  # let it acquire and enter sleep

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # Lock should be released
    assert lock._lock._lock_level == 0
    assert lock._lock._current_mode is None

    # Should be able to acquire again
    await lock.acquire_write(timeout=2)
    await lock.release()

    await lock.close()


@pytest.mark.asyncio
async def test_acquire_return_proxy_cancellation(lock_file: str) -> None:
    """Cancellation during acquire_read/write should not affect the return proxy semantics."""
    holder = AsyncReadWriteLock(lock_file, is_singleton=False)
    await holder.acquire_write()

    waiter = AsyncReadWriteLock(lock_file, is_singleton=False)
    task = asyncio.create_task(waiter.acquire_read(timeout=30))
    await asyncio.sleep(0.1)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    await holder.release()
    await asyncio.sleep(0.2)

    # Normal usage should still work
    async with await waiter.acquire_read() as proxy:
        assert proxy is waiter
        assert waiter._lock._lock_level == 1

    assert waiter._lock._lock_level == 0

    await holder.close()
    await waiter.close()
