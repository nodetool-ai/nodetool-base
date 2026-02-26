import asyncio
import time
import os
import aiofiles
import statistics

# Configuration
FILE_SIZE_MB = 100
FILE_SIZE = FILE_SIZE_MB * 1024 * 1024
FILENAME_SYNC = "test_sync.dat"
FILENAME_ASYNC = "test_async.dat"

async def loop_monitor(interval=0.01):
    """
    Monitors the event loop lag.
    """
    lags = []
    running = True

    async def monitor():
        nonlocal running
        while running:
            start = time.perf_counter()
            await asyncio.sleep(interval)
            end = time.perf_counter()
            actual_sleep = end - start
            lag = actual_sleep - interval
            lags.append(lag)

    task = asyncio.create_task(monitor())
    return task, lags, lambda: setattr(task, 'running', False) or task.cancel()

def sync_write(data):
    """Writes data synchronously (blocking)."""
    with open(FILENAME_SYNC, "wb") as f:
        f.write(data)

async def async_write(data):
    """Writes data asynchronously using aiofiles."""
    async with aiofiles.open(FILENAME_ASYNC, "wb") as f:
        await f.write(data)

async def run_benchmark():
    print(f"Generating {FILE_SIZE_MB}MB of data...")
    data = os.urandom(FILE_SIZE)

    print("\n--- Benchmarking Synchronous Write (Current Behavior) ---")
    monitor_task, lags_sync, stop_monitor = await loop_monitor()

    start_time = time.perf_counter()
    # Simulate the blocking call inside an async function
    sync_write(data)
    end_time = time.perf_counter()

    stop_monitor()
    try:
        await monitor_task
    except asyncio.CancelledError:
        pass

    print(f"Write Duration: {end_time - start_time:.4f}s")
    if lags_sync:
        max_lag = max(lags_sync)
        avg_lag = statistics.mean(lags_sync)
        print(f"Max Loop Lag: {max_lag:.4f}s")
        print(f"Avg Loop Lag: {avg_lag:.4f}s")
    else:
        print("No lag samples collected (loop was completely blocked?)")

    # Cleanup
    if os.path.exists(FILENAME_SYNC):
        os.remove(FILENAME_SYNC)

    print("\n--- Benchmarking Asynchronous Write (Optimized Behavior) ---")
    monitor_task, lags_async, stop_monitor = await loop_monitor()

    start_time = time.perf_counter()
    await async_write(data)
    end_time = time.perf_counter()

    stop_monitor()
    try:
        await monitor_task
    except asyncio.CancelledError:
        pass

    print(f"Write Duration: {end_time - start_time:.4f}s")
    if lags_async:
        max_lag = max(lags_async)
        avg_lag = statistics.mean(lags_async)
        print(f"Max Loop Lag: {max_lag:.4f}s")
        print(f"Avg Loop Lag: {avg_lag:.4f}s")
    else:
        print("No lag samples collected")

    # Cleanup
    if os.path.exists(FILENAME_ASYNC):
        os.remove(FILENAME_ASYNC)

if __name__ == "__main__":
    asyncio.run(run_benchmark())
