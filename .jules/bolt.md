## 2024-05-19 - aiofiles module resolution gotcha
**Learning:** `aiofiles.os.path` does NOT exist as an async module hierarchy in `aiofiles`. `aiofiles.os.path` resolves to the synchronous standard library `os.path`. To use async path operations with `aiofiles`, you must import `aiofiles.ospath` and call `await aiofiles.ospath.exists()` and similar methods. Prefixing `aiofiles.os.path.exists()` with `await` evaluates to `await bool`, resulting in a runtime `TypeError` crash.
**Action:** When replacing blocking `os.path` operations, specifically use `aiofiles.ospath`. Do not string-replace `os.path.` with `aiofiles.os.path.`.

## $(date +%Y-%m-%d) - Optimize CSV File Load/Save Operations
**Learning:** Using `aiofiles.read()` and `.splitlines()` reads the entire file content into an in-memory string list before processing, causing a massive memory spike and significantly worse performance for large files.
**Action:** When reading or writing potentially large structured formats like CSVs, offload the streaming I/O logic using standard synchronous tools (e.g., `csv.DictReader` and `csv.DictWriter` inside a `with open(...)` block) to `asyncio.to_thread` instead of buffering massive strings asynchronously.
## 2025-02-28 - Use asyncio.Semaphore instead of chunking for IO-bound work
**Learning:** For HTTP nodes like `FilterValidURLs` and `DownloadFiles`, using chunked batch processing (`if len(tasks) >= batch_size: await asyncio.gather(*tasks)`) creates artificial bottlenecks where fast requests have to wait for the slowest request in their chunk before moving on.
**Action:** Always prefer `asyncio.Semaphore(max_concurrent)` wrapped inside a closure (`async with sem:`) and gathering all tasks at once for IO-bound concurrency where order doesn't matter, yielding significantly better throughput.
