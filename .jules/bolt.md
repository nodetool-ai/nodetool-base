## 2024-05-19 - aiofiles module resolution gotcha
**Learning:** `aiofiles.os.path` does NOT exist as an async module hierarchy in `aiofiles`. `aiofiles.os.path` resolves to the synchronous standard library `os.path`. To use async path operations with `aiofiles`, you must import `aiofiles.ospath` and call `await aiofiles.ospath.exists()` and similar methods. Prefixing `aiofiles.os.path.exists()` with `await` evaluates to `await bool`, resulting in a runtime `TypeError` crash.
**Action:** When replacing blocking `os.path` operations, specifically use `aiofiles.ospath`. Do not string-replace `os.path.` with `aiofiles.os.path.`.

## 2026-04-05 - Optimize CSV File Load/Save Operations
**Learning:** Using `aiofiles.read()` and `.splitlines()` reads the entire file content into an in-memory string list before processing, causing a massive memory spike and significantly worse performance for large files.
**Action:** When reading or writing potentially large structured formats like CSVs, offload the streaming I/O logic using standard synchronous tools (e.g., `csv.DictReader` and `csv.DictWriter` inside a `with open(...)` block) to `asyncio.to_thread` instead of buffering massive strings asynchronously.

## 2026-04-05 - EAFP list aggregation pitfalls with min/max
**Learning:** When optimizing list aggregations using Python's built-in `min()` and `max()` with an EAFP `try...except TypeError` pattern, be aware that `min/max` will not naturally raise a `TypeError` for lists of homogenous non-numeric types (e.g., `min(['a', 'b'])` simply returns `'a'`).
**Action:** Always follow the execution of `min/max` with a post-calculation type check (e.g., `isinstance(res, (int, float))`) to guarantee numeric results when avoiding explicit O(N) upfront validation.
