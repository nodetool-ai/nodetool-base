## 2024-05-19 - aiofiles module resolution gotcha
**Learning:** `aiofiles.os.path` does NOT exist as an async module hierarchy in `aiofiles`. `aiofiles.os.path` resolves to the synchronous standard library `os.path`. To use async path operations with `aiofiles`, you must import `aiofiles.ospath` and call `await aiofiles.ospath.exists()` and similar methods. Prefixing `aiofiles.os.path.exists()` with `await` evaluates to `await bool`, resulting in a runtime `TypeError` crash.
**Action:** When replacing blocking `os.path` operations, specifically use `aiofiles.ospath`. Do not string-replace `os.path.` with `aiofiles.os.path.`.

## $(date +%Y-%m-%d) - Optimize CSV File Load/Save Operations
**Learning:** Using `aiofiles.read()` and `.splitlines()` reads the entire file content into an in-memory string list before processing, causing a massive memory spike and significantly worse performance for large files.
**Action:** When reading or writing potentially large structured formats like CSVs, offload the streaming I/O logic using standard synchronous tools (e.g., `csv.DictReader` and `csv.DictWriter` inside a `with open(...)` block) to `asyncio.to_thread` instead of buffering massive strings asynchronously.

## $(date +%Y-%m-%d) - Optimize List Aggregations
**Learning:** For list aggregations like `Sum`, `Minimum`, and `Maximum`, using C-optimized built-ins (`sum()`, `min()`, `max()`) with EAFP (`try...except TypeError`) and a final result type check is up to 12x faster than using explicit O(N) upfront validation (`all(isinstance(x, (int, float)))`). However, for `Product` operations (`math.prod()` or `reduce`), you MUST use O(N) upfront validation (`all(isinstance(x, (int, float)))`) before calculation to prevent memory exhaustion (DoS) vulnerabilities caused by Python's sequence repetition (e.g., `10**9 * 'a'`).
**Action:** When implementing or optimizing math operations on dynamic collections, prefer EAFP with C-builtins for linear aggregations, but strictly validate input types before performing operations that can cause exponential resource consumption on strings/sequences (like multiplication/product).
