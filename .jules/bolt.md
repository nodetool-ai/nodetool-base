## 2024-05-19 - aiofiles module resolution gotcha
**Learning:** `aiofiles.os.path` does NOT exist as an async module hierarchy in `aiofiles`. `aiofiles.os.path` resolves to the synchronous standard library `os.path`. To use async path operations with `aiofiles`, you must import `aiofiles.ospath` and call `await aiofiles.ospath.exists()` and similar methods. Prefixing `aiofiles.os.path.exists()` with `await` evaluates to `await bool`, resulting in a runtime `TypeError` crash.
**Action:** When replacing blocking `os.path` operations, specifically use `aiofiles.ospath`. Do not string-replace `os.path.` with `aiofiles.os.path.`.

## $(date +%Y-%m-%d) - Optimize CSV File Load/Save Operations
**Learning:** Using `aiofiles.read()` and `.splitlines()` reads the entire file content into an in-memory string list before processing, causing a massive memory spike and significantly worse performance for large files.
**Action:** When reading or writing potentially large structured formats like CSVs, offload the streaming I/O logic using standard synchronous tools (e.g., `csv.DictReader` and `csv.DictWriter` inside a `with open(...)` block) to `asyncio.to_thread` instead of buffering massive strings asynchronously.

## $(date +%Y-%m-%d) - Optimize Pandas query evaluation for streaming items
**Learning:** Evaluating `pandas.DataFrame.query()` on a 1-row DataFrame inside a streaming loop is extremely slow and has huge overhead per evaluation. It can reduce throughput from thousands of items per second to fewer than 100.
**Action:** When a node requires applying pandas `query()` or similar vectorized operations to a stream of items, accumulate the items into a batch (e.g., 100 items), evaluate the batch as a single DataFrame, convert it back using `.to_dict('records')`, and then yield the individual items. This optimization easily provides a 20x+ speedup.
