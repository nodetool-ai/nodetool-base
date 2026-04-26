## 2024-05-19 - aiofiles module resolution gotcha
**Learning:** `aiofiles.os.path` does NOT exist as an async module hierarchy in `aiofiles`. `aiofiles.os.path` resolves to the synchronous standard library `os.path`. To use async path operations with `aiofiles`, you must import `aiofiles.ospath` and call `await aiofiles.ospath.exists()` and similar methods. Prefixing `aiofiles.os.path.exists()` with `await` evaluates to `await bool`, resulting in a runtime `TypeError` crash.
**Action:** When replacing blocking `os.path` operations, specifically use `aiofiles.ospath`. Do not string-replace `os.path.` with `aiofiles.os.path.`.

## $(date +%Y-%m-%d) - Optimize CSV File Load/Save Operations
**Learning:** Using `aiofiles.read()` and `.splitlines()` reads the entire file content into an in-memory string list before processing, causing a massive memory spike and significantly worse performance for large files.
**Action:** When reading or writing potentially large structured formats like CSVs, offload the streaming I/O logic using standard synchronous tools (e.g., `csv.DictReader` and `csv.DictWriter` inside a `with open(...)` block) to `asyncio.to_thread` instead of buffering massive strings asynchronously.
## 2024-05-18 - Pandas iteration optimization
**Learning:** Using `df.iterrows()` in Pandas is a massive performance bottleneck because it creates a new Series object for each row under the hood, significantly slowing down loops. A more efficient alternative is to iterate over a bulk dictionary representation created via `zip(df.index, df.to_dict('records'))`.
**Action:** When working with Pandas DataFrames inside loop bodies (like generators returning sequential dictionaries), hunt for `iterrows()` calls and replace them with `zip(df.index, df.to_dict('records'))`. Ensure that any `.to_dict()` calls inside the loop body itself are removed, as the `row` variable will now already be a dictionary instead of a Series.
