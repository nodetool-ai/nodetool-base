## 2024-05-19 - aiofiles module resolution gotcha
**Learning:** `aiofiles.os.path` does NOT exist as an async module hierarchy in `aiofiles`. `aiofiles.os.path` resolves to the synchronous standard library `os.path`. To use async path operations with `aiofiles`, you must import `aiofiles.ospath` and call `await aiofiles.ospath.exists()` and similar methods. Prefixing `aiofiles.os.path.exists()` with `await` evaluates to `await bool`, resulting in a runtime `TypeError` crash.
**Action:** When replacing blocking `os.path` operations, specifically use `aiofiles.ospath`. Do not string-replace `os.path.` with `aiofiles.os.path.`.

## $(date +%Y-%m-%d) - Optimize CSV File Load/Save Operations
**Learning:** Using `aiofiles.read()` and `.splitlines()` reads the entire file content into an in-memory string list before processing, causing a massive memory spike and significantly worse performance for large files.
**Action:** When reading or writing potentially large structured formats like CSVs, offload the streaming I/O logic using standard synchronous tools (e.g., `csv.DictReader` and `csv.DictWriter` inside a `with open(...)` block) to `asyncio.to_thread` instead of buffering massive strings asynchronously.
## 2026-04-13 - Replaced df.iterrows() with zip(df.index, df.to_dict('records'))
**Learning:** Iterating over Pandas DataFrames using `iterrows()` in a Python loop is an anti-pattern for performance-critical pathing because Pandas constructs a distinct Series object for each row, adding huge overhead. Using `zip(df.index, df.to_dict('records'))` bulk-converts the data into Python dictionaries immediately, bypassing the Series wrapper and speeding up row iteration by ~20x.
**Action:** When a Node workflow requires iterating and yielding dictionary representations of DataFrame rows, always use `df.to_dict('records')` paired with `zip(df.index, ...)` rather than looping with `iterrows()`.
