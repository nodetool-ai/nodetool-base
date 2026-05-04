## 2024-05-19 - aiofiles module resolution gotcha
**Learning:** `aiofiles.os.path` does NOT exist as an async module hierarchy in `aiofiles`. `aiofiles.os.path` resolves to the synchronous standard library `os.path`. To use async path operations with `aiofiles`, you must import `aiofiles.ospath` and call `await aiofiles.ospath.exists()` and similar methods. Prefixing `aiofiles.os.path.exists()` with `await` evaluates to `await bool`, resulting in a runtime `TypeError` crash.
**Action:** When replacing blocking `os.path` operations, specifically use `aiofiles.ospath`. Do not string-replace `os.path.` with `aiofiles.os.path.`.

## $(date +%Y-%m-%d) - Optimize CSV File Load/Save Operations
**Learning:** Using `aiofiles.read()` and `.splitlines()` reads the entire file content into an in-memory string list before processing, causing a massive memory spike and significantly worse performance for large files.
**Action:** When reading or writing potentially large structured formats like CSVs, offload the streaming I/O logic using standard synchronous tools (e.g., `csv.DictReader` and `csv.DictWriter` inside a `with open(...)` block) to `asyncio.to_thread` instead of buffering massive strings asynchronously.

## $(date +%Y-%m-%d) - Pandas iterrows Performance
**Learning:** Using `df.iterrows()` inside a loop is extremely slow because it creates a new `pd.Series` object for every single row. For a DataFrame iteration that converts each row to a dictionary, bulk dictionary conversion via `zip(df.index, df.to_dict('records'))` is significantly faster (~10-20x) and should be preferred over `iterrows()`. Be aware that the `row` item is already a dictionary in this approach, so subsequent `.to_dict()` calls on the row must be removed.
**Action:** Always replace `df.iterrows()` with `zip(df.index, df.to_dict('records'))` or similar optimized bulk iteration methods when processing DataFrames row-by-row, especially if row modification or dictionary conversion is involved.
