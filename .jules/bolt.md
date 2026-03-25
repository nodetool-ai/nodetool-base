## 2024-05-19 - aiofiles module resolution gotcha
**Learning:** `aiofiles.os.path` does NOT exist as an async module hierarchy in `aiofiles`. `aiofiles.os.path` resolves to the synchronous standard library `os.path`. To use async path operations with `aiofiles`, you must import `aiofiles.ospath` and call `await aiofiles.ospath.exists()` and similar methods. Prefixing `aiofiles.os.path.exists()` with `await` evaluates to `await bool`, resulting in a runtime `TypeError` crash.
**Action:** When replacing blocking `os.path` operations, specifically use `aiofiles.ospath`. Do not string-replace `os.path.` with `aiofiles.os.path.`.

## 2026-03-25 - Optimize CSV File Load/Save Operations
**Learning:** Using `aiofiles.read()` and `.splitlines()` reads the entire file content into an in-memory string list before processing, causing a massive memory spike and significantly worse performance for large files.
**Action:** When reading or writing potentially large structured formats like CSVs, offload the streaming I/O logic using standard synchronous tools (e.g., `csv.DictReader` and `csv.DictWriter` inside a `with open(...)` block) to `asyncio.to_thread` instead of buffering massive strings asynchronously.

## 2026-03-25 - Optimize DataFrame Row Iteration
**Learning:** Using `df.iterrows()` and calling `row.to_dict()` inside the loop is a severe performance bottleneck for large DataFrames because it instantiates a new Pandas Series object for every single row.
**Action:** Replace `df.iterrows()` with `zip(df.index, df.to_dict('records'))` when iterating to yield row dictionaries. This approach performs the dictionary conversion in bulk via C extensions, offering ~20x faster iteration speeds while handling non-unique indexes properly (unlike `df.to_dict('index')`).
