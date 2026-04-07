## 2024-05-19 - aiofiles module resolution gotcha
**Learning:** `aiofiles.os.path` does NOT exist as an async module hierarchy in `aiofiles`. `aiofiles.os.path` resolves to the synchronous standard library `os.path`. To use async path operations with `aiofiles`, you must import `aiofiles.ospath` and call `await aiofiles.ospath.exists()` and similar methods. Prefixing `aiofiles.os.path.exists()` with `await` evaluates to `await bool`, resulting in a runtime `TypeError` crash.
**Action:** When replacing blocking `os.path` operations, specifically use `aiofiles.ospath`. Do not string-replace `os.path.` with `aiofiles.os.path.`.

## $(date +%Y-%m-%d) - Optimize CSV File Load/Save Operations
**Learning:** Using `aiofiles.read()` and `.splitlines()` reads the entire file content into an in-memory string list before processing, causing a massive memory spike and significantly worse performance for large files.
**Action:** When reading or writing potentially large structured formats like CSVs, offload the streaming I/O logic using standard synchronous tools (e.g., `csv.DictReader` and `csv.DictWriter` inside a `with open(...)` block) to `asyncio.to_thread` instead of buffering massive strings asynchronously.
## 2025-04-07 - Pandas Iterrows Performance Bottleneck
**Learning:** `df.iterrows()` inside Python loops is incredibly slow because Pandas creates a new `Series` object for every single row. In workflow nodes that iterate over rows (like `RowIterator` or `ForEachRow`), this creates a massive processing overhead.
**Action:** Replace `for index, row in df.iterrows():` with `for index, row in zip(df.index, df.to_dict('records')):` when you need the row data as dictionaries. This offloads the dictionary conversion to C-optimized code in bulk, running ~20x faster than creating individual Series objects in a loop.
