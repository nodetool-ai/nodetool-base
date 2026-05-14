## 2024-05-19 - aiofiles module resolution gotcha
**Learning:** `aiofiles.os.path` does NOT exist as an async module hierarchy in `aiofiles`. `aiofiles.os.path` resolves to the synchronous standard library `os.path`. To use async path operations with `aiofiles`, you must import `aiofiles.ospath` and call `await aiofiles.ospath.exists()` and similar methods. Prefixing `aiofiles.os.path.exists()` with `await` evaluates to `await bool`, resulting in a runtime `TypeError` crash.
**Action:** When replacing blocking `os.path` operations, specifically use `aiofiles.ospath`. Do not string-replace `os.path.` with `aiofiles.os.path.`.

## $(date +%Y-%m-%d) - Optimize CSV File Load/Save Operations
**Learning:** Using `aiofiles.read()` and `.splitlines()` reads the entire file content into an in-memory string list before processing, causing a massive memory spike and significantly worse performance for large files.
**Action:** When reading or writing potentially large structured formats like CSVs, offload the streaming I/O logic using standard synchronous tools (e.g., `csv.DictReader` and `csv.DictWriter` inside a `with open(...)` block) to `asyncio.to_thread` instead of buffering massive strings asynchronously.

## 2024-05-19 - Optimize List Aggregations (Sum/Average)
**Learning:** Using explicit `all(isinstance(x, (int, float)))` loop type checks for list aggregations is significantly slower (~10x+ slower for large lists) than simply using the C-optimized `sum()` built-in function wrapped in a `try...except TypeError` (EAFP). However, this only applies to `sum()`; operations like `min()` and `max()` have inconsistent mixed-type evaluation and cannot safely use this pattern.
**Action:** When calculating sums or averages for potentially large arrays, offload type validation to C-level built-ins via `try...except TypeError` (EAFP) rather than iterating explicitly in Python, but always double-check the final returned type to ensure mixed non-crashing types didn't result in an invalid structure.
