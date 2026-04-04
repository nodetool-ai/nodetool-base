## 2024-05-19 - aiofiles module resolution gotcha
**Learning:** `aiofiles.os.path` does NOT exist as an async module hierarchy in `aiofiles`. `aiofiles.os.path` resolves to the synchronous standard library `os.path`. To use async path operations with `aiofiles`, you must import `aiofiles.ospath` and call `await aiofiles.ospath.exists()` and similar methods. Prefixing `aiofiles.os.path.exists()` with `await` evaluates to `await bool`, resulting in a runtime `TypeError` crash.
**Action:** When replacing blocking `os.path` operations, specifically use `aiofiles.ospath`. Do not string-replace `os.path.` with `aiofiles.os.path.`.

## $(date +%Y-%m-%d) - Optimize CSV File Load/Save Operations
**Learning:** Using `aiofiles.read()` and `.splitlines()` reads the entire file content into an in-memory string list before processing, causing a massive memory spike and significantly worse performance for large files.
**Action:** When reading or writing potentially large structured formats like CSVs, offload the streaming I/O logic using standard synchronous tools (e.g., `csv.DictReader` and `csv.DictWriter` inside a `with open(...)` block) to `asyncio.to_thread` instead of buffering massive strings asynchronously.

## 2024-05-18 - EAFP Optimization for List Aggregations
**Learning:** In list aggregation nodes (`Sum`, `Average`, `Minimum`, `Maximum`), applying the EAFP pattern by attempting the C-optimized aggregation function first (e.g., `try: res = sum(values) except TypeError: ...`) and verifying the single result type is ~14x faster than performing O(N) explicit type validation on the input list (`all(isinstance(x, (int, float)) for x in self.values)`). However, `min` and `max` allow homogenous strings without raising `TypeError`, making the final result type-check critical to maintain identical validation logic.
**Action:** Use EAFP and final-result type checking for aggregations instead of O(N) upfront element-by-element type validation, unless repetitive execution vulnerabilities exist (like string multiplication in `Product`).
