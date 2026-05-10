## 2024-05-19 - aiofiles module resolution gotcha
**Learning:** `aiofiles.os.path` does NOT exist as an async module hierarchy in `aiofiles`. `aiofiles.os.path` resolves to the synchronous standard library `os.path`. To use async path operations with `aiofiles`, you must import `aiofiles.ospath` and call `await aiofiles.ospath.exists()` and similar methods. Prefixing `aiofiles.os.path.exists()` with `await` evaluates to `await bool`, resulting in a runtime `TypeError` crash.
**Action:** When replacing blocking `os.path` operations, specifically use `aiofiles.ospath`. Do not string-replace `os.path.` with `aiofiles.os.path.`.

## $(date +%Y-%m-%d) - Optimize CSV File Load/Save Operations
**Learning:** Using `aiofiles.read()` and `.splitlines()` reads the entire file content into an in-memory string list before processing, causing a massive memory spike and significantly worse performance for large files.
**Action:** When reading or writing potentially large structured formats like CSVs, offload the streaming I/O logic using standard synchronous tools (e.g., `csv.DictReader` and `csv.DictWriter` inside a `with open(...)` block) to `asyncio.to_thread` instead of buffering massive strings asynchronously.

## 2024-05-10 - Optimize list aggregations with EAFP
**Learning:** Using Python's C-optimized `sum()`, `min()`, and `max()` inside a `try...except` block is much faster than explicit `all(isinstance(...))` loop checks for large lists, but requires a final `isinstance` check because `min(['a', 'b'])` succeeds for homogenous non-numeric types. O(N) checking is still required for `reduce` operations like `Product` to prevent string repetition DoS.
**Action:** Apply the EAFP (`try...except`) pattern with a post-calculation type check for aggregations using C-optimized built-ins on lists, but keep upfront O(N) validation for potentially unbounded memory operations like sequence multiplication.
