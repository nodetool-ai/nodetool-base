## 2024-05-19 - aiofiles module resolution gotcha
**Learning:** `aiofiles.os.path` does NOT exist as an async module hierarchy in `aiofiles`. `aiofiles.os.path` resolves to the synchronous standard library `os.path`. To use async path operations with `aiofiles`, you must import `aiofiles.ospath` and call `await aiofiles.ospath.exists()` and similar methods. Prefixing `aiofiles.os.path.exists()` with `await` evaluates to `await bool`, resulting in a runtime `TypeError` crash.
**Action:** When replacing blocking `os.path` operations, specifically use `aiofiles.ospath`. Do not string-replace `os.path.` with `aiofiles.os.path.`.

## $(date +%Y-%m-%d) - Optimize CSV File Load/Save Operations
**Learning:** Using `aiofiles.read()` and `.splitlines()` reads the entire file content into an in-memory string list before processing, causing a massive memory spike and significantly worse performance for large files.
**Action:** When reading or writing potentially large structured formats like CSVs, offload the streaming I/O logic using standard synchronous tools (e.g., `csv.DictReader` and `csv.DictWriter` inside a `with open(...)` block) to `asyncio.to_thread` instead of buffering massive strings asynchronously.
## 2025-04-03 - O(N) Type Checking Bottlenecks
**Learning:** List aggregation nodes (`Sum`, `Average`, `Minimum`, `Maximum`) previously used a strict $O(N)$ upfront validation (`all(isinstance(x, (int, float)))`) before executing C-optimized built-ins. For homogeneous numerical lists, this resulted in ~19x slower performance.
**Action:** When performing aggregate math on lists, rely on Python's EAFP (Easier to Ask for Forgiveness than Permission) pattern. Execute the optimized built-in wrapped in a `try...except TypeError` block, and perform a single type check on the result. Caution: Do not apply this to `Product` operations involving sequence repetition (e.g. `10**9 * 'a'`), which require upfront O(N) validation to prevent DoS.
