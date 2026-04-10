## 2024-05-19 - aiofiles module resolution gotcha
**Learning:** `aiofiles.os.path` does NOT exist as an async module hierarchy in `aiofiles`. `aiofiles.os.path` resolves to the synchronous standard library `os.path`. To use async path operations with `aiofiles`, you must import `aiofiles.ospath` and call `await aiofiles.ospath.exists()` and similar methods. Prefixing `aiofiles.os.path.exists()` with `await` evaluates to `await bool`, resulting in a runtime `TypeError` crash.
**Action:** When replacing blocking `os.path` operations, specifically use `aiofiles.ospath`. Do not string-replace `os.path.` with `aiofiles.os.path.`.

## $(date +%Y-%m-%d) - Optimize CSV File Load/Save Operations
**Learning:** Using `aiofiles.read()` and `.splitlines()` reads the entire file content into an in-memory string list before processing, causing a massive memory spike and significantly worse performance for large files.
**Action:** When reading or writing potentially large structured formats like CSVs, offload the streaming I/O logic using standard synchronous tools (e.g., `csv.DictReader` and `csv.DictWriter` inside a `with open(...)` block) to `asyncio.to_thread` instead of buffering massive strings asynchronously.

## 2024-04-10 - O(N) checking optimization in list aggregation

**Learning:** When validating numeric lists in aggregations (`sum`, `min`, `max`), running `all(isinstance(x, (int, float)))` is O(N) and can be 10x-20x slower than running the C-optimized calculation directly. However, built-ins like `min` and `max` do not raise `TypeError` for homogeneous comparable strings (e.g., `min(['a', 'b'])`), so a simple `try/except TypeError` around `min()` is not enough.
**Action:** Use Python's built-ins (`res = sum(...)`) inside a `try/except TypeError` block, but **always** follow up with a post-calculation type check on the result `res` (`isinstance(res, (int, float))`) to guarantee the list consisted only of numbers and prevent silently returning string values. Maintain O(N) upfront checks for `Product` to prevent sequence repetition DoS.
