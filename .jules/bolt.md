## 2024-05-19 - aiofiles module resolution gotcha
**Learning:** `aiofiles.os.path` does NOT exist as an async module hierarchy in `aiofiles`. `aiofiles.os.path` resolves to the synchronous standard library `os.path`. To use async path operations with `aiofiles`, you must import `aiofiles.ospath` and call `await aiofiles.ospath.exists()` and similar methods. Prefixing `aiofiles.os.path.exists()` with `await` evaluates to `await bool`, resulting in a runtime `TypeError` crash.
**Action:** When replacing blocking `os.path` operations, specifically use `aiofiles.ospath`. Do not string-replace `os.path.` with `aiofiles.os.path.`.

## $(date +%Y-%m-%d) - Optimize CSV File Load/Save Operations
**Learning:** Using `aiofiles.read()` and `.splitlines()` reads the entire file content into an in-memory string list before processing, causing a massive memory spike and significantly worse performance for large files.
**Action:** When reading or writing potentially large structured formats like CSVs, offload the streaming I/O logic using standard synchronous tools (e.g., `csv.DictReader` and `csv.DictWriter` inside a `with open(...)` block) to `asyncio.to_thread` instead of buffering massive strings asynchronously.
## $(date +%Y-%m-%d) - Optimize Pandas Iteration
**Learning:** `df.iterrows()` in `pandas` is severely unoptimized for production logic loops as it constructs intermediate pandas Series for every single row. Standard list/dict generator iteration is immensely faster.
**Action:** When a sequence of individual row dictionaries is required (like standard `for` loops on big DataFrames), use `zip(df.index, df.to_dict('records'))`. This bypasses per-row Series instantiation and uses the C-backed bulk creation mechanism in `pandas`, returning standard Python dictionaries ~20x faster. Be aware this eliminates the need to call `.to_dict()` inside the body of the loop.
