## 2024-05-19 - aiofiles module resolution gotcha
**Learning:** `aiofiles.os.path` does NOT exist as an async module hierarchy in `aiofiles`. `aiofiles.os.path` resolves to the synchronous standard library `os.path`. To use async path operations with `aiofiles`, you must import `aiofiles.ospath` and call `await aiofiles.ospath.exists()` and similar methods. Prefixing `aiofiles.os.path.exists()` with `await` evaluates to `await bool`, resulting in a runtime `TypeError` crash.
**Action:** When replacing blocking `os.path` operations, specifically use `aiofiles.ospath`. Do not string-replace `os.path.` with `aiofiles.os.path.`.

## 2024-05-19 - Exception Handling Bottleneck in DB Query Nodes
**Learning:** Using `try: json.loads(...) except json.JSONDecodeError: pass` on every string cell fetched from an SQLite query is a massive performance bottleneck because raising exceptions for routine data (like non-JSON strings) in Python is extremely slow.
**Action:** When dynamically parsing strings from a database to identify JSON objects/arrays, use a heuristic check (`value.strip().startswith(('{', '['))`) before attempting `json.loads`. This dramatically reduces exception overhead and optimizes query times for large datasets.
