## 2024-05-19 - aiofiles module resolution gotcha
**Learning:** `aiofiles.os.path` does NOT exist as an async module hierarchy in `aiofiles`. `aiofiles.os.path` resolves to the synchronous standard library `os.path`. To use async path operations with `aiofiles`, you must import `aiofiles.ospath` and call `await aiofiles.ospath.exists()` and similar methods. Prefixing `aiofiles.os.path.exists()` with `await` evaluates to `await bool`, resulting in a runtime `TypeError` crash.
**Action:** When replacing blocking `os.path` operations, specifically use `aiofiles.ospath`. Do not string-replace `os.path.` with `aiofiles.os.path.`.

## 2026-03-11 - Optimize CSV File Load/Save Operations
**Learning:** Using `aiofiles.read()` and `.splitlines()` reads the entire file content into an in-memory string list before processing, causing a massive memory spike and significantly worse performance for large files.
**Action:** When reading or writing potentially large structured formats like CSVs, offload the streaming I/O logic using standard synchronous tools (e.g., `csv.DictReader` and `csv.DictWriter` inside a `with open(...)` block) to `asyncio.to_thread` instead of buffering massive strings asynchronously.

## 2026-03-11 - Optimize Recursive List Flattening Memory Overhead
**Learning:** To optimize the processing of heavily nested list structures (like the `Flatten` node), recursively creating and extending intermediate arrays (`result.extend()`) generates a large number of intermediate lists on the heap, leading to high memory overhead and degraded performance for deep nesting.
**Action:** Prefer using Python generators with `yield` and `yield from` rather than recursive list extensions. This avoids the overhead of constructing intermediate lists during deep call stacks and scales much better for highly nested or large list inputs.

## $(date +%Y-%m-%d) - Pydantic Field Annotation Error with Inner Imports
**Learning:** Do not place generic `import` statements (like `from typing import Iterator`) directly inside a Pydantic `BaseModel` or `BaseNode` class body. Pydantic's namespace inspection will treat the imported name (e.g., `Iterator`) as a non-annotated model field and raise a `PydanticUserError` during class definition, crashing the application.
**Action:** Always place module imports at the top level of the file, outside of Pydantic model classes, or use fully qualified names or string forward references if local imports are strictly necessary.
