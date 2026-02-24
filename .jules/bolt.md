## 2024-05-23 - [Blocking Image Operations]
**Learning:** Found multiple image processing nodes (Resize, Scale, Crop, Fit, Paste) executing blocking PIL operations directly in async def process. This blocks the asyncio event loop.
**Action:** Always wrap CPU-intensive PIL operations and blocking I/O (like image.save) in await asyncio.to_thread(...) within async node methods.
