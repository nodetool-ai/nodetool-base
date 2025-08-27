# Repository Guidelines

## Project Structure & Module Organization
- Source: `src/nodetool/` (nodes in `src/nodetool/nodes/`, examples in `src/nodetool/examples/`, assets in `src/nodetool/assets/`).
- Tests: `tests/` mirror the source tree (e.g., `tests/nodetool/nodes/...`).
- Nodes inherit from `BaseNode` and implement async `process()` or `gen_process()`; use typed refs (`AudioRef`, `ImageRef`, etc.).

## Build, Test, and Development Commands
- Install: `pip install .` (Python 3.11+). If using Poetry, `poetry install` may also work.
- Run tests: `pytest -q`
- Specific test: `pytest tests/nodetool/test_audio.py::test_specific_function -v`
- Lint/format: `black .` • `ruff check .` • `mypy .` • `flake8`
- Generate node metadata: `nodetool package scan`
- Generate DSL code: `nodetool codegen`

## Coding Style & Naming Conventions
- Python: 4-space indentation, type hints required for public APIs.
- Names: modules `snake_case.py`, classes `PascalCase`, functions/vars `snake_case`.
- Nodes: include a concise docstring with tags/keywords for search. Keep async I/O non-blocking.
- Use `black` formatting; keep code `ruff`/`flake8` clean and `mypy`-typed.

## Testing Guidelines
- Framework: `pytest` with `pytest-asyncio` for async nodes.
- Structure: place tests under `tests/` mirroring source paths; name files `test_*.py` and tests `test_*`.
- Practice: mock external APIs; include small sample media when needed (`tests/assets/` or inline fixtures).

## Commit & Pull Request Guidelines
- Commits: clear, imperative subject (max ~72 chars). Prefer Conventional Commits (e.g., `feat:`, `fix:`, `docs:`) when meaningful.
- PRs: include summary, rationale, screenshots/logs for UX/CLI changes, and linked issues.
- CI hygiene: ensure `pytest -q` and all linters pass locally before opening/merging.

## Node Authoring Tips
- Base pattern:
  ```python
  class ExampleNode(BaseNode):
      async def process(self, context: ProcessingContext):
          ...
  ```
- Use `ProcessingContext` for asset management and external access; avoid direct file/network ops where context helpers exist.
- After adding nodes, run `nodetool package scan` and `nodetool codegen` to refresh metadata/DSL.
