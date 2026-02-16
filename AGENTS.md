# Repository Guidelines

## Project Structure & Module Organization

- Source: `src/nodetool/` (nodes in `src/nodetool/nodes/`, examples in `src/nodetool/examples/`, assets in `src/nodetool/assets/`).
- Tests: `tests/` mirror the source tree (e.g., `tests/nodetool/nodes/...`).
- Nodes inherit from `BaseNode` and implement async `process()` or `gen_process()`; use typed refs (`AudioRef`, `ImageRef`, etc.).

## Build, Test, and Development Commands

### ⚠️ Python Environment (IMPORTANT)

**Local Development:** Use the conda `nodetool` environment. Do not use system Python.

```bash
# Option 1: Activate the environment first
conda activate nodetool
python -m pytest tests/...

# Option 2: Use conda run (preferred for scripts/agents)
conda run -n nodetool python -m pytest tests/...
```

**GitHub CI / Copilot Agent:** Uses standard Python 3.11 with pip. Dependencies are pre-installed via `.github/workflows/copilot-setup-steps.yml`. Run commands directly:

```bash
pytest -v
pip install -e .
```

### Commands

- Install: `pip install .` (Python 3.11+). If using Poetry, `poetry install` may also work.
- Run tests: `pytest -q`
- Specific test: `pytest tests/nodetool/test_audio.py::test_specific_function -v`
- Lint/format: `black .` • `ruff check .` • `mypy .` • `flake8`

## Validation

- After making changes, always run `pytest -q` to execute tests and `ruff check .` to lint code.
- Ensure all tests pass and linting is clean before considering the task complete.
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

## Agent Nodes

### Available Agent Types

The repository includes several specialized agent node types:

- **Agent**: General-purpose LLM agent with tool use and streaming output
- **ResearchAgent**: Autonomous research agent that gathers information from the web
- **Summarizer**: Generates concise summaries of text content
- **Extractor**: Extracts structured data from text
- **Classifier**: Classifies text into predefined categories
- **ControlAgent**: Analyzes context and outputs control parameters for dynamic workflow behavior

### ControlAgent (Control Edges Support)

The **ControlAgent** works with control edges from [nodetool-core PR #587](https://github.com/nodetool-ai/nodetool-core/pull/587) to enable dynamic parameter control:

**Purpose**: Analyzes context using an LLM and outputs control parameters that can override downstream node inputs via control edges.

**Key Features**:
- Outputs control parameters via `__control_output__` handle
- Control parameters are routed via control edges (`edge_type="control"`)
- Control edges override normal data inputs on target nodes
- Enables dynamic, context-aware workflow behavior

**Example Use Cases**:
- Dynamic image processing parameters based on content analysis
- Adaptive text generation settings based on requirements
- Context-aware workflow routing decisions

**Documentation**: See `docs/control_agent.md` for detailed implementation notes.

**Example**: See `examples/control_agent_example.py` for usage patterns.
