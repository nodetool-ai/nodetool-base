# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Essential Commands

**Install dependencies:**

```bash
pip install .
```

**Run tests:**

```bash
pytest -q
```

**Run specific test:**

```bash
pytest tests/nodetool/test_audio.py::test_specific_function -v
```

**Linting and formatting (from requirements-dev.txt):**

```bash
black .
ruff check .
mypy .
flake8
```

**Generating node metadata**

```bash
nodetool package scan
```

**Generating DSL code**

```bash
nodetool codegen
```

## Architecture Overview

This is a node-based system for composing AI workflows. Key architectural patterns:

### Node System

- All nodes inherit from `BaseNode` (imported from nodetool-core)
- Nodes implement async `process()` or `gen_process()` methods
- `ProcessingContext` provides runtime services (asset management, API access, etc.)
- Type system uses refs for media types: `AudioRef`, `ImageRef`, `VideoRef`, `FolderRef`

### Node Structure Pattern

```python
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext

class ExampleNode(BaseNode):
    """
    Brief description
    tags, keywords, for, search
    """
    # Pydantic fields for inputs

    async def process(self, context: ProcessingContext):
        # Implementation
```

### Directory Organization

- `src/nodetool/nodes/` - All node implementations organized by namespace
- `src/nodetool/examples/` - JSON workflow examples
- `src/nodetool/assets/` - Workflow thumbnails
- `tests/` - Test files mirror source structure

### Testing Patterns

- Use `pytest-asyncio` for async node tests
- Parametrized tests for multiple scenarios
- Mock external API calls
- Test files include sample media (test.jpg, test.mp3, test.mp4)

## Node Categories

- **calendly/** - Calendly API integration
- **chroma/** - Vector database operations (collections, indexing, queries)
- **google/** - Google services (image generation)
- **lib/** - Library wrappers (BeautifulSoup, LlamaIndex, PyMuPDF, etc.)
- **nodetool/** - Core functionality (audio, image, video, text, math, etc.)
- **openai/** - OpenAI API wrappers (GPT, DALL-E, Whisper)

## Development Notes

- Python 3.11+ required
- Uses uv for dependency management
- `nodetool-core` dependency installed from git
- Async/await patterns throughout
- Strong typing with Pydantic models
- Documentation auto-generated in `docs/` folder
