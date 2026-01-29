# nodetool-base

A collection of reusable nodes for [Nodetool](https://github.com/nodetool-ai/nodetool). These nodes implement common functionality for text, audio, video, images and more. They build on the runtime provided by [nodetool-core](https://github.com/nodetool-ai/nodetool-core).

## Overview

Nodetool lets you compose AI workflows as graphs. `nodetool-base` ships with a rich set of nodes so you can build useful flows out of the box. The nodes are organised into several namespaces under `src/nodetool/nodes`:

- `anthropic` – Claude AI integration for chat and agent interactions
- `gemini` – Google Gemini models for text, audio, images and video generation
- `kie` – Kie.ai integrations for advanced media generation (Suno, Grok, Hailuo, Kling)
- `lib` – helpers using libraries like BeautifulSoup, PyMuPDF, NumPy, Pillow and more
- `messaging` – Discord and Telegram messaging integrations
- `nodetool` – core utilities such as audio, boolean logic, image processing and more
- `openai` – OpenAI API wrappers for chat, agents, audio and image generation
- `search` – Google search integration
- `vector` – vector database operations (Chroma, FAISS)

Example workflows using these nodes can be found in `src/nodetool/examples/nodetool-base`.

## Installation

```bash
git clone https://github.com/nodetool-ai/nodetool-base.git
cd nodetool-base
poetry install
```

This installs `nodetool-core` and other dependencies.

## Documentation

Detailed documentation for each node group lives in the [`docs/`](docs) folder. Start with [`docs/index.md`](docs/index.md) which lists all available node categories:

## Available Nodes

### AI & LLM Integrations
- **anthropic** - Claude AI models for chat, agents, and multimodal interactions
- **gemini** - Google Gemini for text-to-speech, image/video generation, and grounded search
- **kie** - Advanced media generation via Kie.ai (Suno music, Grok/Hailuo/Kling video)
- **openai** - OpenAI models for chat, agents, audio transcription, and image generation

### Core Node Types (nodetool.* - 21 modules)
- **agents** - Agent-based workflows and orchestration
- **audio** - Audio processing and manipulation
- **boolean** - Logical operators and comparisons
- **code** - Execute Python code snippets
- **compare** - Comparison operations
- **constant** - Constant value providers
- **control** - Flow control (if/else, branching)
- **data** - Data manipulation utilities
- **dictionary** - Dictionary/key-value operations
- **document** - Document processing
- **generators** - Data and content generation
- **image** - Image manipulation and processing
- **input** - User input collection nodes
- **list** - List processing utilities
- **model3d** - 3D model operations
- **numbers** - Numeric operations and math
- **output** - Output nodes for results
- **text** - Text processing and manipulation
- **triggers** - Event triggers
- **video** - Video processing and editing
- **workspace** - Workspace management

### Utility Libraries (lib.* - 28 modules)
- **beautifulsoup** - HTML/XML parsing
- **browser** - Web browser automation
- **date** - Date and time utilities
- **docx** - Word document processing
- **excel** - Excel file operations
- **grid** - Grid/table operations
- **http** - HTTP requests and APIs
- **json** - JSON parsing and manipulation
- **mail** - Email operations
- **markdown** - Markdown processing
- **markitdown** - Document to markdown conversion
- **math** - Mathematical operations
- **numpy** - NumPy array operations
- **ocr** - Optical character recognition
- **os** - File system and OS operations
- **pandoc** - Document format conversion
- **pdfplumber** - PDF text extraction
- **pillow** - Advanced image processing
- **pymupdf** - PDF manipulation
- **rss** - RSS feed parsing
- **seaborn** - Data visualization
- **secret** - Secret/credential management
- **sqlite** - SQLite database operations
- **supabase** - Supabase database integration
- **svg** - SVG graphics processing
- **text_utils** - Text utility functions
- **uuid** - UUID generation
- **ytdlp** - YouTube video download

### Messaging & Communication
- **discord** - Discord bot integration
- **telegram** - Telegram bot integration

### Search & Vector Databases
- **search** - Google search integration
- **vector** - Vector databases (Chroma, FAISS)

For detailed documentation on each node, see [`docs/index.md`](docs/index.md).

## Running tests

```bash
pytest -q
```

Most tests run without network access.

## Code Quality & Pre-Commit Hooks

This repository uses pre-commit hooks to ensure code quality and consistency.

**Setup:**

```bash
# Install pre-commit
pip install pre-commit

# Install git hooks
pre-commit install
```

**What gets checked:**
- Ruff linting and formatting
- Trailing whitespace and file endings
- YAML/JSON validation
- Markdown formatting

**Running manually:**

```bash
# Run all hooks on all files
pre-commit run --all-files

# Run on staged files only
pre-commit run
```

Hooks run automatically on `git commit`. If they fail or make changes, stage the changes and commit again.

## Contributing

Contributions are welcome. Feel free to open issues or pull requests on GitHub.

