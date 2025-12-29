# nodetool-base

A collection of reusable nodes for [Nodetool](https://github.com/nodetool-ai/nodetool). These nodes implement common functionality for text, audio, video, images and more. They build on the runtime provided by [nodetool-core](https://github.com/nodetool-ai/nodetool-core).

## Overview

Nodetool lets you compose AI workflows as graphs. `nodetool-base` ships with a rich set of nodes so you can build useful flows out of the box. The nodes are organised into several namespaces under `src/nodetool/nodes`:

- `calendly` – access to Calendly events
- `chroma` – Chroma vector database operations
- `google` – Google image generation
- `lib` – helpers using libraries like BeautifulSoup, LlamaIndex or PyMuPDF
- `nodetool` – core utilities such as audio, boolean logic, image processing and more
- `openai` – wrappers around OpenAI APIs

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

- **[lib.audio](nodetool_audio.md)** - Save audio files to the assets directory.
- **[nodetool.boolean](nodetool_boolean.md)** - Logical operators, comparisons and flow control helpers.
- **[nodetool.code](nodetool_code.md)** - Evaluate expressions or run small Python snippets (development use).
- **[nodetool.constant](nodetool_constant.md)** - Provide constant values like numbers, strings and images.
- **[nodetool.control](nodetool_control.md)** - Basic branching with an if node.
- **[nodetool.date](nodetool_date.md)** - Utilities for manipulating dates and times.
- **[nodetool.dictionary](nodetool_dictionary.md)** - Manipulate key/value data and dictionaries.
- **[nodetool.group](nodetool_group.md)** - Group operations such as looping over inputs.
- **[nodetool.image](nodetool_image.md)** - Image manipulation including crop, resize and save.
- **[nodetool.input](nodetool_input.md)** - Nodes for collecting user input of various types.
- **[nodetool.json](nodetool_json.md)** - Parse, query and validate JSON data.
- **[nodetool.list](nodetool_list.md)** - List processing utilities.
- **[nodetool.math](nodetool_math.md)** - Basic arithmetic and math functions.
- **[nodetool.os](nodetool_os.md)** - File system and path helpers.
- **[nodetool.output](nodetool_output.md)** - Output nodes to return results to the user.
- **[nodetool.text](nodetool_text.md)** - Text processing nodes with regex and templating.
- **[nodetool.video](nodetool_video.md)** - Video editing and generation tools.

Refer to the individual markdown files for usage details and examples of each node.

## Manifesto Alignment

This project follows the [NodeTool Manifesto](https://github.com/nodetool-ai/nodetool) principles of privacy-first, local-first AI workflows. See [MANIFESTO_EVALUATION.md](MANIFESTO_EVALUATION.md) for a comprehensive evaluation of how the current node offering aligns with these principles.

**Key Stats:**
- 600+ nodes across 72 modules
- 35+ AI providers (local and cloud)
- 70+ workflow examples
- Overall alignment score: 8/10

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

