# Repository Context

## Overview

`nodetool-base` provides the essential node collection for the NodeTool platform. This memory is focused on Kie.ai nodes.

## Directory Structure

```
src/nodetool/
├── nodes/kie/                 # Kie.ai node implementations
│   ├── image.py               # Kie image generation/edit nodes + KieBaseNode
│   ├── video.py               # Kie video generation nodes + KieVideoBaseNode
│   └── audio.py               # Kie audio/music generation nodes
├── dsl/kie/                   # Generated DSL wrappers for Kie nodes
└── package_metadata/          # Generated package metadata (nodetool-base.json)
```

## Key Files

### `src/nodetool/nodes/kie/image.py`
- Defines `KieBaseNode`, shared by image and audio nodes.
- Includes upload helpers (`_upload_image`, `_upload_audio`, `_upload_video`).
- Each model node implements `_get_model`, `_get_input_params`, and `process`.

### `src/nodetool/nodes/kie/video.py`
- Defines `KieVideoBaseNode` for video nodes.
- Video nodes typically return `VideoRef` from `process`.

### `src/nodetool/nodes/kie/audio.py`
- Audio nodes subclass `KieBaseNode` and return `AudioRef`.

## Generated Outputs

After adding new nodes, regenerate:

```bash
nodetool package scan
nodetool codegen
```

These commands update:
- `src/nodetool/package_metadata/nodetool-base.json`
- `src/nodetool/dsl/kie/*.py`

## Secrets

- `KIE_API_KEY` is required to call the Kie.ai API.

## Quality Checks

```bash
ruff check .
black --check .
pytest -q
```
