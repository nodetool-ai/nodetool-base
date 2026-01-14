# Node Creation Guide (Kie.ai)

This guide explains how to add new Kie.ai models to the `nodetool-base` repository.

## Step 1: Discover New Models

Visit https://kie.ai/market and look for:
- Newly released or featured models
- Popular models with clear documentation
- Models not already present in `src/nodetool/nodes/kie/*`

## Step 2: Fetch API Docs (.md pages)

Each model has an API documentation page in Markdown. Use the model page URL and append `.md`.

Example:
- Model page: `https://kie.ai/model/seedream/4.5-text-to-image`
- API docs: `https://kie.ai/model/seedream/4.5-text-to-image.md`

Use the `.md` page to capture:
- The `model` identifier (used in `_get_model`)
- Required and optional input parameters
- Parameter types, defaults, enums, and constraints
- Any file inputs (image, audio, video) that need upload handling

## Step 3: Choose the Target Module

Pick the file that matches the modality:
- `src/nodetool/nodes/kie/image.py` for image generation/edit/upscale
- `src/nodetool/nodes/kie/video.py` for video generation
- `src/nodetool/nodes/kie/audio.py` for audio/music generation

## Step 4: Implement the Node

Follow the patterns in nearby classes:

1. **Class name**: PascalCase, descriptive of the model and mode (e.g., `Seedream45TextToImage`).
2. **Docstring**: short description plus keyword tags for search.
3. **Fields**: define inputs with `pydantic.Field`, using enums for constrained values.
4. **Model ID**: return the model identifier from the `.md` page in `_get_model`.
5. **Input params**: build the API payload in `_get_input_params`.
6. **Uploads**:
   - Use `_upload_image`, `_upload_audio`, or `_upload_video` when a parameter expects a file URL.
7. **Process**: return the correct type (`ImageRef`, `VideoRef`, or `AudioRef`).

Minimal template:

```python
class ExampleModel(KieBaseNode):
    """Short description.

    kie, provider, tags, category
    """

    _expose_as_tool: ClassVar[bool] = True

    prompt: str = Field(default="", description="Prompt...")

    def _get_model(self) -> str:
        return "provider/model-name"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        return {"prompt": self.prompt}

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_bytes, task_id = await self._execute_task(context)
        return await context.image_from_bytes(image_bytes, metadata={"task_id": task_id})
```

Video nodes should subclass `KieVideoBaseNode` and return `VideoRef`. Audio nodes subclass `KieBaseNode` and return `AudioRef`.

## Step 5: Regenerate Metadata and DSL

```bash
nodetool package scan
nodetool codegen
```

## Step 6: Quality Checks

```bash
ruff check .
black --check .
pytest -q
```

## Checklist

- [ ] Model appears on https://kie.ai/market
- [ ] `.md` API docs collected and parsed
- [ ] Node added to correct file (`image.py`, `video.py`, or `audio.py`)
- [ ] File inputs uploaded with correct helper
- [ ] Metadata/DSL regenerated
- [ ] Features log updated
