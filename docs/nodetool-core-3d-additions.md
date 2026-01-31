# Nodetool-Core Additions Needed for 3D Generation

This document lists the nodetool-core updates needed to fully support the 3D generation refactor in nodetool-base.

## Provider Capabilities

- Add new provider capabilities:
  - `ProviderCapability.TEXT_TO_3D`
  - `ProviderCapability.IMAGE_TO_3D`
- Extend `BaseProvider._CAPABILITY_METHODS` to include `text_to_3d` and `image_to_3d`.
- Add abstract methods on `BaseProvider`:
  - `async def text_to_3d(self, params, timeout_s=None, context=None, node_id=None) -> bytes`
  - `async def image_to_3d(self, image: bytes, params, timeout_s=None, context=None, node_id=None) -> bytes`

## Provider Dispatch

- Update `ProcessingContext._dispatch_capability`:
  - Handle `TEXT_TO_3D` and call `provider.text_to_3d(params=..., context=...)`.
  - Handle `IMAGE_TO_3D` and call `provider.image_to_3d(image=..., params=..., context=...)`.
- Ensure `run_provider_prediction` supports the new capabilities (no other changes required beyond dispatch).

## Provider Implementations

- Add 3D methods to `HuggingFaceProvider` (or other provider backends as appropriate):
  - `text_to_3d`: call `AsyncInferenceClient.text_to_3d(...)`
  - `image_to_3d`: call `AsyncInferenceClient.image_to_3d(...)`
- Extend HuggingFace model discovery to include pipeline tags:
  - `text-to-3d`
  - `image-to-3d`
- Update `ProviderCapability` detection so `text_to_3d`/`image_to_3d` are included in `get_capabilities()`.

## Provider Types

- Add a `TextTo3DParams` and `ImageTo3DParams` type to `nodetool.providers.types` to match the new 3D generation nodes.

## Prediction Conversion

- Update `ProcessingContext.convert_value_for_prediction` to handle `Model3DRef`:
  - Use `model3d_ref_to_data_uri(...)` for 3D outputs if needed.
  - Allow `model_3d` inputs to be sent to providers as data URIs (similar to ImageRef/VideoRef).

## Environment/Secrets

- If provider-specific API keys are supported in core, add secrets mapping:
  - `MESHY_API_KEY`, `RODIN_API_KEY`, `TRELLIS_API_KEY`, `TRIPO_API_KEY`, `HUNYUAN3D_API_KEY`, `SHAP_E_API_KEY`, `POINT_E_API_KEY`

