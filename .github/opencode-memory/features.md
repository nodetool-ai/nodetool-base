# Features Log

This file tracks nodes and features added by the automated OpenCode agent.

## Format

Each entry should follow this format:
```
## YYYY-MM-DD - Feature/Node Name
- Model ID: `provider/model-name`
- Category: `image`, `video`, or `audio`
- Description: Brief summary of what was added
```

---

## 2026-01-17 - Bug Fix: Missing uuid Import

- **Issue**: The `uuid` module was used in `_resolve_upload_filename` method in `image.py` but was not imported.
- **Fix**: Added `import uuid` to `src/nodetool/nodes/kie/image.py`.
- **Models Scanned**: Attempted to discover new models from Kie.ai marketplace but no new models were identified for addition.

---

## Initial Setup - 2026-01-15

Repository configured with OpenCode memory and automated Kie.ai sync workflow.
