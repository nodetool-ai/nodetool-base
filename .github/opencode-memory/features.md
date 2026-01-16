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

## 2026-01-16 - GPT-4o Image, Runway, and Luma Models

### Image Nodes
- Model ID: `gpt4o-image`
- Category: `image`
- Added `Gpt4oTextToImage` - Generate images using OpenAI's GPT-4o Image model via Kie.ai
- Added `Gpt4oEdit` - Edit images using masks and prompts with GPT-4o
- Added `Gpt4oImageToImage` - Generate image variants from reference images

### Video Nodes
- Model ID: `runway`
- Category: `video`
- Added `RunwayTextToVideo` - Generate videos from text using Runway
- Added `RunwayImageToVideo` - Animate images into videos using Runway
- Added `RunwayExtendVideo` - Extend existing videos using Runway

### Video Nodes
- Model ID: `luma/modify`
- Category: `video`
- Added `LumaModifyVideo` - Modify and transform videos using Luma AI

---

## Initial Setup - 2026-01-15

Repository configured with OpenCode memory and automated Kie.ai sync workflow.
