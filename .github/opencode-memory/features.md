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

## 2026-01-22 - Google Imagen 4 Image Generation Models

- **Model IDs:**
  - `google/imagen4`
  - `google/imagen4-fast`
  - `google/imagen4-ultra`

- **Category:** `image`

- **Description:** Added support for Google's Imagen 4 image generation models via Kie.ai API. The standard Imagen 4 model provides high-quality photorealistic images with excellent detail and text rendering. Imagen 4 Fast offers faster generation with support for multiple images (1-4) per request. Imagen 4 Ultra provides the highest quality variant for professional production use. All variants support configurable aspect ratios (1:1, 16:9, 9:16, 4:3, 3:4), negative prompts, and seed values for reproducibility.

---

## Initial Setup - 2026-01-15

Repository configured with OpenCode memory and automated Kie.ai sync workflow.

---

## 2026-01-15 - ElevenLabs Text-to-Speech

- Model ID: `elevenlabs/text-to-speech-turbo-2-5`
- Category: `audio`
- Description: Added ElevenLabs text-to-speech node supporting natural-sounding voice synthesis with multiple voices, stability controls, and multilingual output via Kie.ai API.
