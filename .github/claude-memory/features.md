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

## 2026-01-16 - Runway & Luma Video Models

- **Model IDs:**
  - `runway/gen-3-alpha-text-to-video`
  - `runway/gen-3-alpha-image-to-video`
  - `runway/gen-3-alpha-extend-video`
  - `runway/generate-aleph-video`
  - `luma/generate-luma-modify-video`

- **Category:** video

- **Description:** Added support for Runway Gen-3 Alpha video generation (text-to-video, image-to-video, extend-video) and Aleph model, plus Luma's video modification API via Kie.ai.

---

## Initial Setup - 2026-01-15

Repository configured with OpenCode memory and automated Kie.ai sync workflow.

---

## 2026-01-15 - ElevenLabs Text-to-Speech

- Model ID: `elevenlabs/text-to-speech-turbo-2-5`
- Category: `audio`
- Description: Added ElevenLabs text-to-speech node supporting natural-sounding voice synthesis with multiple voices, stability controls, and multilingual output via Kie.ai API.

---

## 2026-03-05 - Qwen Image Edit Node

- **Model ID:** `qwen/image-edit`

- **Category:** image

- **Description:** Added Qwen Image Edit node supporting both semantic and appearance editing with precise, visually coherent results. Features include:
  - Semantic edits (changing objects, backgrounds, etc.)
  - Appearance modifications (style, colors, textures)
  - Bilingual (Chinese and English) text editing with preserved typography
  - Multiple acceleration modes (none, regular, high)
  - Configurable image sizes and aspect ratios
  - Adjustable inference steps and guidance scale
  - Multiple output formats (JPEG, PNG)
  - Support for generating 1-4 images at once
  - Optional negative prompts and seed control

---

## 2026-02-19 - Ideogram Character Nodes

- **Model IDs:**
  - `ideogram/character`
  - `ideogram/character-edit`

- **Category:** image

- **Description:** Added two new Ideogram character generation nodes:
  - `IdeogramCharacter`: Generate character images in various settings while maintaining character consistency using reference images and text prompts
  - `IdeogramCharacterEdit`: Edit masked parts of character images with inpainting while maintaining character consistency

---
