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

## 2026-01-20 - GPT Image 1.5 & Seedance 1.5 Pro Models

- **Model IDs:**
  - `gpt-image/1.5-text-to-image`
  - `gpt-image/1.5-image-to-image`
  - `bytedance/seedance-1.5-pro` (text-to-video and image-to-video)

- **Categories:** image, video

- **Description:** Added OpenAI GPT Image 1.5 support for text-to-image and image-to-image generation with quality and aspect ratio controls. Added Bytedance Seedance 1.5 Pro for advanced video generation with dynamic camera movements and optional audio generation.

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
