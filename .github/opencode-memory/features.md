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

## 2026-01-20 - GPT Image 1.5 & Google Imagen 4 Models

- **Model IDs:**
  - `gpt-image/1.5-text-to-image`
  - `gpt-image/1.5-image-to-image`
  - `google/imagen4`
  - `google/imagen4-fast`
  - `google/imagen4-ultra`
  - `recraft/crisp-upscale`

- **Category:** image

- **Description:** Added support for OpenAI's GPT Image 1.5 models (text-to-image and image-to-image), Google's Imagen 4 family (standard, fast, and ultra variants), and Recraft Crisp Upscale for AI-powered image enhancement via Kie.ai API.

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
