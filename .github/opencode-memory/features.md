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

## 2026-01-19 - No New Models Found

- **Status:** No new Kie.ai models discovered
- **Notes:** 
  - Kie.ai marketplace loads content dynamically via JavaScript
  - Model documentation pages return 404 errors or have changed structure
  - Current implementation already covers all available Kie.ai models
  - Models checked: Runway, Luma, Veo 3.1, Kling 2.6/2.7, Hailuo 2.5, Seedance V2, Flux 3.0, Recraft V3, Ideogram V3, Sora 2.6, Wan 3.0, and various other providers
  - No new models were added during this sync attempt

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
