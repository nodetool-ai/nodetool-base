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

## Initial Setup - 2026-01-15

Repository configured with OpenCode memory and automated Kie.ai sync workflow.

---

## 2026-01-17 - Wan 2.5 Video Generation & Ideogram V3 Image Generation

- Model ID: `wan/2-5-text-to-video`
- Category: `video`
- Description: Added Wan 2.5 Text-to-Video node supporting high-quality video generation from text descriptions with advanced motion and visual fidelity from Alibaba.

- Model ID: `wan/2-5-image-to-video`
- Category: `video`
- Description: Added Wan 2.5 Image-to-Video node supporting transformation of static images into dynamic videos with realistic motion and temporal consistency from Alibaba.

- Model ID: `ideogram/v3-text-to-image`
- Category: `image`
- Description: Added Ideogram V3 Text-to-Image node supporting high-quality image generation from text descriptions with improved text rendering, style consistency, and visual quality.

---

## 2026-01-15 - ElevenLabs Text-to-Speech

- Model ID: `elevenlabs/text-to-speech-turbo-2-5`
- Category: `audio`
- Description: Added ElevenLabs text-to-speech node supporting natural-sounding voice synthesis with multiple voices, stability controls, and multilingual output via Kie.ai API.
