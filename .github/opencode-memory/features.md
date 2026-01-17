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

## 2026-01-17 - Model Sync Review

Performed comprehensive review of available Kie.ai models. Found extensive coverage with 47 unique model IDs already implemented across:

**Image Generation (23 models):**
- Flux 2 Pro/Flex (text-to-image, image-to-image)
- Seedream 4.5 (text-to-image, edit)
- Z-Image Turbo
- Nano Banana/Pro/Edit (Google Gemini variants)
- Flux Kontext
- Grok Imagine (text-to-image, upscale)
- Qwen (text-to-image, image-to-image)
- Topaz Image Upscale
- Recraft (remove-background, crisp-upscale)
- Ideogram (character-remix, v3-reframe)
- Google Imagen 4 (standard, fast, ultra)

**Video Generation (23 models):**
- Kling 2.6/2.5 Turbo (text-to-video, image-to-video)
- Kling AI Avatar (standard, pro)
- Grok Imagine (text-to-video, image-to-video)
- Seedance V1 Lite/Pro (text-to-video, image-to-video)
- Seedance V1 Pro Fast Image-to-Video
- Hailuo 2.3 Pro/Standard (text-to-video, image-to-video)
- Sora 2 Pro/Standard (text-to-video, image-to-video, storyboard)
- Wan 2.1/2.6 (text-to-video, image-to-video, video-to-video)
- Topaz Video Upscale
- Infinitalk V1
- Google Veo 3.1 (text-to-video, image-to-video, reference-to-video)

**Audio Generation (1 model):**
- Suno Music Generator

No new models found to add in this sync cycle. Marketplace appears fully covered.
