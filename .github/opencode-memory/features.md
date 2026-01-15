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

## 2026-01-15 - GPT Image 1.5 Nodes Added

- Model ID: `gpt-image-1-5/text-to-image`, `gpt-image-1-5/image-to-image`
- Category: `image`
- Description: Added two new nodes for OpenAI's GPT Image 1.5 model via Kie.ai:
  - **GPTImage15TextToImage**: Text-to-image generation with support for multiple sizes (256x256 to 1792x1024), quality settings (standard/hd), and styles (natural/vivid)
  - **GPTImage15ImageToImage**: Image-to-image transformation with strength control, supporting the same size/quality/style options

**Discovery Process:**
- Found documentation at `https://docs.kie.ai` (4o Image API and Suno API pages)
- Identified GPT Image 1.5 as a new model not previously implemented
- Model endpoints: `https://api.kie.ai/api/v1/gpt4o-image/generate`
- Parameters: prompt, n, size, quality, style, strength (for image-to-image)

**Implementation Details:**
- Added to `src/nodetool/nodes/kie/image.py` following existing patterns
- Uses `KieBaseNode` as base class
- Supports aspect ratios: 256x256, 512x512, 1024x1024, 1024x1792, 1792x1024
- Quality options: standard, hd
- Style options: natural, vivid
- Image transformation strength: 0.0 to 1.0

**Other Models Discovered (Not Implemented):**
- Elevenlabs audio models (audio-isolation, sound-effect-v2)
- Runway Gen-4 models (via Aleph API)
- Suno V5 model (newer version)
- GPT-4o Image (GPT IMAG 1)

---

## 2026-01-15 - Discovery Attempt

**Status:** Unable to access Kie.ai model documentation

**Issue:** All Kie.ai model documentation URLs return 404 errors:
- `https://kie.ai/model/seedream/4.5-text-to-image.md`
- `https://kie.ai/model/flux-3/text-to-image.md`
- `https://kie.ai/model/kling-3/text-to-video.md`
- And all other tested model documentation URLs

**Analysis:**
- Existing implementation covers comprehensive set of models:
  - **Video (28 models):** Kling 2.6/2.5, Hailuo 2.3, Seedance V1, Sora 2, Wan 2.6, Veo 3.1, Grok Imagine, Infinitalk, Topaz
  - **Image (22 models):** Flux 2, Seedream 4.5, Z-Image, Nano Banana/Pro, Flux Kontext, Grok Imagine, Qwen, Ideogram, Recraft, Google Imagen 4
  - **Audio (1 model):** Suno

**Recommendation:**
- Model documentation may require authentication or have changed URL format
- Consider using the Kie.ai API directly to list available models
- Manual review of https://kie.ai/market may be needed
- The automated workflow may need API credentials for discovery

---

## Initial Setup - 2026-01-15

Repository configured with OpenCode memory and automated Kie.ai sync workflow.
