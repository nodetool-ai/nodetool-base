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

## 2026-01-15 - Kie.ai Model Sync Check

- Reviewed Kie.ai marketplace at https://kie.ai/market for new models
- Analyzed existing implementation in `src/nodetool/nodes/kie/`
- Found that all major available models are already implemented:
  - **Image models** (23+): Flux 2 (Pro/Flex), Flux Kontext, Seedream 4.5, Z-Image Turbo, Nano Banana, Nano Banana Pro, Nano Banana Edit, Grok Imagine, Qwen, Topaz Image Upscaler, Recraft (Crisp Upscale, Remove Background), Ideogram (Character Remix, V3 Reframe), Imagen 4, Imagen 4 Fast, Imagen 4 Ultra
  - **Video models** (23+): Kling 2.6, Kling AI Avatar, Grok Imagine, Seedance V1 (Lite/Pro), Hailuo 2.3, Kling 2.5 Turbo, Sora 2, Wan 2.1/2.6, Topaz Video Upscaler, Infinitalk V1, Veo 3.1
  - **Audio models**: Suno
- Verified code quality: All ruff and black checks pass
- No new model implementations needed this cycle
- Repository is comprehensive and up-to-date with available Kie.ai APIs
