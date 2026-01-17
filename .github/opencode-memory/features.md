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

## 2026-01-15 - ElevenLabs Text-to-Speech

- Model ID: `elevenlabs/text-to-speech-turbo-2-5`
- Category: `audio`
- Description: Added ElevenLabs text-to-speech node supporting natural-sounding voice synthesis with multiple voices, stability controls, and multilingual output via Kie.ai API.

---

## 2026-01-17 - Kie Model Sync (No New Models)

- Model ID: `N/A`
- Category: `maintenance`
- Description: Completed scheduled sync run. All available Kie.ai models are already implemented in the codebase. Verified existing implementations for Google Imagen 4 (standard, fast, ultra variants) and Google Nano Banana Edit models. No new models were found to add during this cycle.
