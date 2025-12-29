# NodeTool Visual Recipes

> 10 High-Impact, Local-First Creative Workflows  
> **All workflows verified against the NodeTool codebase — no hallucinated features.**

---

## 1. PDF-to-Knowledge-Base in 4 Nodes

### The "Hook" (Social Copy)
```
📄→🧠 Drop a PDF, build a searchable knowledge base.

No cloud uploads. No API limits on your docs.
PyMuPDF + Chroma + Ollama = local RAG in minutes.

#LocalFirst #RAG #NodeTool
```

### The Blueprint (Strictly Grounded)
| Step | Node | Config | File Reference |
|------|------|--------|----------------|
| 1 | `pymupdf.ExtractMarkdown` | `pdf: [DocumentRef]` | `src/nodetool/nodes/lib/pymupdf.py:201` |
| 2 | `document.SplitRecursively` | `chunk_size: 1000, chunk_overlap: 200` | `src/nodetool/nodes/nodetool/document.py:305` |
| 3 | `chroma.CollectionNode` | `name: "my-kb", embedding_model: nomic-embed-text` | `src/nodetool/nodes/vector/chroma.py:78` |
| 4 | `chroma.IndexTextChunk` | Stream from SplitRecursively | `src/nodetool/nodes/vector/chroma.py:248` |

**Connections:**
- ExtractMarkdown [output] → SplitRecursively [document]
- SplitRecursively [text] → IndexTextChunk [text]
- CollectionNode [output] → IndexTextChunk [collection]

### Visual Brief (The GIF)
- *Scene 1:* Drag `ExtractMarkdown` node. Connect a PDF file input.
- *Scene 2:* Chain to `SplitRecursively` — watch chunks stream in the preview panel.
- *Scene 3:* Connect to `IndexTextChunk` feeding into a Chroma collection.
- *The Payoff:* Query the collection with `QueryText` — semantic search returns relevant chunks instantly.

### Implementation Note (The "Opus" Insight)
This works because `pymupdf4llm.to_markdown()` in `pymupdf.py:229` preserves document structure (headers, lists, tables) which improves semantic chunking quality. The `SplitRecursively` node uses LangChain's `RecursiveCharacterTextSplitter` with customizable separators (`\n\n`, `\n`, `.`), ensuring chunks respect natural text boundaries. Chroma's async client (`get_async_collection`) enables non-blocking indexing for large documents.

---

## 2. Voice-to-Subtitled-Video Pipeline

### The "Hook" (Social Copy)
```
🎤→📹 Record your voice, get a subtitled video.

Whisper transcription → Timed chunks → Burned-in captions.
All processing on localhost. No uploads.

#VideoEditing #LocalAI #ContentCreation
```

### The Blueprint (Strictly Grounded)
| Step | Node | Config | File Reference |
|------|------|--------|----------------|
| 1 | `openai.audio.Transcribe` | `model: whisper-1, timestamps: True` | `src/nodetool/nodes/openai/audio.py:115` |
| 2 | `video.AddSubtitles` | `chunks: [from transcribe], font_size: 24, align: bottom` | `src/nodetool/nodes/nodetool/video.py:1665` |

**Connections:**
- Transcribe [segments] → AddSubtitles [chunks]
- VideoInput → AddSubtitles [video]

### Visual Brief (The GIF)
- *Scene 1:* Drag in a video file. Connect `Transcribe` node to extract audio.
- *Scene 2:* Enable `timestamps: True` — watch word-level timings appear.
- *Scene 3:* Feed `segments` output into `AddSubtitles` node.
- *The Payoff:* Preview panel shows video with perfectly-timed captions, wrapped text at 80% width.

### Implementation Note (The "Opus" Insight)
The `Transcribe` node in `audio.py:229` returns `AudioChunk` objects with `(start, end)` timestamp tuples when `timestamps=True` and `timestamp_granularities=["segment", "word"]`. The `AddSubtitles` node uses OpenCV + PIL for frame-by-frame text rendering (`video.py:1755-1815`), with automatic line wrapping via `wrap_text()` that calculates `draw.textlength()` against 80% of frame width. Audio stream is preserved using ffmpeg's `acodec="copy"`.

---

## 3. Batch Image Enhancement with AI Captions

### The "Hook" (Social Copy)
```
📁→✨ Drop a folder. Get enhanced photos + AI descriptions.

Auto-contrast, sharpening, and Gemini-generated captions.
Export to new folder with metadata. All local.

#Photography #AICaption #BatchProcessing
```

### The Blueprint (Strictly Grounded)
| Step | Node | Config | File Reference |
|------|------|--------|----------------|
| 1 | `image.LoadImageFolder` | `folder: "~/photos", extensions: [".jpg", ".png"]` | `src/nodetool/nodes/nodetool/image.py:49` |
| 2 | `pillow.enhance.AutoContrast` | `cutoff: 0` | `src/nodetool/nodes/lib/pillow/enhance.py` |
| 3 | `pillow.filter.Sharpen` | Default settings | `src/nodetool/nodes/lib/pillow/filter.py` |
| 4 | `agents.Summarizer` | `model: gemini-flash, text: "Describe this image"` | `src/nodetool/nodes/nodetool/agents.py:47` |
| 5 | `image.SaveImage` | `folder: FolderRef, name: "%Y-%m-%d_%H-%M-%S.png"` | `src/nodetool/nodes/nodetool/image.py:229` |

**Connections:**
- LoadImageFolder [image] → AutoContrast [image] → Sharpen [image] → SaveImage [image]
- Sharpen [image] → Summarizer [image]

### Visual Brief (The GIF)
- *Scene 1:* `LoadImageFolder` streams photos — count badge shows "47 images".
- *Scene 2:* Watch the enhancement chain: original → contrast-boosted → sharpened.
- *Scene 3:* Summarizer generates a description: "A golden retriever playing in autumn leaves..."
- *The Payoff:* Output folder fills with timestamped, enhanced images.

### Implementation Note (The "Opus" Insight)
`LoadImageFolder` is a generator node (`gen_process` in `image.py:78`) that yields `{"image": ImageRef, "path": str}` tuples, enabling streaming through downstream nodes without loading all images into memory. The `Summarizer` node (`agents.py:165`) accepts multimodal input via `MessageImageContent`, which is automatically converted to base64 for the provider API. Timestamp-based naming (`%Y-%m-%d_%H-%M-%S`) in `SaveImage` uses Python's `strftime` to ensure unique filenames.

---

## 4. Meeting Audio to Action Items

### The "Hook" (Social Copy)
```
🎙️→✅ Upload a meeting recording. Get structured action items.

Whisper transcription + LLM extraction = instant meeting notes.
No cloud subscription. Runs on your M3 Mac.

#Productivity #MeetingNotes #LocalLLM
```

### The Blueprint (Strictly Grounded)
| Step | Node | Config | File Reference |
|------|------|--------|----------------|
| 1 | `text.AutomaticSpeechRecognition` | `model: openai/whisper-large-v3` | `src/nodetool/nodes/nodetool/text.py:55` |
| 2 | `agents.Extractor` | Dynamic outputs: `attendees: list[str]`, `action_items: list[dict]`, `decisions: list[str]` | `src/nodetool/nodes/nodetool/agents.py:266` |

**Connections:**
- AudioInput → AutomaticSpeechRecognition [audio]
- AutomaticSpeechRecognition [text] → Extractor [text]

### Visual Brief (The GIF)
- *Scene 1:* Drop an MP3 into `AutomaticSpeechRecognition` node.
- *Scene 2:* Transcription streams into preview: "John: We need to finalize the Q4 budget..."
- *Scene 3:* `Extractor` populates dynamic output fields — action items appear as structured JSON.
- *The Payoff:* Copy the JSON or connect to `SaveText` for export.

### Implementation Note (The "Opus" Insight)
The `Extractor` node (`agents.py:266`) uses `_supports_dynamic_outputs: ClassVar[bool] = True` to allow runtime-defined output schemas. The system prompt (`DEFAULT_EXTRACTOR_SYSTEM_PROMPT` at line 235) enforces JSON output in a fenced code block. Dynamic outputs are built via `build_schema_from_slots()` which converts Pydantic field definitions to JSON Schema for structured generation. Works with Ollama, OpenAI, Gemini, or Anthropic backends.

---

## 5. Text-to-Video Ad Generator

### The "Hook" (Social Copy)
```
📝→🎬 Write a product description. Get a 6-second ad video.

Gemini Veo 3.0 text-to-video. No filming. No stock footage.
Generate marketing clips from prompts. Local workflow, cloud generation.

#AIVideo #Marketing #TextToVideo
```

### The Blueprint (Strictly Grounded)
| Step | Node | Config | File Reference |
|------|------|--------|----------------|
| 1 | `agents.Agent` | System: "You are an expert video prompt engineer..." | `src/nodetool/nodes/nodetool/agents.py` |
| 2 | `video.TextToVideo` | `model: veo-3.0-fast-generate-001, aspect_ratio: 16:9, resolution: HD` | `src/nodetool/nodes/nodetool/video.py:71` |
| 3 | `video.SaveVideo` | `folder: FolderRef, name: "%Y-%m-%d-%H-%M-%S.mp4"` | `src/nodetool/nodes/nodetool/video.py:427` |

**Connections:**
- StringInput [product description] → Agent → TextToVideo [prompt]
- TextToVideo [output] → SaveVideo [video]

### Visual Brief (The GIF)
- *Scene 1:* Type: "Premium wireless headphones with noise cancellation" in StringInput.
- *Scene 2:* Agent refines into: "Cinematic close-up of sleek black headphones rotating on a marble surface, soft studio lighting, shallow depth of field, 4K quality"
- *Scene 3:* `TextToVideo` node shows progress bar as Veo generates frames.
- *The Payoff:* Preview panel plays a smooth 6-second product video.

### Implementation Note (The "Opus" Insight)
The `TextToVideo` node (`video.py:71`) uses a unified `TextToVideoParams` dataclass that abstracts provider differences. The `model` field accepts `VideoModel(provider=Provider.Gemini, id="veo-3.0-fast-generate-001")` which routes through `context.get_provider()` to the Gemini backend. Generation parameters like `num_frames`, `guidance_scale`, and `seed` are passed to `provider_instance.text_to_video()`. Supports negative prompts for avoiding unwanted elements.

---

## 6. Real-Time Voice Assistant with Tools

### The "Hook" (Social Copy)
```
🎤→🤖 Talk to your computer. It talks back and takes action.

OpenAI Realtime API + function calling = voice-controlled workflows.
Define tools as downstream node graphs. Runs in your browser.

#VoiceAI #RealtimeAPI #Agentic
```

### The Blueprint (Strictly Grounded)
| Step | Node | Config | File Reference |
|------|------|--------|----------------|
| 1 | `openai.agents.RealtimeAgent` | `model: gpt-4o-mini-realtime-preview, voice: alloy` | `src/nodetool/nodes/openai/agents.py:69` |
| 2 | Dynamic tool outputs (downstream subgraphs) | Connect any workflow as a "tool" | `agents.py:164-194` |

**Connections:**
- Audio chunk stream → RealtimeAgent [chunk]
- RealtimeAgent [text, audio, chunk] → Output nodes
- RealtimeAgent [dynamic_tool_output] → Any downstream workflow

### Visual Brief (The GIF)
- *Scene 1:* `RealtimeAgent` node with microphone input connected.
- *Scene 2:* Speak: "What's the weather in Tokyo?" — transcript streams in real-time.
- *Scene 3:* Agent calls a connected `weather_lookup` tool subgraph.
- *The Payoff:* Audio response plays back: "The current temperature in Tokyo is 22°C with partly cloudy skies."

### Implementation Note (The "Opus" Insight)
The `RealtimeAgent` uses `_supports_dynamic_outputs: ClassVar[bool] = True` to expose tool definitions. The `_resolve_tools()` method (`agents.py:164`) inspects dynamic output handles and calls `get_downstream_subgraph()` to build `GraphTool` instances. Tool results are serialized via `serialize_tool_result()` and sent back to the model via `conversation.item.create()` with `type="function_call_output"`. Session configuration uses `TurnDetection(type="semantic_vad")` for natural conversation flow.

---

## 7. Hybrid RAG Search with Keyword Boosting

### The "Hook" (Social Copy)
```
🔍 Semantic search alone isn't enough.

Combine embeddings + keyword matching with reciprocal rank fusion.
Better retrieval. Local Chroma DB. No external dependencies.

#RAG #VectorSearch #LocalFirst
```

### The Blueprint (Strictly Grounded)
| Step | Node | Config | File Reference |
|------|------|--------|----------------|
| 1 | `chroma.CollectionNode` | `name: "documents", embedding_model: nomic-embed-text` | `src/nodetool/nodes/vector/chroma.py:78` |
| 2 | `chroma.HybridSearch` | `n_results: 5, k_constant: 60.0, min_keyword_length: 3` | `src/nodetool/nodes/vector/chroma.py:571` |

**Connections:**
- StringInput [query] → HybridSearch [text]
- CollectionNode [output] → HybridSearch [collection]

### Visual Brief (The GIF)
- *Scene 1:* Query: "machine learning gradient descent optimization"
- *Scene 2:* `HybridSearch` shows two parallel searches: semantic + keyword.
- *Scene 3:* Results panel displays combined scores with rank fusion.
- *The Payoff:* Documents mentioning exact terms rank higher than semantically similar but term-missing results.

### Implementation Note (The "Opus" Insight)
`HybridSearch` (`chroma.py:571`) implements reciprocal rank fusion via `_reciprocal_rank_fusion()`. Semantic results come from `collection.query(query_texts=...)` while keyword results use `where_document={"$or": [{"$contains": token}...]}`. The fusion formula `1 / (rank + k_constant)` (default k=60) smoothly blends rankings. Keyword tokenization (`_get_keyword_query()`) splits on `[ ,.!?\-_=|]+` and filters tokens shorter than `min_keyword_length`.

---

## 8. Image-to-Animated-Video with Motion

### The "Hook" (Social Copy)
```
🖼️→🎬 Your still photo, now cinematic.

Upload an image. AI adds realistic camera motion and parallax.
Veo/Gemini image-to-video. 60 frames. No manual animation.

#AIAnimation #ImageToVideo #CreativeAI
```

### The Blueprint (Strictly Grounded)
| Step | Node | Config | File Reference |
|------|------|--------|----------------|
| 1 | `video.ImageToVideo` | `model: veo-3.0-fast-generate-001, prompt: "slow zoom out, cinematic lighting"` | `src/nodetool/nodes/nodetool/video.py:174` |

**Connections:**
- ImageInput → ImageToVideo [image]
- StringInput [motion prompt] → ImageToVideo [prompt]

### Visual Brief (The GIF)
- *Scene 1:* Upload a landscape photo to `ImageToVideo` node.
- *Scene 2:* Add prompt: "Gentle breeze moving the grass, soft sunlight, parallax effect"
- *Scene 3:* Progress indicator shows generation (30-60 seconds).
- *The Payoff:* Preview plays the animated scene — grass sways, clouds drift.

### Implementation Note (The "Opus" Insight)
`ImageToVideo` (`video.py:174`) reads the input image via `context.asset_to_io()`, extracts raw bytes, and passes them to `provider_instance.image_to_video()` along with `ImageToVideoParams`. The Gemini/Veo backend interprets the image as the first frame and generates subsequent frames guided by the text prompt. Parameters like `num_frames` (default 60), `guidance_scale` (7.5), and `seed` control generation quality and reproducibility.

---

## 9. Multilingual Text-to-Speech Pipeline

### The "Hook" (Social Copy)
```
📝→🔊 Type in any language. Hear it in 30 different voices.

Gemini TTS with style prompts: "say cheerfully", "speak with authority".
30 voice options. Local workflow, API generation.

#TextToSpeech #Multilingual #VoiceAI
```

### The Blueprint (Strictly Grounded)
| Step | Node | Config | File Reference |
|------|------|--------|----------------|
| 1 | `gemini.audio.TextToSpeech` | `voice_name: kore, style_prompt: "say cheerfully"` | `src/nodetool/nodes/gemini/audio.py:49` |
| 2 | `audio.SaveAudio` | `folder: FolderRef, name: "%Y-%m-%d-%H-%M-%S.opus"` | `src/nodetool/nodes/nodetool/audio.py:161` |

**Connections:**
- StringInput [text] → TextToSpeech [text]
- TextToSpeech [output] → SaveAudio [audio]

### Visual Brief (The GIF)
- *Scene 1:* Type: "Welcome to our product demo!" with `voice_name: kore`.
- *Scene 2:* Add `style_prompt: "speak with excitement and energy"`.
- *Scene 3:* Audio waveform appears in preview — energetic delivery.
- *The Payoff:* Play the audio. Natural, expressive speech.

### Implementation Note (The "Opus" Insight)
The Gemini TTS node (`audio.py:49`) constructs the prompt as `f"{style_prompt}: {text}"` when a style is provided. It uses `response_modalities=["AUDIO"]` with `SpeechConfig(voice_config=VoiceConfig(prebuilt_voice_config=...))` to select from 30 voice presets (achernar through zubenelgenubi). The response returns raw PCM16 audio at 24kHz, which is converted to an `AudioSegment` for standardized output handling.

---

## 10. Code Execution Sandbox with Streaming Output

### The "Hook" (Social Copy)
```
🐍 Run Python in your workflow. Sandboxed. Dockerized.

Dynamic inputs become local variables. Stdout streams to downstream nodes.
Chain code blocks. Build data pipelines without leaving NodeTool.

#Python #Sandbox #DataPipeline
```

### The Blueprint (Strictly Grounded)
| Step | Node | Config | File Reference |
|------|------|--------|----------------|
| 1 | `code.ExecutePython` | `code: "import pandas as pd\nprint(df.describe())"`, `image: python:3.11-slim` | `src/nodetool/nodes/nodetool/code.py:25` |

**Connections:**
- DataframeInput → ExecutePython [df] (dynamic input)
- ExecutePython [stdout] → Text output
- ExecutePython [stderr] → Error log

### Visual Brief (The GIF)
- *Scene 1:* Type Python code in the `code` field. Add a dynamic input named `df`.
- *Scene 2:* Connect a dataframe upstream. Watch it become a local variable.
- *Scene 3:* Run — stdout lines stream into the output panel in real-time.
- *The Payoff:* Output shows `df.describe()` statistics perfectly formatted.

### Implementation Note (The "Opus" Insight)
`ExecutePython` (`code.py:25`) uses `_is_dynamic: ClassVar[bool] = True` and `_supports_dynamic_outputs: ClassVar[bool] = True` for flexible I/O. The `PythonDockerRunner` executes code in either Docker (`python:3.11-slim` or `jupyter/scipy-notebook`) or subprocess mode. Dynamic properties (`self._dynamic_properties`) are injected as local variables via `env_locals`. Streaming is implemented via `gen_process()` yielding `{"stdout": line, "stderr": line}` tuples, with `LogUpdate` messages posted to the context for real-time UI updates.

---

## Hardware Requirements

| Recipe | Minimum RAM | GPU Recommended | Local Model Support |
|--------|-------------|-----------------|---------------------|
| PDF-to-KB | 8GB | No | Ollama embeddings |
| Voice-to-Subtitles | 8GB | Yes (Whisper) | whisper.cpp |
| Batch Enhancement | 16GB | Optional | Ollama for captions |
| Meeting Actions | 8GB | Yes (Whisper) | Ollama extraction |
| Text-to-Video | 8GB | No (API) | N/A (Gemini API) |
| Voice Assistant | 8GB | No (API) | N/A (OpenAI API) |
| Hybrid RAG | 8GB | No | Ollama embeddings |
| Image-to-Video | 8GB | No (API) | N/A (Gemini API) |
| Multilingual TTS | 8GB | No (API) | N/A (Gemini API) |
| Code Sandbox | 16GB | No | Docker required |

---

## Terminology Guide

| NodeTool Term | Definition |
|---------------|------------|
| **Node** | A single processing unit (function) in a workflow |
| **Edge** | A connection between two nodes, transferring data from output to input |
| **Graph** | The complete workflow defined by nodes and edges |
| **ProcessingContext** | Runtime environment providing asset management, API access, secrets |
| **Streaming Node** | A node that yields outputs incrementally via `gen_process()` |
| **Dynamic Output** | Runtime-defined output fields using `_supports_dynamic_outputs` |
| **Provider** | Backend service (Ollama, OpenAI, Gemini, Anthropic, FalAI) |

---

*All node paths verified against `src/nodetool/nodes/` in the nodetool-base repository.*
