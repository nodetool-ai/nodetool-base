# NodeTool Base Nodes: Manifesto Alignment Evaluation

**Date:** December 2025  
**Version:** 0.6.2-rc.17

## Executive Summary

This document evaluates the current NodeTool base nodes offering against the principles outlined in the NodeTool Manifesto. The analysis assesses alignment across seven key manifesto principles and provides recommendations for improving adherence to the stated vision.

### Overall Alignment Score: 8/10

**Strengths:**
- Strong foundation of local-first capabilities with Ollama and ComfyUI integration
- Comprehensive mix of local and cloud services (35+ providers)
- Rich utility node library for data manipulation (600+ nodes across 72 modules)
- Transparent workflow execution model with real-time streaming
- Excellent "mix your tools" philosophy - seamlessly combine local and cloud AI

**Key Gaps:**
- Native local image generation nodes (though ComfyUI provides alternative)
- No native local audio transcription (Whisper mentioned but only via OpenAI)
- Limited explicit privacy controls in node design
- Missing batch processing nodes
- No version control or workflow template nodes
- No export/deployment infrastructure

---

## Manifesto Principle Analysis

### 1. "Build visually. Connect blocks, not code. Accessible to everyone."

**Status:** ✅ **FULLY ALIGNED**

**Evidence:**
- All nodes inherit from `BaseNode` with standardized visual interface
- 600+ nodes across 72 modules in 10 categories provide comprehensive building blocks
- Clear input/output typing for visual connections
- Rich node categories:
  - `nodetool.constant` - Constant values
  - `nodetool.control` - Flow control (if/else, ForEach)
  - `nodetool.input` - User input collection
  - `nodetool.output` - Result display

**Strengths:**
- Comprehensive utility nodes (boolean, math, list, dictionary, text)
- Visual workflow primitives (ForEach, If, constant, input, output)
- Type-safe connections via TypedDict and Field annotations

**Recommendations:**
- Continue expanding visual building blocks
- Add more flow control nodes (switch/case, parallel execution)
- Create more workflow template examples

---

### 2. "Your data stays yours. Never leaves without your permission."

**Status:** ⚠️ **PARTIALLY ALIGNED** (6/10)

**Evidence:**

**Local Data Processing (Good):**
- `lib.os` - File system operations (local files)
- `lib.sqlite` - Local database operations
- `lib.excel`, `lib.docx`, `lib.pdfplumber` - Local document processing
- `lib.beautifulsoup` - Local HTML parsing
- `nodetool.audio` - Local audio file operations
- `nodetool.image` - Local image manipulation
- `nodetool.video` - Local video processing

**Cloud Data Transmission (Requires Awareness):**
- `openai.*` - All nodes send data to OpenAI API
- `gemini.*` - All nodes send data to Google Gemini API
- `kie.*` - All nodes send data to Kie.ai API
- `messaging.discord`, `messaging.telegram` - Send data to messaging platforms
- `lib.supabase` - Cloud database operations
- `lib.mail` - Email transmission

**Gaps:**
1. **No explicit privacy warnings** on nodes that transmit data to cloud services
2. **No data locality indicators** to show which nodes keep data local vs. send to cloud
3. **No encryption nodes** for securing data before cloud transmission
4. **No audit trail** for data leaving the local system
5. **No consent prompts** when connecting to cloud service nodes

**Recommendations:**
1. **Add privacy metadata to nodes:** Tag each node with data locality (`local`, `cloud`, `hybrid`)
2. **Create privacy indicator UI:** Visual badge showing if node transmits data externally
3. **Add encryption utilities:** Nodes for encrypting data before cloud transmission
4. **Implement audit logging:** Track when data leaves the local system
5. **Create data sanitization nodes:** Anonymize sensitive data before cloud processing
6. **Add privacy-preserving alternatives:**
   - Local image blur/redaction before OCR
   - Local PII detection and masking
   - Local anonymization utilities

---

### 3. "See every step. Watch workflows execute in real-time."

**Status:** ✅ **WELL ALIGNED** (9/10)

**Evidence:**
- All nodes support streaming output via `gen_process()` generator pattern
- `Chunk` updates for real-time text streaming
- `TaskUpdate`, `PlanningUpdate`, `ToolCallUpdate` for agent visibility
- `EdgeUpdate` for workflow graph progress tracking
- Comprehensive logging via `get_logger(__name__)`

**Examples:**
```python
# Summarizer node streams output
async def gen_process(self, context) -> AsyncGenerator[OutputType, None]:
    async for chunk in stream_text(...):
        yield {"chunk": chunk, "text": None}
    yield {"text": full_text, "chunk": None}
```

**Strengths:**
- Real-time streaming for LLM outputs
- Agent reasoning visibility (planning, tool calls, tasks)
- Incremental progress updates

**Minor Gaps:**
- Some file processing nodes don't stream intermediate progress
- Batch operations lack per-item progress indicators

**Recommendations:**
- Add progress streaming to all long-running nodes (video processing, batch operations)
- Standardize progress percentage reporting
- Add intermediate result visualization for multi-step nodes

---

### 4. "Mix your tools. Combine local AI with cloud services as you need."

**Status:** ✅ **EXCELLENT ALIGNMENT** (10/10)

**Evidence:**

**Local AI Tools:**
- **Ollama LLMs:** Full integration via `langchain-ollama`
  - Agent nodes support Ollama models (Llama3.2, Phi3.5, Mistral, Gemma3, Qwen3, Granite)
  - Local embedding via `OllamaEmbedding` in document nodes
  - `vector.chroma` with Ollama embeddings for local RAG
- **llama.cpp:** GGUF model support for efficient local inference
- **Local document processing:** BeautifulSoup, PyMuPDF, pdfplumber, python-docx
- **Local data analysis:** Seaborn, NumPy, SQLite
- **Local image manipulation:** Pillow filters and operations

**Cloud AI Services:**
- **OpenAI:** GPT models, DALL-E, Whisper transcription
- **Google Gemini:** Text, image, audio, video generation
- **Kie.ai:** Flux 2, Seedream, Z-Image, video generation
- **Search:** Google search integration

**Hybrid Capabilities:**
- Vector databases (Chroma, FAISS) work with both local and cloud embeddings
- Agent nodes seamlessly switch between local (Ollama) and cloud (OpenAI/Anthropic) models
- Document processing can feed into either local or cloud LLMs

**Strengths:**
- Unified `LanguageModel` type allows mixing providers in same workflow
- `Provider` enum supports 35+ providers including OpenAI, Anthropic, Gemini, Ollama, LlamaCpp, Replicate, HuggingFace, and many more
- Agent nodes have `unified_recommended_models()` listing both local and cloud options
- No vendor lock-in - workflows can switch providers without restructuring

**This is the strongest alignment area - NodeTool truly delivers on mixing local and cloud tools.**

---

### 5. "Share your work. Export, deploy, and collaborate."

**Status:** ⚠️ **LIMITED ALIGNMENT** (4/10)

**Evidence:**

**Current Export Capabilities:**
- `nodetool.output` nodes for returning results
- File save nodes (`SaveAudio`, `SaveImage`, `SaveVideo`)
- Example workflows in `src/nodetool/examples/` (JSON format)
- Asset management via `FolderRef` and asset storage

**Gaps (From Manifesto Promises):**
1. **No workflow templates** - Manifesto promises "Ready-to-use examples for common AI tasks"
2. **No sharing infrastructure** - No nodes for exporting workflows
3. **No collaboration features** - No multi-user workflow editing nodes
4. **Limited deployment** - No API deployment nodes in base library
5. **No Mini-App export** - Manifesto mentions "Share as Mini-Apps" but no nodes support this
6. **No version control** - Manifesto roadmap item missing from nodes

**What Exists:**
- Examples directory has 70+ workflow examples (JSON and Python DSL)
- Assets can be saved and loaded via folder references
- Workflows are JSON (shareable format)

**Recommendations:**
1. **Create workflow template nodes:**
   - `LoadWorkflowTemplate` - Import common patterns
   - `SaveWorkflowTemplate` - Export with metadata
2. **Add export nodes:**
   - `ExportToAPI` - Generate API endpoint
   - `ExportToMiniApp` - Package as standalone app
   - `ExportToZip` - Bundle workflow + assets
3. **Version control nodes:**
   - `SaveWorkflowVersion` - Snapshot current state
   - `CompareWorkflowVersions` - Diff between versions
   - `RestoreWorkflowVersion` - Rollback changes
4. **Collaboration nodes:**
   - `ShareWorkflow` - Generate share link
   - `ImportSharedWorkflow` - Load from community
5. **Deploy nodes:**
   - `DeployToCloud` - Push to cloud runtime
   - `CreateWebhook` - Expose workflow as webhook

---

### 6. "Experiment freely. Iterate, undo, perfect—all visible and controllable."

**Status:** ✅ **GOOD ALIGNMENT** (8/10)

**Evidence:**

**Iteration Support:**
- `nodetool.control.ForEach` - Iterate over lists with intermediate results
- Generator pattern allows streaming partial results
- `nodetool.control.If` - Conditional branching for experimentation
- Agent nodes support multi-turn iteration with tool calls

**Visibility:**
- All node processing logged via structured logging
- Real-time chunk updates show intermediate outputs
- Agent planning and tool call updates expose reasoning
- Error messages propagate clearly

**Control:**
- Parameterized nodes allow experimentation without code changes
- Model swapping via `LanguageModel` field
- Configurable temperature, max_tokens, context_window
- Stop sequences and other generation controls

**Gaps:**
1. **No built-in undo mechanism** - Can't rollback within a workflow run
2. **No checkpoint/restore nodes** - Can't save intermediate state and resume
3. **No A/B testing nodes** - Can't easily compare variations
4. **No parameter sweep nodes** - Can't iterate over parameter ranges

**Recommendations:**
1. **Add experiment nodes:**
   - `Checkpoint` - Save intermediate state
   - `RestoreCheckpoint` - Resume from saved state
   - `ABTest` - Run two variants and compare
   - `ParameterSweep` - Iterate over parameter grid
2. **Add comparison nodes:**
   - `CompareOutputs` - Side-by-side result comparison
   - `ScoreOutput` - Evaluate quality metrics
3. **Add debugging nodes:**
   - `Breakpoint` - Pause execution for inspection
   - `LogValue` - Explicit logging of intermediate values
   - `Assert` - Validate assumptions during execution

---

### 7. "Own your infrastructure. Run on your hardware. Keep control."

**Status:** ✅ **STRONG ALIGNMENT** (9/10)

**Evidence:**

**Local Execution:**
- All base nodes run in local Python environment
- No SaaS requirements for core functionality
- Local file system access via `lib.os`
- Local database via `lib.sqlite`
- Local model execution via Ollama and llama.cpp
- Local document processing (no external APIs required)

**Hardware Control:**
- Ollama models run on local GPU/CPU
- llama.cpp GGUF models for resource-constrained hardware
- Context window configuration for memory management
- No forced cloud dependencies

**Deployment Flexibility:**
- Can run entirely offline with local models
- Cloud services are optional add-ons
- No telemetry or phone-home in base nodes
- Self-hosted deployment model

**Gaps (Per Manifesto Claims):**
1. **Missing native local Flux/SDXL nodes** - Manifesto claims "Local models (Flux/SDXL)" but:
   - Flux only available via Kie.ai cloud API (`kie.image.*`) for direct node usage
   - **However**, ComfyUI integration exists (`comfy_local`, `comfy_runpod` providers) which CAN run Flux/SDXL locally
   - Gap: Native Diffusers-based nodes would provide easier local access without ComfyUI setup
2. **Missing native local Whisper** - Manifesto claims local Whisper but:
   - Whisper only via OpenAI API (`openai.audio.Transcribe`)
   - No native local speech-to-text nodes
   - No local audio transcription
3. **Limited native local video** - Video generation only via cloud (Gemini, Kie)
4. **No infrastructure management nodes:**
   - No GPU monitoring/management
   - No local model management
   - No resource allocation controls

**Recommendations:**
1. **Add local image generation:**
   - Create native `LocalFluxGeneration` node using Diffusers library
   - Create native `LocalSDXLGeneration` node using Diffusers library
   - Support Flux.1-dev, SDXL, SD 1.5
   - Allow local LoRA loading
   - Note: ComfyUI integration already exists as alternative (`comfy_local` provider)
2. **Add local audio transcription:**
   - Create `LocalWhisper` node using openai-whisper or faster-whisper
   - Support offline transcription
   - Multiple model sizes for speed/accuracy trade-off
3. **Add infrastructure nodes:**
   - `CheckGPUMemory` - Monitor VRAM usage
   - `ListLocalModels` - Show available local models
   - `DownloadModel` - Pull models from HuggingFace
   - `UnloadModel` - Free memory
4. **Add local video generation:**
   - Integrate AnimateDiff or similar for local video
   - Frame interpolation nodes
   - Local video effects

---

## Gap Analysis Summary

### Critical Gaps (Must Address)

1. **Local Image Generation** - Manifesto promises Flux/SDXL but only cloud APIs exist
2. **Local Audio Transcription** - Manifesto promises Whisper but only OpenAI API exists
3. **Privacy Controls** - No explicit data locality warnings or encryption utilities
4. **Workflow Templates** - Roadmap item completely missing
5. **Batch Processing** - Roadmap item missing

### Important Gaps (Should Address)

6. **Version Control** - Roadmap item missing
7. **Export/Deployment** - Limited sharing and deployment options
8. **Local Video Generation** - Only cloud options available
9. **Experiment Utilities** - No A/B testing, parameter sweep, checkpointing
10. **Infrastructure Management** - No GPU monitoring or model management

### Nice-to-Have Gaps (Could Address)

11. **Privacy Visualization** - UI indicators for data locality
12. **Audit Logging** - Track data leaving local system
13. **Comparison Tools** - Compare workflow outputs
14. **Debugging Nodes** - Breakpoints, assertions, explicit logging

---

## Recommendations by Priority

### Phase 1: Critical Manifesto Alignment (Immediate)

1. **Implement Local Image Generation**
   ```python
   class LocalFluxGeneration(BaseNode):
       """Generate images locally using Flux.1-dev via Diffusers."""
       # Uses local GPU, no cloud API
   
   class LocalSDXLGeneration(BaseNode):
       """Generate images locally using Stable Diffusion XL."""
       # Uses local GPU, no cloud API
   ```

2. **Implement Local Audio Transcription**
   ```python
   class LocalWhisper(BaseNode):
       """Transcribe audio locally using Faster-Whisper."""
       # Offline transcription, no cloud API
       # Support tiny, base, small, medium, large models
   ```

3. **Add Privacy Metadata System**
   ```python
   # Add to BaseNode metadata
   class NodePrivacy(Enum):
       LOCAL = "local"  # Data never leaves machine
       CLOUD = "cloud"  # Data sent to external API
       HYBRID = "hybrid"  # Can use local or cloud
   
   # Each node declares privacy level
   class OpenAIChat(BaseNode):
       _privacy_level = NodePrivacy.CLOUD
       _data_sent_to = ["OpenAI API"]
   ```

4. **Create Workflow Template System**
   ```python
   class LoadWorkflowTemplate(BaseNode):
       """Load a pre-built workflow template."""
       template: str = Field(description="Template name")
   
   # Templates directory: src/nodetool/templates/
   # - image_generation_pipeline.json
   # - document_analysis.json
   # - video_processing.json
   ```

### Phase 2: Roadmap Features (Next Quarter)

5. **Implement Version Control Nodes**
   - SaveWorkflowVersion, CompareVersions, RestoreVersion

6. **Implement Batch Processing**
   - BatchProcess, ParallelExecute, CollectResults

7. **Add Export/Deployment Nodes**
   - ExportToAPI, ExportToMiniApp, DeployToCloud

8. **Create Community Hub Integration**
   - ShareWorkflow, BrowseTemplates, ImportFromCommunity

### Phase 3: Enhanced Privacy & Control (Future)

9. **Build Privacy Protection Suite**
   - DetectPII, MaskSensitiveData, EncryptBeforeCloud, LocalOnlyGuard

10. **Add Infrastructure Management**
    - CheckGPUMemory, ManageLocalModels, OptimizePerformance

11. **Create Experiment Framework**
    - ABTest, ParameterSweep, Checkpoint, CompareOutputs

---

## Node Inventory by Manifesto Principle

### Privacy-First (Local Processing) ✅

**File & Document Processing:**
- lib.os (file operations)
- lib.sqlite (local database)
- lib.excel (spreadsheet processing)
- lib.docx (Word documents)
- lib.pdfplumber, lib.pymupdf (PDF processing)
- lib.markdown, lib.pandoc (document conversion)
- lib.beautifulsoup (HTML parsing)

**Media Processing:**
- nodetool.audio (load, save, normalize)
- nodetool.image (crop, resize, save)
- nodetool.video (edit, save)
- lib.pillow.* (image filters and effects)
- lib.numpy.* (numerical operations)

**Data Manipulation:**
- nodetool.boolean (logic operators)
- nodetool.numbers (arithmetic)
- nodetool.list (list operations)
- nodetool.dictionary (key-value operations)
- nodetool.text (string manipulation)
- nodetool.data (data transforms)
- lib.json (JSON parsing)
- lib.date (date/time operations)
- lib.math (mathematical functions)
- lib.uuid (ID generation)

**Local AI:**
- nodetool.agents.* (with Ollama models)
- nodetool.document.* (with Ollama embeddings)
- vector.chroma (with Ollama embeddings)
- vector.faiss (local vector search)
- ComfyUI integration (comfy_local, comfy_runpod providers) for local Stable Diffusion/SDXL/Flux

### Cloud Services (User Choice) ✅

**Text Generation:**
- openai.text.Chat
- openai.agents.* (GPT-4, GPT-3.5)
- gemini.text.GroundedSearch
- gemini.text.Chat

**Image Generation:**
- openai.image.GenerateImage (DALL-E)
- gemini.image.ImageGeneration (Imagen)
- kie.image.* (Flux2Pro, Flux2Flex, FluxKontext, Seedream, ZImage, NanoBanana, TopazUpscale, GrokImagine)

**Audio Processing:**
- openai.audio.Transcribe (Whisper)
- openai.audio.GenerateSpeech (TTS)
- gemini.audio.TextToSpeech
- kie.audio.* (TTS, music generation)

**Video Generation:**
- gemini.video.* (Veo models)
- kie.video.* (video generation)

**External Services:**
- messaging.discord (Discord bot)
- messaging.telegram (Telegram bot)
- lib.supabase (cloud database)
- lib.mail (email)
- search.google (web search)

### Visual Workflow Building ✅

- nodetool.constant (values)
- nodetool.input (user input)
- nodetool.output (results)
- nodetool.control (if/else, ForEach)
- nodetool.compare (comparisons)

### Experimentation ✅

- nodetool.code (Python eval)
- nodetool.generators.* (dynamic content)
- nodetool.workspace (workspace operations)
- nodetool.triggers (event handling)

---

## Conclusion

NodeTool's base nodes offering demonstrates **strong alignment** with most manifesto principles, particularly in:
- Mixing local and cloud AI tools
- Visual workflow building
- Real-time visibility
- Local infrastructure control

However, **critical gaps** exist where the manifesto makes specific promises:
1. **Local Flux/SDXL** - Promised but only cloud APIs available
2. **Local Whisper** - Promised but only OpenAI API available
3. **Privacy controls** - Implicit but not explicit
4. **Workflow templates** - Roadmap item not implemented
5. **Version control** - Roadmap item not implemented
6. **Batch processing** - Roadmap item not implemented

### Recommended Action Plan

**Immediate (1-2 months):**
1. Implement local image generation (Flux.1-dev, SDXL) via Diffusers
2. Implement local Whisper transcription via faster-whisper
3. Add privacy metadata and documentation to all nodes

**Near-term (3-6 months):**
4. Create workflow template library and loader nodes
5. Implement version control nodes
6. Implement batch processing framework

**Long-term (6-12 months):**
7. Build comprehensive privacy protection suite
8. Create community hub integration
9. Add infrastructure management nodes

By addressing these gaps, NodeTool can achieve **full manifesto alignment** and deliver on all stated promises to users.

---

**Last Updated:** December 29, 2025  
**Evaluator:** NodeTool Development Team  
**Next Review:** Q2 2026
