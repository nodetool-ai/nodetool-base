# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 2025-12-20

### Added

- DataframeInput node for data processing workflows
- ToString text node for type conversion
- Text processing nodes for comparison, transformation, and normalization
- Experimental agent templates
- Agent thread node for conversation management
- Agent examples and DSL templates
- Research agent DSL example with autonomous web research
- Text-to-video campaign DSL example
- Game encounter planner DSL example
- Supabase DSL examples and CRUD node implementations
- New node packages for Google Gemini, OpenAI, and common library operations
- ImageToVideo node for video generation from images
- TextToVideo node for generating videos from text prompts
- ResearchAgent node for autonomous web-based research
- StructuredOutputGenerator for generating structured JSON objects
- Document and Excel processing nodes
- Comprehensive test dependency mocking

### Changed

- Updated metadata across multiple nodes
- Enhanced Gemini audio voice options with new names and case-insensitive validation
- Removed `await` from `provider.get_client()` calls in Gemini nodes
- Refactored RSS feed data access
- Enhanced provider retrieval in Agent class
- Updated agent configurations and model definitions
- Enhanced mathematical operation nodes with operator symbols tags
- Updated GoogleImages class with improved type annotations
- Streamlined imports and enhanced node processing
- Enhanced OCR result processing in PaddleOCRNode
- Removed sentence_transformers dependency
- Updated Playwright to run without Docker
- Improved research agent implementation

### Fixed

- Import errors and linting issues
- Test and default values
- Extract text from PDF functionality
- Forgiving join operations
- Bogus files and init files cleanup

### Removed

- RealtimeWhisper node (replaced with improved implementation)
- Unused environment variable nodes
- Unused nodes and streamlined agent configurations
- Simple Chat example files
- HuggingFace inference node implementations (moved to dedicated package)
- Leann node classes and related dependencies

## [September 2025]

### Added

- StructuredOutputGenerator for JSON objects
- Pattern matching functionality for image and input nodes
- Agent classes with image and audio support
- Streaming output support for multiple nodes
- ExecuteLua node for Lua code execution
- SaveTextFile node for file operations
- DataGenerator with streaming support
- RealtimeAgent class for streaming responses
- Browser and Screenshot nodes with Docker-based Playwright
- WorkspaceDirectory node
- Bash, JavaScript, and Ruby execution nodes with Docker

### Changed

- Refactored Agent class for improved output handling and type safety
- Updated to use ClassVar for tool exposure flags
- Enhanced node execution with execution mode support
- Refactored code nodes to remove dynamic outputs
- Refactored to use TypedDict for output types
- Replaced FilePath and FolderPath with string types
- Updated Python version requirement to ^3.11

### Fixed

- Test coverage and async operations
- File handling in various nodes
- Output validation in test cases

### Removed

- Deprecated MLX and WhisperCpp nodes
- Redundant ListIterator node
- Deprecated dependencies

## [May-August 2025]

### Added

- Calendly API integration nodes
- Collection node with enhanced functionality
- Evaluate Expression and Execute Python nodes
- Multiple Python standard library node wrappers:
  - textwrap nodes for text formatting
  - random utility nodes
  - zlib compression nodes
  - hashlib for cryptographic hashing
  - base64 encode/decode nodes
  - ftplib nodes for FTP operations
  - urllib helper nodes
  - gzip compression nodes
  - tarfile nodes for archive handling
  - difflib utility nodes
  - HTML utility nodes
- SendEmail node for email operations
- Meeting Transcript Summarizer workflow
- Daily Digest workflow
- Enhanced agent descriptions and documentation
- Gemini Imagen, Veo, TTS integration
- IRIS-based evaluation framework

### Changed

- Transitioned from Poetry to uv for dependencies
- Refactored evaluation agents and scripts
- Enhanced math operation descriptions
- Updated agent configurations
- Improved asset loading consistency across nodes
- Refactored metadata extraction in Browser class
- Enhanced ChromaNode classes for async operations

### Fixed

- Multiple test case improvements
- Date processing to use UTC
- ChatInput test expectations
- Browser environment variable handling
- Test assertions for result checks

### Removed

- Librosa dependency and related nodes
- Deprecated files and node types
- Color Boost Video example
- Email Classifier example
- Paper2Podcast example

## [January 2025]

### Added

- SKLearn nodes for data processing and machine learning workflows
- Jinja templating support for MapTemplate nodes
- LangChain and LlamaIndex text splitting nodes
- Gmail label functionality for email processing
- Deepseek model recommendations
- FAL AI integration nodes (LLM, text-to-audio, video models)
- Elevenlabs integration nodes
- Ollama embedding function with configurable timeout
- PDF extraction capabilities
- Document and DocumentOutput nodes

### Changed

- Moved file nodes to nodetool namespace (removed from lib.file)
- Moved image processing nodes from lib.image.pillow to nodetool.image namespace
- Refactored Chroma and SKLearn dataset nodes
- Updated agent configurations and model recommendations
- Enhanced Collection indexing with semantic splitting

### Fixed

- Test coverage improvements
- Image transform module imports and paths

### Removed

- QuestionAnswerAgent node
- Text splitting nodes (replaced with LangChain/LlamaIndex versions)
- IndexRequest functionality

## [October-December 2024]

### Added

- Initial node library development
- Basic processing nodes
- Foundation workflows

### Changed

- Node organization and structure
- Initial development iterations

### Fixed

- Early bug fixes and improvements

## [February-April 2025]

### Added

- Foundation node library
- Core processing nodes for:
  - Audio processing
  - Video processing
  - Image processing
  - Text processing
- Initial example workflows
- LLM nodes (later refactored to Agent nodes)
- Chroma vector database integration
- Index PDFs workflow

### Changed

- Initial node architecture setup
- Core processing implementations

### Fixed

- Initial bug fixes and improvements

## [September 2024]

### Added

- Initial comprehensive node library
- Audio, video, image, and text processing nodes
- Agent system integration
- Core node classes and base implementations

### Changed

- Core node architecture improvements
- Initial development setup
