"""
Transcribe Audio DSL Example

Convert speech to text using Whisper model with word-level timestamps.

Workflow:
1. **Audio Input** - Record your voice or upload an audio file
2. **Automatic Speech Recognition** - Processes the audio through Whisper model
3. **String Output** - Displays the transcribed text
"""

from nodetool.dsl.graph import graph_result
from nodetool.dsl.nodetool.input import AudioInput
from nodetool.dsl.nodetool.text import AutomaticSpeechRecognition
from nodetool.dsl.nodetool.output import StringOutput
from nodetool.metadata.types import LanguageModel


async def example():
    """
    Transcribe audio using OpenAI's Whisper model.
    """
    output = StringOutput(
        name="transcription",
        value=AutomaticSpeechRecognition(
            audio=AudioInput(
                name="audio",
                description="",
                value={},
            ),
            model=LanguageModel(
                type="asr_model",
                id="whisper-1",
                provider="openai",
                name="Whisper",
            ),
        ),
    )

    result = await graph_result(output)
    return result


if __name__ == "__main__":
    import asyncio

    result = asyncio.run(example())
    print(f"Transcription: {result}")
