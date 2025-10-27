"""
Meeting Transcript Summarizer DSL Example

Automatically transcribe a meeting recording and generate concise notes.

Workflow:
1. **Audio Input** - Load meeting recording
2. **Automatic Speech Recognition** - Transcribe audio using Whisper model
3. **Summarizer** - Generate concise summary from the transcript
4. **String Output** - Display the summary
"""

from nodetool.dsl.graph import AssetOutputMode, graph_result
from nodetool.dsl.nodetool.constant import Audio
from nodetool.dsl.nodetool.text import AutomaticSpeechRecognition
from nodetool.dsl.nodetool.agents import Summarizer
from nodetool.dsl.nodetool.output import StringOutput
from nodetool.metadata.types import ASRModel, AudioRef, LanguageModel, Provider


async def example():
    """
    Transcribe a meeting recording and summarize it.
    """
    # Load meeting audio
    audio_input = Audio(
        value=AudioRef(
            uri="https://app.nodetool.ai/examples/remove_silence.mp3",
            type="audio",
        )
    )

    # Transcribe audio using Whisper
    transcription = AutomaticSpeechRecognition(
        audio=audio_input.output,
        model=ASRModel(
            type="asr_model",
            provider=Provider.HuggingFaceFalAI,
            id="openai/whisper-large-v3",
        ),
    )

    # Summarize the transcript
    summary = Summarizer(
        text=transcription.out.text,
        model=LanguageModel(
            type="language_model",
            id="openai/gpt-oss-120b",
            provider=Provider.HuggingFaceCerebras
        ),
    )

    # Output the summary
    output = StringOutput(
        name="summary",
        value=summary.out.text,
    )

    result = await graph_result(output, asset_output_mode=AssetOutputMode.WORKSPACE)
    return result


if __name__ == "__main__":
    import asyncio

    result = asyncio.run(example())
    print(f"Meeting summary: {result}")
