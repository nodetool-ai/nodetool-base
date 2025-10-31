"""
Language Practice Feedback DSL Example

Provide pronunciation and fluency feedback for spoken language practice recordings.

Workflow:
1. **Audio Input** - Load a sample speaking practice recording
2. **Automatic Speech Recognition** - Transcribe the learner's speech to text
3. **Coach Prompt Builder** - Prepare coaching instructions that include the transcript
4. **Language Coach Agent** - Generate actionable feedback and improvement tips
5. **String Output** - Display the feedback for the learner
"""

import os

from nodetool.dsl.graph import create_graph, run_graph
from nodetool.dsl.nodetool.input import AudioInput
from nodetool.dsl.nodetool.text import AutomaticSpeechRecognition, FormatText
from nodetool.dsl.nodetool.agents import Agent
from nodetool.dsl.nodetool.output import StringOutput
from nodetool.metadata.types import ASRModel, AudioRef, LanguageModel, Provider

sample_audio_file = os.path.join(os.path.dirname(__file__), "harvard.mp3")

# Learner provides a speaking practice recording
practice_audio = AudioInput(
    name="practice_audio",
    description="Recording of a learner practicing a language passage",
    value=AudioRef(
        uri=f"file://{sample_audio_file}",
        type="audio",
    ),
)

# Convert speech to text using Whisper-compatible model
transcription = AutomaticSpeechRecognition(
    audio=practice_audio.output,
    model=ASRModel(
        type="asr_model",
        provider=Provider.OpenAI,
        id="gpt-4o-mini-transcribe",
    ),
)

# Build a coaching prompt that references the transcript
coach_prompt = FormatText(
    template="""You are a friendly language learning coach. A learner just practiced reading aloud.

Transcript of the learner's recording:
{{ transcript }}

Provide feedback with three sections:
1. Overall impression (tone, confidence, pacing)
2. Pronunciation corrections (list specific words)
3. Suggested practice tips for the next session
""",
    transcript=transcription.out.text,
)

# Generate feedback using an LLM agent
coach_feedback = Agent(
    model=LanguageModel(
        type="language_model",
        provider=Provider.OpenAI,
        id="gpt-4o-mini",
    ),
    system="You are an encouraging speaking coach who gives concise, actionable advice.",
    prompt=coach_prompt.output,
)

# Present the coaching feedback
feedback_output = StringOutput(
    name="feedback",
    value=coach_feedback.out.text,
)

# Create the graph
graph = create_graph(feedback_output)


if __name__ == "__main__":
    result = run_graph(graph)
    print("Coaching feedback:\n", result["feedback"])
